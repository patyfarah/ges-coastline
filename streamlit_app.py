# -------------------------------------------------------
# Libraries
# -------------------------------------------------------
import streamlit as st
import ee
import geemap.foliumap as geemap
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import os
import gc

# -------------------------------------------------------
# Earth Engine Initialization
# -------------------------------------------------------
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["earthengine"]),
    scopes=['https://www.googleapis.com/auth/earthengine']
)
ee.Initialize(credentials)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
NDVI_PRODUCTS = {"MOD13A1": ee.ImageCollection("MODIS/061/MOD13A1")}
LST_PRODUCTS = {"MOD11A1": ee.ImageCollection("MODIS/061/MOD11A1")}

GES_PALETTE = ['#a50026', '#f88d52', '#ffffbf', '#86cb66', '#006837']
GES_CLASSES = {
    'Very Severe': (-100, -25),
    'Severe': (-25, -5),
    'No Change': (-5, 5),
    'Good Environmental': (5, 25),
    'Excellent Improvement': (25, float('inf'))
}
VIS_PARAMS = {
    'bands': ['GES'],
    'palette': GES_PALETTE,
    'min': -50,
    'max': 50,
    'labels': list(GES_CLASSES.keys())
}

# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------
def mask_ndvi(image):
    return image.updateMask(image.select('SummaryQA').lte(1))

def mask_lst(image):
    qc = image.select('QC_Day')
    mask = qc.bitwiseAnd(3).lte(1)
    lst = image.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    return lst.updateMask(mask).updateMask(lst.gte(-20).And(lst.lte(50)))

def get_image_collection(products, key, region, start, end, mask_fn=None):
    col = products[key].filterBounds(region).filterDate(start, end)
    return col.map(mask_fn) if mask_fn else col

def return_intersect(country, buffer_km):
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    region = countries.filter(ee.Filter.eq('country_na', country)).geometry()
    inner = region.buffer(-buffer_km * 1000)
    outer = region.difference(inner)
    coast = ee.FeatureCollection('projects/ee-project-457404/assets/coastlines') \
        .filterBounds(region).geometry().buffer(buffer_km * 1000)
    return outer.intersection(coast), region

def get_ges(region, year):
    start, end = f"{year}-01-01", f"{year}-12-31"
    ndvi = get_image_collection(NDVI_PRODUCTS, "MOD13A1", region, start, end, mask_ndvi).median().select('NDVI').multiply(0.0001)
    lst = get_image_collection(LST_PRODUCTS, "MOD11A1", region, start, end, mask_lst).median()

    ndvi = ndvi.clip(region)
    lst = lst.unmask().focal_mean(radius=1, units='pixels', iterations=1).clip(region)

    ndvi_stats = ndvi.reduceRegion(ee.Reducer.minMax(), region, 1000, maxPixels=1e13)
    lst_stats = lst.reduceRegion(ee.Reducer.minMax(), region, 1000, maxPixels=1e13)

    ndvi_norm = ndvi.subtract(ndvi_stats.get('NDVI_min')).divide(
        ee.Number(ndvi_stats.get('NDVI_max')).subtract(ndvi_stats.get('NDVI_min'))).multiply(100)
    lst_norm = lst.subtract(lst_stats.get('LST_Day_1km_min')).divide(
        ee.Number(lst_stats.get('LST_Day_1km_max')).subtract(lst_stats.get('LST_Day_1km_min'))).multiply(100).subtract(100)

    return ndvi_norm.multiply(0.5).add(lst_norm.multiply(0.5)).rename('GES')

def plot_ges_bar(image):
    counts = {}
    for label, (low, high) in GES_CLASSES.items():
        mask = image.gte(low) if high == float('inf') else image.gte(low).And(image.lt(high))
        count = image.updateMask(mask).reduceRegion(ee.Reducer.count(), scale=1000, maxPixels=1e13).get('GES').getInfo()
        counts[label] = count

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.keys(), counts.values(), color=GES_PALETTE)
    ax.set_title("GES Change Classification")
    ax.set_xlabel("Class")
    ax.set_ylabel("Pixel Count")
    st.pyplot(fig)

def export_if_needed(image, region, filename, key):
    if key not in st.session_state.get("exported_files", {}) or not os.path.exists(filename):
        geemap.ee_export_image(image, filename=filename, region=region, scale=1000)
        st.session_state["exported_files"][key] = filename

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("🌍 GES Mapping Tool")
with st.sidebar:
    st.header("Configuration")
    country = st.selectbox("Country", ["Morocco", "Algeria", "Tunisia", "Libya", "Egypt", "Syria", "Lebanon", "Yemen", "Mauritania"], index=6)
    start_year = st.number_input("Start Year", 2000, 2030, 2002)
    end_year = st.number_input("End Year", 2000, 2030, 2022)
    buffer_km = st.slider("Coast Buffer (km)", 1, 10, 5)

if st.button("Run Analysis"):
    st.info("Running analysis, please wait...")
    try:
        inter, region = return_intersect(country, buffer_km)
        GES_start = get_ges(inter, start_year)
        GES_end = get_ges(inter, end_year)
        GES_diff = GES_end.subtract(GES_start)

        st.session_state.update({
            "analysis_done": True,
            "region": region,
            "intersection": inter,
            "GES_first": GES_start,
            "GES_last": GES_end,
            "GES_diff": GES_diff,
            "exported_files": {}
        })

        for key, img, fname in zip(
            ["GES_diff", "GES_first", "GES_last"],
            [GES_diff, GES_start, GES_end],
            ["ges-change.tif", "ges-first.tif", "ges-last.tif"]
        ):
            export_if_needed(img, inter, fname, key)

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------------------------------
# Output Display
# -------------------------------------------------------
if st.session_state.get("analysis_done"):
    st.subheader("🗺️ GES Map")
    m = geemap.Map()
    m.centerObject(st.session_state["region"], 6)
    m.addLayer(st.session_state["GES_first"], VIS_PARAMS, "GES Start Year", shown=False)
    m.addLayer(st.session_state["GES_last"], VIS_PARAMS, "GES End Year", shown=False)
    m.addLayer(st.session_state["GES_diff"], VIS_PARAMS, "GES Change")
    m.add_legend(title="GES Classification", legend_dict=dict(zip(VIS_PARAMS['labels'], VIS_PARAMS['palette'])))
    m.to_streamlit(height=600)

    st.subheader("📊 GES Classification Chart")
    plot_ges_bar(st.session_state["GES_diff"])

    st.subheader("📥 Downloads")
    for key, label in zip(
        ["GES_diff", "GES_first", "GES_last"],
        ["Download GES Change", "Download GES Start Year", "Download GES End Year"]
    ):
        filepath = st.session_state["exported_files"].get(key)
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                st.download_button(label=label, data=f, file_name=filepath, mime="image/tiff", key=f"dl-{key}")

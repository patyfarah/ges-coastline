#-------------------------------------------------------
# Libraries
#-------------------------------------------------------
import streamlit as st
import geopandas as gpd
import ee
import geemap.foliumap as geemap
import folium
from google.oauth2 import service_account
import gc

#--------------------------------------------------------
# Initialization
#--------------------------------------------------------
# Load service account info from Streamlit secrets
service_account_info = dict(st.secrets["earthengine"])

SCOPES = ['https://www.googleapis.com/auth/earthengine']

# Create Google credentials object
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES)

# Initialize Earth Engine
ee.Initialize(credentials)

# Visualization Params
vis_params = {
    'bands': ['NDVI'],
    'palette': ['#a50026', '#f88d52', '#ffffbf', '#86cb66', '#006837'],
    'min': -50.0,
    'max': 50.0
}

lst_params = {
    'bands': ['LST_Day_1km'],
    'palette': ['#053061', '#6bacd0', '#f7f7f7', '#e58268', '#67001f'],
    'min': -50,
    'max': 50
}

ges_params = {
    'bands': ['GES'],
    'palette': ['#a50026', '#f88d52', '#ffffbf', '#86cb66', '#006837'],
    'min': -50,
    'max': 50,
    'labels': ['Very Severe', 'Severe', 'No Change', 'Good Environmental', 'Excellent Improvement']
}

NDVI_PRODUCTS = {"MOD13A1": ee.ImageCollection("MODIS/061/MOD13A1")}
LST_PRODUCTS = {"MOD11A1": ee.ImageCollection("MODIS/061/MOD11A1")}

def mask_ndvi(image):
    qa = image.select('SummaryQA')
    mask = qa.lte(1)
    return image.updateMask(mask)

def mask_lst(image):
    qc = image.select('QC_Day')
    quality_mask = qc.bitwiseAnd(3).lte(1)
    lst = image.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    lst = lst.updateMask(quality_mask)
    lst = lst.updateMask(lst.gte(-50).And(lst.lte(70)))
    return lst.copyProperties(image, image.propertyNames())

def get_image_collection(collection_dict, product, region, start_date, end_date, mask_function=None):
    collection = collection_dict[product].filterBounds(region).filterDate(start_date, end_date)
    if mask_function:
        collection = collection.map(mask_function)
    return collection

def return_intersect(country, buffer_dist_km):
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    filtered = countries.filter(ee.Filter.eq('country_na', country))
    region = filtered.geometry()
    buffered = region.buffer(-buffer_dist_km * 1000)
    outer_band = region.difference(buffered)
    asset_id ='projects/ee-project-457404/assets/coastlines'    
    ee_fc = ee.FeatureCollection(asset_id).filterBounds(region)
    coastline = ee_fc.geometry()
    coastline_buffer = coastline.buffer(buffer_dist_km * 1000)
    intersection = outer_band.intersection(coastline_buffer)
    del countries, buffered, outer_band, asset_id,ee_fc,coastline, coastline_buffer
    return intersection, region, filtered

def get_ges(intersection, year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    ndvi = get_image_collection(NDVI_PRODUCTS, "MOD13A1", intersection, start_date, end_date, mask_ndvi)
    lst = get_image_collection(LST_PRODUCTS, "MOD11A1", intersection, start_date, end_date, mask_lst)

    ndvi_median = ndvi.median().select('NDVI').multiply(0.0001)
    lst_temp = lst.median()

    ndvi_mean = ndvi_median.clip(intersection)
    lst_mean = lst_temp.unmask().focal_mean(radius=1, units='pixels', iterations=1).clip(intersection)

    ndvi_minmax = ndvi_mean.reduceRegion(ee.Reducer.minMax(), intersection, 1000, maxPixels=1e13)
    lst_minmax = lst_mean.reduceRegion(ee.Reducer.minMax(), intersection, 1000, maxPixels=1e13)

    ndvi_min = ee.Number(ndvi_minmax.get('NDVI_min'))
    ndvi_max = ee.Number(ndvi_minmax.get('NDVI_max'))
    lst_min = ee.Number(lst_minmax.get('LST_Day_1km_min'))
    lst_max = ee.Number(lst_minmax.get('LST_Day_1km_max'))

    ndvi_normal = (ndvi_mean.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min))).multiply(100)
    lst_normal = (lst_mean.subtract(lst_min).divide(lst_max.subtract(lst_min))).multiply(100).subtract(100)

    GES = ndvi_normal.multiply(0.5).add(lst_normal.multiply(0.5)).rename('GES')
    del ndvi,lst,ndvi_mean,lst_mean,ndvi_minmax,lst_minmax,ndvi_min,ndvi_max,lst_min,lst_max,ndvi_normal, lst_normal
    gc.collect()
    return GES

# --- Streamlit UI --- #
st.title("🌍 Good Environmental Status (GES) Mapping Tool")

# Main content (not in sidebar)
st.markdown("### GES Analysis Results")
st.markdown("The map below shows the Good Environmental Status (GES) for the selected country.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    country = st.selectbox("Select Country", ["Morocco", "Algeria", "Tunisia", "Libya", "Egypt",
                                              "Syria", "Lebanon", "Yemen", "Mauritania"],
                            index=6
                          )
    start_year = st.number_input("Start Year", min_value=2000, max_value=2030, value=2002)
    end_year = st.number_input("End Year", min_value=2000, max_value=2030, value=2022)
    buffer_km = st.slider("Coast Buffer (km)", 1, 10, 5)

if st.button("Run Analysis"):
    st.info("Processing... Please wait a few moments.")
    intersection, region, filtered = return_intersect(country, buffer_km)
    GES_first = get_ges(intersection, start_year)
    GES_last = get_ges(intersection, end_year)
    GES_diff = GES_last.subtract(GES_first)
    
    
    # Create and display the map below the title
    m = geemap.Map()
    m.centerObject(region, 8)
    m.addLayer(GES_first, ges_params, "GES Start Year",shown=False)
    m.addLayer(GES_last, ges_params, "GES End Year",shown=False)
    m.addLayer(GES_diff, ges_params, "GES Change")
    m.addLayer(filtered.style(**{"color": "black", "fillColor": "#00000000", "width": 2}), {}, "Border")
    m.add_legend(title="GES Classification", legend_dict=dict(zip(ges_params['labels'], ges_params['palette'])))
    m.to_streamlit(height=600)
    
    gc.collect()
    
    

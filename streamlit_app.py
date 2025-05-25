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
import matplotlib.pyplot as plt
import numpy as np
import time
import os

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
ges_params1 = {
    'bands': ['GES'],
    'palette': ['#a50026', '#f88d52', '#ffffbf', '#86cb66', '#006837'],
    'min': -50,
    'max': 50,
    'labels': ['Very Severe', 'Severe', 'No Change', 'Good Environmental', 'Excellent Improvement']
}

# Define the class labels and corresponding ranges for GES_diff
ges_params = {
    'Very Severe': (-100, -25),
    'Severe': (-25, -5),
    'No Change': (-5, 5),
    'Good Envionmental': (5, 25),
    'Excellent improvement': (25, float('inf'))
}

# Define the palette separately so it's accessible for the bar chart color
ges_palette = ['#a50026', '#f88d52', '#ffffbf', '#86cb66', '#006837']

NDVI_PRODUCTS = {"MOD13A1": ee.ImageCollection("MODIS/061/MOD13A1")}
LST_PRODUCTS = {"MOD11A1": ee.ImageCollection("MODIS/061/MOD11A1")}

# Functions for NDVI and LST masking
def mask_ndvi(image):
    qa = image.select('SummaryQA')
    mask = qa.lte(1)
    return image.updateMask(mask)

def mask_lst(image):
    qc = image.select('QC_Day')
    quality_mask = qc.bitwiseAnd(3).lte(1)
    lst = image.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    lst = lst.updateMask(quality_mask)
    lst = lst.updateMask(lst.gte(-20).And(lst.lte(50)))
    return lst.copyProperties(image, image.propertyNames())

# Function to handle Earth Engine image collection fetching
def get_image_collection(collection_dict, product, region, start_date, end_date, mask_function=None):
    collection = collection_dict[product].filterBounds(region).filterDate(start_date, end_date)
    if mask_function:
        collection = collection.map(mask_function)
    return collection

# Function to get intersection and region geometry
def return_intersect(country, buffer_dist_km):
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    filtered = countries.filter(ee.Filter.eq('country_na', country))
    region = filtered.geometry()
    buffered = region.buffer(-buffer_dist_km * 1000)
    outer_band = region.difference(buffered)
    asset_id = 'projects/ee-project-457404/assets/coastlines'    
    ee_fc = ee.FeatureCollection(asset_id).filterBounds(region)
    coastline = ee_fc.geometry()
    coastline_buffer = coastline.buffer(buffer_dist_km * 1000)
    intersection = outer_band.intersection(coastline_buffer)
    
    # Release unnecessary variables
    del countries, buffered, outer_band, asset_id, ee_fc, coastline, coastline_buffer
    gc.collect()
    
    return intersection, region, filtered

# Function to compute GES for a given year
def get_ges(intersection, year):
    try:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        ndvi = get_image_collection(NDVI_PRODUCTS, "MOD13A1", intersection, start_date, end_date, mask_ndvi)
        lst = get_image_collection(LST_PRODUCTS, "MOD11A1", intersection, start_date, end_date, mask_lst)

        # Reduce collections to median and normalize
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

        # Normalize the NDVI and LST
        ndvi_normal = (ndvi_mean.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min))).multiply(100)
        lst_normal = (lst_mean.subtract(lst_min).divide(lst_max.subtract(lst_min))).multiply(100).subtract(100)

        # Calculate GES
        GES = ndvi_normal.multiply(0.5).add(lst_normal.multiply(0.5)).rename('GES')
        
        # Release unnecessary variables
        del ndvi, lst, ndvi_mean, lst_mean, ndvi_minmax, lst_minmax, ndvi_min, ndvi_max, lst_min, lst_max, ndvi_normal, lst_normal
        gc.collect()
        
        return GES

    except ee.EEException as e:
        error_message = str(e)
        if "out of memory" in error_message.lower() or "memory" in error_message.lower():
            raise MemoryError("The operation exceeded the memory limit. Please try selecting a smaller area or a shorter time range.")
        elif "timeout" in error_message.lower():
            raise TimeoutError("The operation timed out. Please try again with a smaller area or shorter time range.")
        else:
            raise

# Function to process and display the GES classification
def process_and_display(image):
    try:
        GES_first = image
        
        # Calculate the number of pixels in each class
        class_counts = {}
        for class_name, (lower, upper) in ges_params.items():
            if upper == float('inf'):
                class_mask = GES_first.gte(lower)
            else:
                class_mask = GES_first.gte(lower).And(GES_first.lt(upper))
            
            count = GES_first.updateMask(class_mask).reduceRegion(
                reducer=ee.Reducer.count(),
                scale=1000,
                maxPixels=1e13
            ).get('GES').getInfo()
            class_counts[class_name] = count
        
        # Extract class names and counts for plotting
        class_names = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Map class names to colors for the bar chart
        colors = [ges_palette[list(ges_params.keys()).index(name)] for name in class_names]

        # Streamlit App - Plotting the Bar Chart
        st.title('GES Change Classification')

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(class_names, counts, color=colors)

        # Customize the chart
        ax.set_title('GES Change Classification')
        ax.set_xlabel('Classification')
        ax.set_ylabel('Pixel Count')
        ax.grid(axis='y')  # Add horizontal grid lines

        # Display the chart in Streamlit
        st.pyplot(fig)

    except MemoryError as e:
        st.error(f"Error: {str(e)}")
    except TimeoutError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")


def download_gee_image(image: ee.Image, region: ee.Geometry, filename: str = 'gee_image.tif', scale: int = 1000):
    """
    Exports an Earth Engine image to a GeoTIFF and provides a Streamlit download button.

    Parameters:
    - image: ee.Image object to export.
    - region: ee.Geometry defining the export region.
    - filename: Name of the output file (TIF format).
    - scale: Resolution in meters per pixel.
    """
    try:
        # Export the image to local GeoTIFF file
        geemap.ee_export_image(
            image,
            filename=filename,
            scale=scale,
            region=region,
        )
        st.success("Image exported successfully!")

        # Offer the file for download
        with open(filename, "rb") as f:
            st.download_button(
                label="Download GeoTIFF",
                data=f,
                file_name=filename,
                mime="image/tiff"
            )
    except Exception as e:
        st.error(f"Image export failed: {e}")


        

# --- Streamlit UI --- #
st.title("🌍 Good Environmental Status (GES) Mapping Tool")

# Main content
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
    try:
        st.info("Processing... Please wait a few moments.")
        intersection, region, filtered = return_intersect(country, buffer_km)
        GES_first = get_ges(intersection, start_year)
        GES_last = get_ges(intersection, end_year)
        GES_diff = GES_last.subtract(GES_first)
        
        # Display the map
        m = geemap.Map()
        m.centerObject(region, 6)
        m.addLayer(GES_first, ges_params1, "GES Start Year", shown=False)
        m.addLayer(GES_last, ges_params1, "GES End Year", shown=False)
        m.addLayer(GES_diff, ges_params1, "GES Change")
        m.addLayer(filtered.style(**{"color": "black", "fillColor": "#00000000", "width": 2}), {}, "Border")
        m.add_legend(title="GES Classification", legend_dict=dict(zip(ges_params1['labels'], ges_params1['palette'])))
        m.to_streamlit(height=600)
                    
        process_and_display(GES_diff)

        download_gee_image(GES_diff, intersection, filename="ges-change.tif", scale=1000)
        download_gee_image(GES_first, intersection, filename="ges-first.tif", scale=1000)
        download_gee_image(GES_last, intersection, filename="ges-last.tif", scale=1000)


    except MemoryError as e:
        st.error(f"Memory Error: {str(e)}")
    except TimeoutError as e:
        st.error(f"Timeout Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

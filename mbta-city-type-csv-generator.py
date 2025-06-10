import geopandas as gpd
import pandas as pd

# Load shapefile
gdf = gpd.read_file("../CENSUS2020TOWNS_SHP/CENSUS2020TOWNS_POLY.shp")
# Compute density
gdf['density'] = gdf['POP2020'] / gdf['SQ_MILES']

# Classify
def classify_area_type(density):
    if density >= 5000:
        return 'urban'
    elif density >= 1500:
        return 'suburban_commercial'
    elif density >= 250:
        return 'suburban_residential'
    else:
        return 'rural'

gdf['area_type'] = gdf['density'].apply(classify_area_type)

# Save to CSV
gdf[['TOWN20', 'POP2020', 'SQ_MILES', 'density', 'area_type']].to_csv("auto_area_type_classification.csv", index=False)
print("Saved auto_area_type_classification.csv")


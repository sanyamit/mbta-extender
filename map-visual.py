import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Load shapefile again (adjust path as needed)
gdf = gpd.read_file("../CENSUS2020TOWNS_SHP/CENSUS2020TOWNS_POLY.shp")

# Load your classification file
area_df = pd.read_csv("auto_area_type_classification.csv")

# Merge on town name
merged = gdf.merge(area_df, on="TOWN20")

# Define color mapping
color_dict = {
    "urban": "red",
    "suburban_commercial": "orange",
    "suburban_residential": "yellow",
    "rural": "green"
}

# Plot
fig, ax = plt.subplots(figsize=(12, 12))
merged.plot(
    column='area_type',
    color=merged['area_type'].map(color_dict),
    edgecolor='black',
    linewidth=0.2,
    ax=ax,
    legend=True
)
plt.title("Massachusetts Area Types (Auto-Classified)")
plt.axis('off')
plt.show()
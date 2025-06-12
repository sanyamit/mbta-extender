import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, LineString
import glob
import os

# Load shapefile again (adjust path as needed)
gdf = gpd.read_file('/Users/dylankao/Desktop/AI/final proj/CENSUS2020TOWNS_SHP/CENSUS2020TOWNS_POLY.shp')

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


station_lines = {}
all_station_points = []

csv_files = glob.glob("Lines/*.csv")
for path in csv_files:
    line_name = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)
    df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    station_lines[line_name] = df
    all_station_points.extend(df["geometry"].tolist())

# --- Plot stations ---
station_gdf = gpd.GeoDataFrame(geometry=all_station_points, crs="EPSG:4326")
station_gdf = station_gdf.to_crs(merged.crs)
station_gdf.plot(ax=ax, marker='*', color='blue', markersize=50, label='Stations')

for i, pt in enumerate(station_gdf.geometry):
    ax.text(pt.x, pt.y, f"{i+1}", fontsize=6, ha='center', va='center', color='white')


# --- Plot rail lines between stations ---
for line_df in station_lines.values():
    line_df = gpd.GeoDataFrame(line_df, geometry=line_df["geometry"], crs="EPSG:4326").to_crs(merged.crs)
    line_geom = LineString(line_df["geometry"].tolist())
    gpd.GeoSeries([line_geom]).plot(ax=ax, linestyle='--', color='blue', linewidth=1, alpha=0.4)



plt.title("Massachusetts Area Types (Auto-Classified)")
plt.axis('off')
plt.show()
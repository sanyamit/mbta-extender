import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
import numpy as np
import random
import math

gdf = gpd.read_file("CENSUS2020TOWNS_SHP/CENSUS2020TOWNS_POLY.shp").to_crs("EPSG:4326")
area_df = pd.read_csv("auto_area_type_classification.csv")
merged = gdf.merge(area_df, on="TOWN20")

if "POP2020" not in merged.columns:
    merged["POP2020"] = merged["area_type"].map({
        "urban": 50000,
        "suburban_commercial": 20000,
        "suburban_residential": 15000,
        "rural": 5000
    }) + (1000 * np.random.rand(len(merged)))

station_df = pd.read_csv("stations.csv")
station_df = station_df.dropna(subset=["lat", "lon"])
station_df["geometry"] = station_df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
station_gdf = gpd.GeoDataFrame(station_df, geometry="geometry", crs="EPSG:4326")

station_points = list(station_gdf.geometry)

def nearest_station(town_pt):
    return min(station_points, key=lambda s: geodesic((town_pt.y, town_pt.x), (s.y, s.x)).km)

def simulated_annealing_towns(df, num_towns=5, max_iter=300):
    current = df.sample(num_towns)
    best = current.copy()

    def score(towns):
        total = 0
        for _, row in towns.iterrows():
            pt = row.geometry.centroid
            station = nearest_station(pt)
            dist = geodesic((pt.y, pt.x), (station.y, station.x)).km
            total += row["POP2020"] / (dist + 1)
        return total

    best_score = score(best)
    T = 10.0

    for _ in range(max_iter):
        candidate = df.sample(num_towns)
        delta = score(candidate) - score(current)
        if delta > 0 or random.random() < math.exp(delta / T):
            current = candidate
            if score(candidate) > best_score:
                best = candidate
                best_score = score(candidate)
        T *= 0.995

    return best

selected_towns = simulated_annealing_towns(merged, num_towns=5)

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 12))
merged.plot(column="area_type",
            cmap="Set1",
            edgecolor="black",
            linewidth=0.3,
            ax=ax)

station_gdf.plot(ax=ax, marker='*', color='blue', markersize=80)

for _, row in selected_towns.iterrows():
    town_pt = row.geometry.centroid
    nearest = nearest_station(town_pt)
    line = LineString([town_pt, nearest])
    gpd.GeoSeries([line], crs="EPSG:4326").plot(ax=ax, color='black', linewidth=2)

selected_towns.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
plt.title("Simple Simulated Annealing + A* Paths")
plt.axis('off')
plt.show()

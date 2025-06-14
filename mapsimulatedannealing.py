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

terrain_stats = {
    "urban": {"cost": 8, "color": "red", "population": (70000, 100000)},
    "suburban_commercial": {"cost": 5, "color": "orange", "population": (20000, 40000)},
    "suburban_residential": {"cost": 3, "color": "yellow", "population": (10000, 30000)},
    "rural": {"cost": 2, "color": "green", "population": (500, 1000)}
}

merged["POP2020"] = merged["area_type"].map(lambda t: random.randint(*terrain_stats[t]["population"]))

stations = pd.read_csv("stations.csv").dropna(subset=["lat", "lon"])
stations["geometry"] = stations.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
station_gdf = gpd.GeoDataFrame(stations, geometry="geometry", crs="EPSG:4326")
station_points = list(station_gdf.geometry)

def nearest_station(pt):
    return min(station_points, key=lambda s: geodesic((pt.y, pt.x), (s.y, s.x)).km)

def simulated_annealing_towns(df, num_towns=7, max_iter=150):
    current = df.sample(num_towns)
    best = current.copy()

    def score(towns):
        total = 0
        for _, row in towns.iterrows():
            pt = row.geometry.centroid
            station = nearest_station(pt)
            dist = geodesic((pt.y, pt.x), (station.y, station.x)).km
            cost = terrain_stats[row["area_type"]]["cost"]
            pop = row["POP2020"]
            total += pop - (dist * cost * 1.5)
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
        T *= 0.99

    return best

selected_towns = simulated_annealing_towns(merged, num_towns=7)

fig, ax = plt.subplots(figsize=(14, 14))
merged.plot(
    column="area_type",
    color=merged["area_type"].map({k: v["color"] for k, v in terrain_stats.items()}),
    edgecolor="black",
    ax=ax
)

station_gdf.plot(ax=ax, marker='*', color='blue', markersize=80)

for _, row in selected_towns.iterrows():
    town_pt = row.geometry.centroid
    nearest = nearest_station(town_pt)
    line = LineString([town_pt, nearest])
    gpd.GeoSeries([line], crs="EPSG:4326").plot(ax=ax, color='black', linewidth=2)

selected_towns.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
plt.title("Simulated Annealing: Towns Selected by Population, Terrain Type, Distance")
plt.axis('off')
plt.show()

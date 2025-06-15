import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
import heapq
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
merged["cost"] = merged["area_type"].map(lambda t: terrain_stats[t]["cost"])

stations = pd.read_csv("stations.csv").dropna(subset=["lat", "lon"])
stations["geometry"] = stations.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
station_gdf = gpd.GeoDataFrame(stations, geometry="geometry", crs="EPSG:4326")
station_points = list(station_gdf.geometry)

# a star
def build_graph(df):
    graph = {}
    centroids = df.geometry.centroid
    for i, a in centroids.items():
        graph[i] = []
        for j, b in centroids.items():
            if i != j:
                dist = geodesic((a.y, a.x), (b.y, b.x)).km
                if dist < 20:  # limit to nearby towns
                    cost = dist * df.iloc[j]["cost"]
                    graph[i].append((j, cost))
    return graph

def astar(graph, df, start_idx, end_idx):
    frontier = [(0, start_idx)]
    came_from = {start_idx: None}
    cost_so_far = {start_idx: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == end_idx:
            break
        for neighbor, cost in graph.get(current, []):
            new_cost = cost_so_far[current] + cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                end_pt = df.geometry[end_idx].centroid
                heuristic = geodesic((df.geometry[neighbor].centroid.y, df.geometry[neighbor].centroid.x), (end_pt.y, end_pt.x)).km
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current

    path = []
    cur = end_idx
    while cur is not None:
        path.append(df.geometry[cur].centroid)
        cur = came_from.get(cur)
    return path[::-1]

def nearest_station(pt):
    return min(station_points, key=lambda s: geodesic((pt.y, pt.x), (s.y, s.x)).km)

# sa
def simulated_annealing_towns(df, num_towns=7, max_iter=150):
    current = df.sample(num_towns)
    best = current.copy()

    def score(towns):
        total = 0
        for _, row in towns.iterrows():
            pt = row.geometry.centroid
            station = nearest_station(pt)
            dist = geodesic((pt.y, pt.x), (station.y, station.x)).km
            cost = row["cost"]
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
        T *= 0.98

    return best

selected_towns = simulated_annealing_towns(merged, num_towns=7)
graph = build_graph(merged)

fig, ax = plt.subplots(figsize=(14, 14))
merged.plot(column="area_type",
            color=merged["area_type"].map({k: v["color"] for k, v in terrain_stats.items()}),
            edgecolor="black", ax=ax)

station_gdf.plot(ax=ax, marker='*', color='blue', markersize=80)

for _, town in selected_towns.iterrows():
    town_pt = town.geometry.centroid
    station = nearest_station(town_pt)
    start_idx = merged.geometry.centroid.distance(town_pt).idxmin()
    end_idx = merged.geometry.centroid.distance(station).idxmin()
    path = astar(graph, merged, start_idx, end_idx)
    if len(path) >= 2:
        gpd.GeoSeries([LineString(path)], crs="EPSG:4326").plot(ax=ax, color="black", linewidth=2)

selected_towns.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
plt.title("Pathfinding on MA Towns")
plt.axis("off")
plt.show()

Rail Network Optimization with Local Search and A*

This project demonstrates a system for optimizing the placement of new commuter rail stations and the routing of new rail lines. The system uses a combination of A pathfinding* and local search algorithms (hill climbing and simulated annealing) to optimize rail network expansion, balancing between cost and population served.

Running the Grid Environment Simulation using A* and Hill Climbing

To run: Open the localsearch_Astar.py script, and run it. No additional packages are necessary.

The localsearch_Astar.py script runs a simulation of the grid-based rail network optimization process. It initializes a grid, applies local search and A* to optimize station placement, and visualizes the results. 
The simulation will generate three distinct images during its run:
- Initial Setup: The grid with terrain types and existing station locations.
- After Station Placement Optimization: The grid with newly optimized station placements.
- Final Network with Connections: The grid showing optimized paths between new stations and existing stations, including total population served and construction cost metrics.



For Simulated Annealing and A* algorithm implemented on map: 

Map Setup:

The Massachusetts town shapefile (CENSUS2020TOWNS_POLY.shp) is loaded using GeoPandas.
Towns are merged with area classification data (auto_area_type_classification.csv) containing population, area type (urban, rural, etc.), and density.
Simulated Annealing (SA):
Starts with a random sample of towns (num_towns, e.g. 15).
Each town is scored using: score = population − ( distance_to_station × terrain_cost × penalty )
Over 150 iterations, the algorithm accepts better town sets and occasionally accepts worse ones (to escape local optima), gradually settling on the best combination.
A* Pathfinding:
Each town is treated as a node, with connections to nearby towns within 30 km.
Edge cost = distance × terrain cost (cheaper to build in rural areas).
Heuristic = geodesic (straight-line) distance to the nearest existing station.
A* finds the most cost-effective path from each new town to its nearest station.
Visualization:
Towns are color-coded by area type.
Existing stations are shown as blue stars (from stations.csv).
Selected towns are highlighted with boundary outlines.
Black lines show A*-generated connection paths.
Required Files
These need to be in the directory:
main.py – Python script with the full SA + A* implementation
auto_area_type_classification.csv – Includes town names, area types, and populations
stations.csv – Contains existing stations with name, lat, lon
CENSUS2020TOWNS_SHP/ – Folder with MA shapefile data:
Includes .shp, .dbf, .shx, .prj
How to Run
Install dependencies:
pip install geopandas shapely matplotlib pandas geopy
Run the simulation:
python main.py
The script will generate a map showing:
Selected towns for expansion
Optimal A* rail paths
Existing stations and town classification

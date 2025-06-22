Rail Network Optimization with Local Search and A*
This project demonstrates a system for optimizing the placement of new commuter rail stations and the routing of new rail lines. The system uses a combination of A pathfinding* and local search algorithms (hill climbing and simulated annealing) to optimize rail network expansion, balancing between cost and population served.

Running the Grid Environment Simulation using A* and Hill Climbing
The localsearch_Astar.py script runs a simulation of the grid-based rail network optimization process. It initializes a grid, applies local search and A* to optimize station placement, and visualizes the results. 
The simulation will generate three distinct images during its run:
Initial Setup: The grid with terrain types and existing station locations.
After Station Placement Optimization: The grid with newly optimized station placements.
Final Network with Connections: The grid showing optimized paths between new stations and existing stations, including total population served and construction cost metrics.


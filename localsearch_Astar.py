import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from collections import defaultdict

class RailEnv:  
    def __init__(self, grid_size=20, num_stations=5, num_new_stations=20):   #This sets the rules for the initial grid env
        self.grid_size = grid_size
        self.num_stations = num_stations
        self.num_new_stations = num_new_stations

        self.terrain_types = {
            "urban": {"cost": 100, "color": "red", "population": (70000, 200000)},
            "suburban_commercial": {"cost": 30, "color": "orange", "population": (40000, 60000)},
            "suburban_residential": {"cost": 10, "color": "yellow", "population": (10000, 30000)},
            "rural": {"cost": 3, "color": "green", "population": (500, 9000)}
        }
        self.terrain_list = list(self.terrain_types.keys())
        self.terrain_probs = [0.07, 0.15, 0.48, 0.3]

        self.local_search_iterations = 100
        self.branch_penalty = 900

        self.reset()

    def reset(self): # This sets the grid env to its initial state, and classifies which cells are which terrain and where the existing stations are
        self.grid = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                terrain = random.choices(self.terrain_list, weights=self.terrain_probs, k=1)[0]
                pop_range = self.terrain_types[terrain]["population"]
                self.grid[(x, y)] = {
                    "terrain": terrain,
                    "cost": self.terrain_types[terrain]["cost"],
                    "population": random.randint(*pop_range),
                    "station": False,
                    "new_town": False,
                    "connected": False
                }

        self.stations = random.sample(list(self.grid.keys()), self.num_stations)
        for coord in self.stations:
            self.grid[coord]["station"] = True
            self.grid[coord]["connected"] = True

        self.new_towns = []
        self.connected_paths = []
        self.path_node_usage = defaultdict(int)
        self.total_population_served = 0
        self.total_construction_cost = 0

    def render(self, title="Rail Network Planning"):  #This renders the grid environment, giving the viewer a visual
        fig, ax = plt.subplots(figsize=(12, 12))
        for (x, y), info in self.grid.items():
            color = self.terrain_types[info["terrain"]]["color"]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='gray', alpha=0.7))
            if info["population"] > 5000:
                ax.text(x + 0.5, y + 0.5, f"{info['population']//1000}k", ha='center', va='center', fontsize=8)

        for (x, y), info in self.grid.items():
            if info["station"]:
                ax.plot(x + 0.5, y + 0.5, '*', color='blue', markersize=15)
            elif info["new_town"]:
                ax.plot(x + 0.5, y + 0.5, 'o', color='black', markersize=10)

        for path in self.connected_paths:
            xs = [x + 0.5 for x, y in path]
            ys = [y + 0.5 for x, y in path]
            linewidth = 1 + sum(self.path_node_usage.get((x, y), 0) for (x, y) in path) / len(path)
            ax.plot(xs, ys, color='black', linewidth=linewidth)

        ax.set_title(f"{title}\nPopulation Served: {self.total_population_served:,}\nConstruction Cost: {self.total_construction_cost:,}")
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.show()

    def _get_neighbors(self, x, y): #gets neighboring cells in each of the 4 directions
        return [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] if 0 <= x+dx < self.grid_size and 0 <= y+dy < self.grid_size]

    def _manhattan(self, a, b): #obtains manhattan distance for heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar(self, start, goals):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        goal_set = set(goals)

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in goal_set:
                break
            for neighbor in self._get_neighbors(*current):
                new_cost = cost_so_far[current] + self.grid[neighbor]["cost"]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + min(self._manhattan(neighbor, g) for g in goals)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        reachable = [g for g in goals if g in came_from]
        if not reachable:
            return [], float('inf')
        best_goal = min(reachable, key=lambda g: cost_so_far[g])
        path = []
        cur = best_goal
        while cur:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path, cost_so_far[best_goal]

    def _calculate_3x3_population(self, coord):
        total = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x, y = coord[0]+dx, coord[1]+dy
                if (x, y) in self.grid:
                    total += self.grid[(x, y)]["population"]
        return total

    def _is_valid_location(self, coord, existing_locations):
        return all(self._manhattan(coord, other) > 2 for other in existing_locations)

    def optimize_station_placement(self):
        candidates = sorted([c for c in self.grid.keys() if not self.grid[c]["station"]],
                            key=lambda c: -self._calculate_3x3_population(c))
        current_solution = []
        for coord in candidates:
            if self._is_valid_location(coord, current_solution + self.stations):
                current_solution.append(coord)
                if len(current_solution) >= self.num_new_stations:
                    break

        best_solution = current_solution.copy()
        best_score = self._evaluate_solution(best_solution)

        for _ in range(self.local_search_iterations):
            neighbor = current_solution.copy()
            idx = random.randint(0, len(neighbor)-1)
            alternatives = [c for c in self.grid.keys() if not self.grid[c]["station"] and
                            self._is_valid_location(c, [x for i,x in enumerate(neighbor) if i != idx] + self.stations)]
            if not alternatives:
                continue
            neighbor[idx] = random.choice(alternatives)
            neighbor_score = self._evaluate_solution(neighbor)
            if neighbor_score > best_score:
                best_solution = neighbor.copy()
                best_score = neighbor_score
                current_solution = neighbor.copy()

        self.new_towns = best_solution
        for coord in self.new_towns:
            self.grid[coord]["new_town"] = True

    def _evaluate_solution(self, solution):
        score = 0
        for town in solution:
            path, cost = self._astar(town, self.stations)
            if path:
                score += self._calculate_3x3_population(town) - cost*100
        return score

    def connect_towns(self):
        self.total_population_served = 0
        self.total_construction_cost = 0

        for town in self.new_towns:
            path, cost = self._astar(town, self.stations)
            if not path:
                continue
            path_cost = sum(self.grid[p]["cost"] for p in path[:-1])
            overlap_penalty = sum(self.path_node_usage[p] for p in path) * self.branch_penalty
            total_cost = path_cost + overlap_penalty
            pop_served = self._calculate_3x3_population(town)
            if pop_served - total_cost*0.1 > 0:
                for cell in path:
                    self.grid[cell]["connected"] = True
                    self.path_node_usage[cell] += 1
                self.connected_paths.append(path)
                self.total_population_served += pop_served
                self.total_construction_cost += total_cost

def show_legend(pop_served, cost, cost_per):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    legend_text = (
        "Legend:\n"
        "- Red: Urban\n"
        "- Orange: Suburban Commercial\n"
        "- Yellow: Suburban Residential\n"
        "- Green: Rural\n"
        "- *: Existing Station\n"
        "- o: New Station\n"
        "\n"
        f"Total Population Served: {pop_served:,}\n"
        f"Total Construction Cost (milions): {cost:,}\n"
        f"Cost per Person: {cost_per:.2f}"
    )
    ax.text(0, 1, legend_text, fontsize=12, va='top')
    plt.show()

def run_simulation():
    print("Rail Network Optimization with Local Search")
    env = RailEnv(grid_size=15, num_stations=6, num_new_stations=10)
    env.reset()
    env.render("Initial Setup with Existing Stations")
    print("Optimizing new station locations...")
    env.optimize_station_placement()
    env.render("After Station Placement Optimization")
    print("Connecting towns to rail network...")
    env.connect_towns()
    env.render("Final Network with Connections")
    print("\nFinal Statistics:")
    print(f"Total Population Served: {env.total_population_served:,}")
    print(f"Total Construction Cost (in millions): {env.total_construction_cost:,}")

run_simulation()


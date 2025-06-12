import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

class RailEnv:
    def __init__(self, grid_size=15, num_towns=9, num_stations=5):
        self.grid_size = grid_size
        self.num_towns = num_towns
        self.num_stations = num_stations

        self.terrain_types = {
            "urban": 8,
            "suburban_commercial": 5,
            "suburban_residential": 3,
            "rural": 2
        }
        self.terrain_list = list(self.terrain_types.keys())
        self.terrain_probs = [0.1, 0.35, 0.2, 0.35]

        self.reset()

    def _generate_grid(self):
        grid = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                terrain = random.choices(self.terrain_list, weights=self.terrain_probs, k=1)[0]
                grid[(x, y)] = {
                    "terrain": terrain,
                    "cost": self.terrain_types[terrain],
                    "station": False,
                    "new_town": False,
                    "connected": False
                }
        return grid

    def _get_neighbors(self, x, y):
        return [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= x+dx < self.grid_size and 0 <= y+dy < self.grid_size]

    def _astar(self, start, goals):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in goals:
                break
            for neighbor in self._get_neighbors(*current):
                if neighbor not in self.grid:
                    continue
                new_cost = cost_so_far[current] + self.grid[neighbor]["cost"]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))
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

    def reset(self):
        self.grid = self._generate_grid()
        all_coords = list(self.grid.keys())

        self.stations = random.sample(all_coords, self.num_stations)
        for coord in self.stations:
            self.grid[coord]["station"] = True
            self.grid[coord]["connected"] = True

        remaining = [c for c in all_coords if c not in self.stations]
        self.new_towns = random.sample(remaining, self.num_towns)
        for coord in self.new_towns:
            self.grid[coord]["new_town"] = True

        self.connected_paths = []
        self.unconnected_towns = set(self.new_towns)
        return self.get_observation()

    def get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for (x, y), info in self.grid.items():
            if info["station"]:
                obs[x, y] = 3
            elif info["new_town"]:
                obs[x, y] = 2
            else:
                obs[x, y] = 1
        return obs

    def step(self, town_coord):
        if town_coord not in self.unconnected_towns:
            return self.get_observation(), -10, False, {"error": "Invalid action"}

        path, cost = self._astar(town_coord, self.stations)
        reward = 100 - cost
        for cell in path:
            self.grid[cell]["connected"] = True
        self.connected_paths.append(path)
        self.unconnected_towns.remove(town_coord)
        done = len(self.unconnected_towns) == 0
        return self.get_observation(), reward, done, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        terrain_colors = {
            "urban": "red",
            "suburban_commercial": "orange",
            "suburban_residential": "yellow",
            "rural": "green"
        }

        for (x, y), info in self.grid.items():
            color = terrain_colors[info["terrain"]]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black'))
            if info["station"]:
                ax.plot(x + 0.5, y + 0.5, '*', color='blue', markersize=12)
            elif info["new_town"]:
                ax.plot(x + 0.5, y + 0.5, 'o', color='black', markersize=8)

        for path in self.connected_paths:
            xs = [x + 0.5 for x, y in path]
            ys = [y + 0.5 for x, y in path]
            ax.plot(xs, ys, color='black', linewidth=2)

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.set_title("A* connected Rail Lines")
        plt.gca().invert_yaxis()
        plt.show()

# --- Example run ---

env = RailEnv(grid_size=15, num_towns=10, num_stations=7)
obs = env.reset()
env.render()

# Simulate manually connecting towns
for town in list(env.unconnected_towns):
    obs, reward, done, info = env.step(town)
    print(f"Connected {town} with reward {reward}")
    if done:
        break

env.render()

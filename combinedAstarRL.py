import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq

# --- RailEnv with A* and External Town Control ---
class RailEnv:
    def __init__(self, grid_size=10, num_stations=5):
        self.grid_size = grid_size
        self.num_stations = num_stations
        self.terrain_types = {
            "urban": 10,
            "suburban_commercial": 7,
            "suburban_residential": 5,
            "rural": 2
        }
        self.terrain_list = list(self.terrain_types.keys())
        self.terrain_probs = [0.2, 0.25, 0.35, 0.2]
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
        self.connected_paths = []
        return self

    def add_new_town_and_build(self, town_coord):
        self.grid[town_coord]["new_town"] = True
        path, cost = self._astar(town_coord, self.stations)
        for cell in path:
            self.grid[cell]["connected"] = True
        self.connected_paths.append(path)

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
        ax.set_title("Q-learning-Informed Rail Network")
        plt.gca().invert_yaxis()
        plt.show()

# --- Run Q-learning plan and apply it to RailEnv ---

# Towns and positions
towns = {
    'Lexington': {'population': 33000, 'cost': 40000000, 'type': 'suburban_residential', 'pos': (2, 2)},
    'Burlington': {'population': 25000, 'cost': 40000000, 'type': 'suburban_commercial', 'pos': (3, 2)},
    'Waltham': {'population': 60000, 'cost': 50000000, 'type': 'suburban_commercial', 'pos': (4, 2)},
    'Concord': {'population': 18000, 'cost': 40000000, 'type': 'suburban_residential', 'pos': (2, 3)},
    'Lincoln': {'population': 7000, 'cost': 30000000, 'type': 'rural', 'pos': (1, 4)}
}
actions = list(towns.keys())
Q = {}

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500
budget = 10_000_000_000

def get_state(current_budget, built_towns):
    return (round(current_budget, -7), frozenset(built_towns))

def choose_action(state, available_actions):
    if random.random() < epsilon:
        return random.choice(available_actions)
    else:
        q_vals = [Q.get((state, a), 0) for a in available_actions]
        return available_actions[np.argmax(q_vals)]

def get_reward(town):
    pop_reward = towns[town]['population'] * 10
    cost_penalty = towns[town]['cost'] * 0.001
    return pop_reward - cost_penalty

# Q-learning training
for episode in range(episodes):
    current_budget = budget
    built = set()
    state = get_state(current_budget, built)
    for step in range(10):
        available = [t for t in actions if t not in built and towns[t]['cost'] <= current_budget]
        if not available:
            break
        action = choose_action(state, available)
        reward = get_reward(action)
        cost = towns[action]['cost']
        new_budget = current_budget - cost
        new_built = built | {action}
        new_state = get_state(new_budget, new_built)
        old_q = Q.get((state, action), 0)
        future_q = max([Q.get((new_state, a), 0) for a in actions if a not in new_built], default=0)
        Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)
        state = new_state
        current_budget = new_budget
        built = new_built

# Final greedy plan
final_budget = budget
built_final = set()
state = get_state(final_budget, built_final)
final_plan = []

while True:
    available = [t for t in actions if t not in built_final and towns[t]['cost'] <= final_budget]
    if not available:
        break
    q_vals = [(t, Q.get((state, t), 0)) for t in available]
    if not q_vals:
        break
    action, _ = max(q_vals, key=lambda x: x[1])
    final_plan.append(action)
    built_final.add(action)
    final_budget -= towns[action]['cost']
    state = get_state(final_budget, built_final)

# Apply final plan to the grid and visualize
env = RailEnv(grid_size=10, num_stations=5)
env.reset()

for town_name in final_plan:
    coord = towns[town_name]["pos"]
    env.add_new_town_and_build(coord)

env.render()

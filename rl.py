import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq

'''
Build efficient train routes that:
Maximize population served
Stay within budget
Minimize track cost and urban congestion
Visualize the final rail network on a color-coded grid
'''

#total budget 
GRID_SIZE = 15
BUDGET = 1_100_000_000
# Add Stations manually
# Randomly assign positions to other towns from the data 
STATIONS = [("North Station", (7, 7)), ("South Station", (7, 8))]


'''Each area type is assigned:
A station cost (e.g., $50M for urban)
A reward multiplier (serving more suburban commuters = better)
A track cost multiplier (urban/rural harder to build)'''

terrain_types = {
    'urban':    (50_000_000, 1.0, 1.5),  # cost, reward, track multiplier
    'suburban_residential': (40_000_000, 2.0, 0.8),
    'suburban_commercial':  (45_000_000, 1.5, 1.0),
    'rural':     (35_000_000, 0.5, 2.0)
}

# ----------------- LOAD DATA -----------------
df = pd.read_csv("auto_area_type_classification.csv")
df = df.head(20)  # optional: limit for speed

# Add fixed stations
for name, _ in STATIONS:
    df.loc[len(df.index)] = [name, 0, 0, 0, 'urban']

# Assign positions
positions = [pos for _, pos in STATIONS]
used = set(positions)
while len(positions) < len(df):
    p = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    if p not in used:
        positions.append(p)
        used.add(p)

towns = {}
#each town has all of this data
for (i, row), pos in zip(df.iterrows(), positions):
    cost, reward_mult, track_mult = terrain_types.get(row['area_type'], (30_000_000, 1.0, 1.0))
    towns[row['TOWN20']] = {
        'pos': pos,
        'pop': row['POP2020'],
        'cost': cost,
        'reward_mult': reward_mult,
        'track_mult': track_mult,
        'type': row['area_type']
    }

# ----------------- Q-LEARNING -----------------
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000
all_towns = list(towns.keys())

def get_state(budget_left, built):
    return (budget_left // 10_000_000, frozenset(built))

def get_reward(town):
    t = towns[town]
    return t['pop'] * t['reward_mult'] - t['cost']/1e6 - t['track_mult']*100

for _ in range(episodes):
    budget = BUDGET
    built = set(t[0] for t in STATIONS)
    state = get_state(budget, built)
    for _ in range(15):
        available = [t for t in all_towns if t not in built and towns[t]['cost'] <= budget]
        if not available: break
        action = random.choice(available) if random.random() < epsilon else max(available, key=lambda a: Q.get((state,a),0))
        reward = get_reward(action)
        new_budget = budget - towns[action]['cost']
        new_built = built | {action}
        new_state = get_state(new_budget, new_built)
        future_q = max([Q.get((new_state, a), 0) for a in all_towns if a not in new_built], default=0)
        Q[(state, action)] = Q.get((state, action), 0) + alpha * (reward + gamma * future_q - Q.get((state, action), 0))
        state, budget, built = new_state, new_budget, new_built

# ----------------- FINAL PLAN -----------------
budget = BUDGET
built = set(t[0] for t in STATIONS)
state = get_state(budget, built)
plan = []

while True:
    available = [t for t in all_towns if t not in built and towns[t]['cost'] <= budget]
    if not available: break
    action = max(available, key=lambda a: Q.get((state, a), 0))
    plan.append(action)
    built.add(action)
    budget -= towns[action]['cost']
    state = get_state(budget, built)

# ----------------- A* FUNCTION -----------------
terrain_grid = {(x, y): random.choice([2, 5, 7, 10]) for x in range(GRID_SIZE) for y in range(GRID_SIZE)}
area_color_map = {
    'rural': 2, #green
    'suburban_residential': 5, #yellow
    'suburban_commercial': 7, #orange
    'urban': 10 #red
}

for t in towns.values():
    terrain_grid[t['pos']] = area_color_map.get(t['type'], 5)

def astar(start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal: break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                new_cost = cost_so_far[current] + terrain_grid[(nx, ny)]
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    came_from[(nx, ny)] = current
                    heapq.heappush(frontier, (new_cost, (nx, ny)))
    path = []
    cur = goal
    while cur and cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    return path[::-1]

# ----------------- PLOT -----------------
terrain_colors = {2: 'green', 5: 'yellow', 7: 'orange', 10: 'red'}
fig, ax = plt.subplots(figsize=(8,8))
for (x,y), v in terrain_grid.items():
    c = terrain_colors[min(terrain_colors, key=lambda k: abs(k-v))]
    ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=c, edgecolor='gray'))

# Draw towns
for name in plan:
    x, y = towns[name]['pos']
    ax.plot(x+.5, y+.5, 'ko')
    ax.text(x+.5, y+.5, name, fontsize=6, ha='center', va='center', color='white')

# Draw stations
for name, pos in STATIONS:
    x, y = pos
    ax.plot(x+.5, y+.5, 'o', color='blue', markersize=10)
    ax.text(x+.5, y+.5, name, fontsize=7, ha='center', va='center', color='white')

# Draw paths
path_names = [s[0] for s in STATIONS] + plan
for i in range(len(path_names)-1):
    p1 = towns[path_names[i]]['pos']
    p2 = towns[path_names[i+1]]['pos']
    path = astar(p1, p2)
    xs = [x+.5 for x, y in path]
    ys = [y+.5 for x, y in path]
    ax.plot(xs, ys, 'black', linewidth=2)

ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect('equal')
plt.gca().invert_yaxis()
plt.title("MBTA Expansion Plan (Q-learning)")
plt.grid(True)
plt.show()

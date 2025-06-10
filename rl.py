import random
import numpy as np
import pandas as pd

# Q-learning parameters
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor
epsilon = 0.2     # exploration rate
episodes = 500
budget = 10_000_000_000  # $10 billion

# Sample town data
towns = {
    'Lexington': {'population': 33000, 'cost': 40000000, 'type': 'Suburban Residential'},
    'Burlington': {'population': 25000, 'cost': 40000000, 'type': 'Suburban Commercial'},
    'Waltham': {'population': 60000, 'cost': 50000000, 'type': 'Suburban Commercial'},
    'Concord': {'population': 18000, 'cost': 40000000, 'type': 'Suburban Residential'},
    'Lincoln': {'population': 7000, 'cost': 30000000, 'type': 'Rural'}
}

actions = list(towns.keys())
Q = {}

# Define the state as the budget (rounded) and the towns already selected
def get_state(current_budget, built_towns):
    return (round(current_budget, -7), frozenset(built_towns))

# Choose action using epsilon-greedy policy
def choose_action(state, available_actions):
    if random.random() < epsilon:
        return random.choice(available_actions)
    else:
        q_vals = [Q.get((state, a), 0) for a in available_actions]
        return available_actions[np.argmax(q_vals)]

# Reward is based on population benefit minus construction penalty
def get_reward(town):
    pop_reward = towns[town]['population'] * 10
    cost_penalty = towns[town]['cost'] * 0.001
    return pop_reward - cost_penalty

# Training loop
for episode in range(episodes):
    current_budget = budget
    built = set()
    state = get_state(current_budget, built)

    for step in range(10):  # limit steps per episode
        available = [t for t in actions if t not in built and towns[t]['cost'] <= current_budget]
        if not available:
            break

        action = choose_action(state, available)
        cost = towns[action]['cost']
        reward = get_reward(action)

        new_budget = current_budget - cost
        new_built = built | {action}
        new_state = get_state(new_budget, new_built)

        old_q = Q.get((state, action), 0)
        future_q = max([Q.get((new_state, a), 0) for a in actions if a not in new_built], default=0)

        Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

        state = new_state
        current_budget = new_budget
        built = new_built

# Greedy rollout to get final plan
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
    final_plan.append((action, towns[action]['population'], towns[action]['cost']))
    built_final.add(action)
    final_budget -= towns[action]['cost']
    state = get_state(final_budget, built_final)

# Output as DataFrame
df_plan = pd.DataFrame(final_plan, columns=["Town", "Population", "Cost"])
print(df_plan)

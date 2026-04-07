# Assignment 1: Q-Learning Implementation Report

This report documents the implementation of the Q-Learning algorithm on the `FrozenLake-v1` environment. The goal was to train an agent to navigate from a start tile to a goal tile while avoiding holes.

---

## T1. Implementation of Q-Learning
The implementation was carried out in the `01_LAB_A_Q_Learning.ipynb` notebook. The process involved defining the state-action space and initializing the knowledge base.

### Q-Table Shape and Initialization
* **Identification**: The code uses `env.observation_space.n` and `env.action_space.n` to determine dimensions. For `FrozenLake-v1`, this results in 16 states and 4 actions.
* **Implementation**: `Q_table = np.zeros((state_size, action_size))` creates a $16 \times 4$ matrix where every state-action pair is initially valued at zero, representing a lack of prior knowledge.

---

## T2. Detailed Explanation of the Core Code

### 1. Calculation of the TD Error
* **Code Connection**: 
    ```python
    td_target = reward + gamma * np.max(Q_table[new_state, :])
    td_error = td_target - Q_table[state, action]
    ```
* **Theory**: The **Temporal Difference (TD) Error** measures the discrepancy between the current Q-value estimate and the updated estimate provided by the reward and the discounted value of the next state.

### 2. The Q-Table Update Rule
* **Code Connection**: 
    ```python
    Q_table[state, action] = Q_table[state, action] + alpha * td_error
    ```
* **Theory**: This follows the Bellman equation to iteratively improve the agent's policy. By adding a fraction (`alpha`) of the TD error to the current value, the table gradually converges toward the optimal values.

### 3. The $\epsilon$-greedy Action Selection
* **Code Connection**:
    ```python
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(Q_table[state, :]) # Exploitation
    else:
        action = env.action_space.sample()     # Exploration
    ```
* **Theory**: This strategy balances **exploration** (trying new actions) and **exploitation** (using known best actions).

### 4. The $\epsilon$ Decay Schedule
* **Code Connection**: 
    ```python
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    ```
* **Theory**: As the agent learns, the need for exploration decreases. The exponential decay ensures the agent moves from random searching to informed decision-making over time.

---

## Hyperparameters Table
[cite: 19, 20]

| Hyperparameter | Value | Justification |
| :--- | :--- | :--- |
| $\alpha$ (Learning Rate) | 0.1 | Determines how much new information overrides old information. |
| $\gamma$ (Discount Factor) | 0.99 | High value to prioritize long-term rewards (reaching the goal). |
| Episodes | 1000 | Sufficient iterations for convergence in a deterministic $4 \times 4$ grid. |
| Max Steps | 100 | Limits the episode length to prevent infinite loops. |
| $\epsilon_{max}$ | 1.0 | Start with total exploration of the environment. |
| $\epsilon_{min}$ | 0.01 | Ensures the agent never completely stops exploring. |
| Decay Rate | 0.0005 | Controls the speed at which the agent shifts to exploitation. |

---

## T3. Learning Analysis and Visualization
The implementation includes a reward tracking system and a video rendering module.
* **Learning Curve**: Rewards were stored in the `rewards` list and plotted to show how the agent's success rate increases across 1000 episodes.
* **Maximum Q-Value Heatmap**: By calculating `np.max(Q_table, axis=1)`, we visualize the "safest" paths toward the goal.
* **Video Output**: Using `RecordVideo` and `show_video`, the final trained policy is visualized to demonstrate the agent navigating the ice successfully.

---

## T4. Environment Variation
The code was tested with `is_slippery=False`. 
* **Deterministic (`is_slippery=False`)**: The agent learns a direct path since actions result in the intended state transition.
* **Stochastic (`is_slippery=True`)**: If enabled, the agent would require more episodes and a more conservative policy because actions have only a probability of moving the agent in the desired direction.
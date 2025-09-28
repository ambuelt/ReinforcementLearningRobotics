import Graph.Graph as g
import OnlineLearning.QLearn_Online as ql

import numpy as np
import matplotlib.pyplot as plt

def OnlineLearning(lr: float, df: float, er: float, num_episodes: int) -> None:
    """
    Runs Q-learning online on the GridWorld environment and tracks metrics.
    """
    
    # create grid world environment using default penalty
    env = g.GridWorld()
    
    # create Q-learning online agent
    agent = ql.QLearningOnline(learning_rate=lr, discount_factor=df, exploration_rate=er, init="zeros")

    # metrics tracking
    q_changes = []       # average |delta Q| per episode
    policy_stable = []   # fraction of states with unchanged greedy action
    prev_policy = None
    returns = []         # total reward per episode
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        deltas = []  # track Q-value changes in this episode

        while not done:
            # i) choose action
            action = agent.choose_action(state)

            # ii) take action and observe next state and reward
            next_state, reward, done = env.step(action)

            # iii) update Q-value
            old_q = agent.q_table.copy()
            agent.update_q_value(state, action, reward, next_state, done)
            diff = np.abs(agent.q_table - old_q)
            deltas.append(np.mean(diff))

            # iv) transition to next state
            state = next_state
            total_reward += reward

        # metrics update
        q_changes.append(np.mean(deltas))
        greedy_policy = np.argmax(agent.q_table, axis=2)
        if prev_policy is None:
            stable_frac = 0.0
        else:
            stable_frac = np.mean(greedy_policy == prev_policy)
        policy_stable.append(stable_frac)
        prev_policy = greedy_policy.copy()
        returns.append(total_reward)

    # Final metrics output
    window = 10
    print("\n=== Metrics (last {} episodes) ===".format(window))
    print("Average return     :", [f"{v:.2f}" for v in returns[-window:]])
    print("Q-value changes    :", [f"{v:.6f}" for v in q_changes[-window:]])
    print("Policy stability   :", [f"{v:.2f}" for v in policy_stable[-window:]])
    
    # Plotting metrics
    episodes = range(1, num_episodes + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(episodes, returns, label="Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episodes, q_changes, label="$\delta$Q", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg |$\delta$Q|")
    plt.title("Q-value Changes")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(episodes, policy_stable, label="Stability", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Stable fraction")
    plt.title("Policy Stability")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    

def main():
    
    # Parameters
    lr = 0.1 # learning rate example
    df = 0.9 # discount factor example
    er = 0.2 # exploration rate example
    num_episodes = 500 # number of episodes for training
    
    # Run online learning
    OnlineLearning(lr, df, er, num_episodes)



if __name__ == "__main__":
    main()
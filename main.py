import Graph.Graph as g
import OnlineLearning.QLearn_Online as ql
import OfflineLearning.QLearn_Offline as ql_off
import time
import numpy as np
import matplotlib.pyplot as plt

BREAK_CON = 0.00008

def OnlineLearning(lr: float, df: float, er: float, num_episodes: int, penalty: float) -> None:
    """
    Runs Q-learning online on the GridWorld environment and tracks metrics.
    """
    
    # create grid world environment using default penalty
    env = g.GridWorld(penalty = penalty)
    
    # create Q-learning online agent
    agent = ql.QLearningOnline(learning_rate=lr, discount_factor=df, exploration_rate=er, init="random")

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
        q_val_delta = np.mean(deltas)
        q_changes.append(q_val_delta)
        greedy_policy = np.argmax(agent.q_table, axis=2)
        if prev_policy is None:
            stable_frac = 0.0
        else:
            stable_frac = np.mean(greedy_policy == prev_policy)
        policy_stable.append(stable_frac)
        prev_policy = greedy_policy.copy()
        returns.append(total_reward)
        if q_val_delta <= BREAK_CON:
            break
    
    # Show final Q-Action pair for each grid
    q_action_pair = np.zeros_like(env.grid,dtype='str')
    q_action_dict = {0:'N', 1:'E',2:'S',3:'W'}
    for i in range(0,env.grid.shape[0]):
        for j in range(0, env.grid.shape[1]):
            q_action_pair[i][j]=(q_action_dict[agent.choose_action((i,j))])
        
        

    # Final metrics output
    window = 10
    print("\n=== Metrics (last {} episodes) ===".format(window))
    print("Average return     :", [f"{v:.2f}" for v in returns[-window:]])
    print("Q-value changes    :", [f"{v:.6f}" for v in q_changes[-window:]])
    print("Policy stability   :", [f"{v:.2f}" for v in policy_stable[-window:]])
    print('State Action Grid:\n',q_action_pair)

    
    # Plotting metrics
    episodes = range(1, len(q_changes)+1)

    # 1 = plot the data, 0 = turn plot off
    if 1:
        plt.figure(figsize=(12, 4))
        plt.suptitle(f'ONLINE : LR {lr:.2f}, DF {df:.2f}, ER {er:.2f}, P {penalty:.2f}')
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

def OfflineLearning(episode: int, epoch: int, lr: float, df: float, penalty: float) -> None:
    """
    Runs offline Q-learning in the GridWorld environment and tracks metrics
    """
    
    # i) Create grid world environment
    grid = g.GridWorld(penalty = penalty)
    
    # ii) Create Q-learning offline agent
    agent = ql_off.QLearningOffline(grid, epoch=epoch, learning_rate=lr, discount_factor=df)

    # iii) Create dataset to train on made up of a list of tuples containing the current state, action, reward, and next state
    data_set = agent.generate_training_dataset(grid, episode)

    # iv) Update Offline Qlearning
    q_table = agent.offline_q_learning(grid, data_set)
    
      

    # Final metrics output
    window = 10
    print("\n=== Metrics (last {} episodes) ===".format(window))
    print("Average return     :", [f"{v:.2f}" for v in agent.returns[-window:]])
    print("Q-value changes    :", [f"{v:.6f}" for v in agent.q_changes[-window:]])
    print("Policy stability   :", [f"{v:.2f}" for v in agent.policy_stable[-window:]])

    # Show final Q-Action pair for each grid
    offline_best_output = agent.display_policy(grid, q_table)

    # Print by symbolic best policy by row
    for row in offline_best_output:
        print(row)

    
    # Plotting metrics
    episodes = range(1, len(agent.q_changes)+1)
    
    # 1 = plot the data, 0 = turn plot off
    if 0:
        plt.figure(figsize=(12, 4))
        plt.suptitle(f'OFFLINE : LR {lr:.2f}, DF {df:.2f}, P {penalty:.2f}')
        plt.subplot(1, 3, 1)
        plt.plot(episodes, agent.returns, label="Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode Returns")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(episodes, agent.q_changes, label="$\delta$Q", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Avg |$\delta$Q|")
        plt.title("Q-value Changes")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(episodes, agent.policy_stable, label="Stability", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Stable fraction")
        plt.title("Policy Stability")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    

def main():
    lr_ = [0.05, 0.1, 0.3]
    df_ = [0.8, 0.95,0.99]
    er_ = [0.05,0.2, 0.5]
    
    
    p = -1
    tik = time.time()
    num_episodes = 500 # number of episodes for training
    for i in lr_:
        for j in df_:
            for k in er_:
                # Run online learning
                OnlineLearning(i, j, k, num_episodes,penalty=p)
    tok = time.time()
    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')
    
   
    
    tik = time.time()
    # Parameters
    lr = 0.1 # learning rate example
    df = 0.9 # discount factor example
    er = 0.2 # exploration rate example
    p = -200
    
    # Run online learning
    OnlineLearning(lr, df, er, num_episodes,penalty=p)
    tok = time.time()
    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')



    ################################################################################
    # Start of Offline Q-learning
    ################################################################################

    # Parameters for Offline Learning
    lr_ = [0.05, 0.1, 0.3]
    df_ = [0.8, 0.95,0.99]
    penalty = -1            # Value in penalty state
    episodes = 100          # Number of episodes for dataset creation (used to train off of)
    epoch = 100             # Number of times to run Qlearning for offline

    # Time how long it takes to run offline learning
    tik = time.time()  # Start time

    for i in lr_:
        for j in df_:
            OfflineLearning(episodes, epoch, i, j, penalty=penalty)

    tok = time.time()  # End time

    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')

    ### Offline Learning Large Penalty #############################################
    
    # Parameters for Offline Learning
    lr = 0.1           # learning rate
    df = 0.9           # discount factor
    p = -200           # Value in penalty state
    episodes = 100     # Number of episodes for dataset creation (used to train off of)
    epoch = 100        # Number of times to run Qlearning for offline

    # Time how long it takes to run offline learning
    tik = time.time()  # Start time
    
    # Run online learning
    OfflineLearning(episodes, epoch, i, j, penalty=penalty)
    tok = time.time()  # End time

    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')



if __name__ == "__main__":
    main()

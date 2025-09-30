import Graph.Graph as g
import OnlineLearning.QLearn_Online as ql
import OfflineLearning.QLearn_Offline as ql_off
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools

BREAK_CON = 0.00008

def OnlineLearning(lr: float, df: float, er: float, num_episodes: int, penalty: float) -> dict:
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


    # Create dictionary to use to plot all lr, df, and er values on same plot
    results = {
        'lr': lr,
        'df': df,
        'er': er,
        'penalty': penalty,
        'returns': returns,
        'q_changes': q_changes,
        'policy_stable': policy_stable
    }

    return results

def OfflineLearning(grid, agent, data_set, epoch: int, lr: float, df: float, penalty: float) -> dict:
    """
    Runs offline Q-learning in the GridWorld environment and tracks metrics
    """

    # iv) Update Offline Qlearning
    q_table = agent.offline_q_learning(grid, data_set)
    

    # Final metrics output
    window = 10
    print("\n=== Metrics (last {} episodes) ===".format(window))
    print("Q-value changes    :", [f"{v:.6f}" for v in agent.q_changes[-window:]])
    print("Policy stability   :", [f"{v:.2f}" for v in agent.policy_stable[-window:]])

    # Show final Q-Action pair for each grid
    offline_best_output = agent.display_policy(grid, q_table, penalty)

    # Print by symbolic best policy by row
    for row in offline_best_output:
        print(row)


    # Create dictionary to use to plot all lr and df values on same plot
    results = {
        'lr': lr,
        'df': df,
        'er': 0.0,
        'penalty': penalty,
        'returns': agent.returns,
        'q_changes': agent.q_changes,
        'policy_stable': agent.policy_stable
    }

    return results


def plot_metrics(results, variable: str, subtitle: str):
    """
    Plots curves for a singular parameter sweep rates

    Args:
            results (list): The environment to run the training
            variable (str): Determines what parameter is being varied per plot
            subtitle (str): Displays the type of QLearning being run

    """
    
    plt.figure(figsize=(12, 4))
    linestyles = ['-', '--', ':']
    lines = itertools.cycle(linestyles)
    marker_list = ['.', 'x', 's', '^', '+', 'D', 'o']
    markers = itertools.cycle(marker_list)

    # Check if there is no varying hyper parameter
    if variable == None:
        plt.suptitle(f'{subtitle}')
    else:
        plt.suptitle(f'{subtitle} for Varied - {variable.upper()}')

    # Plot Returns
    plt.subplot(1, 3, 1)
    
    # Plot all results from dictionary
    for res in results:
        episodes = range(1, len(res['returns'])+1)
        marker = next(markers)
        line = next(lines)
        
        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Return'
            plt.plot(episodes, res['returns'], label=plt_label)
        else:
            plt_label = f'{variable.upper()} = {res[variable]}'
            plt.plot(episodes, res['returns'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.legend()
    plt.grid(True)

    # Plot Average change in Q
    plt.subplot(1, 3, 2)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)

    # Plot all results from dictionary
    for res in results:
        episodes = range(1, len(res['q_changes'])+1)
        marker = next(markers)
        line = next(lines)
        
        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = '$\delta$Q'
            plt.plot(episodes, res['q_changes'], label=plt_label, color='orange')
        else:
            plt_label = f'{variable.upper()} = {res[variable]}'
            plt.plot(episodes, res['q_changes'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Avg |$\delta$Q|")
    plt.title("Q-value Changes")
    plt.legend()
    plt.grid(True)

    # Plot Returns
    plt.subplot(1, 3, 3)
    markers = itertools.cycle(marker_list) # Reset Marker list
    lines = itertools.cycle(linestyles)
    
    # Plot all results from dictionary
    for res in results:
        episodes = range(1, len(res['policy_stable'])+1)
        marker = next(markers)
        line = next(lines)

        # Check if there is no varying hyper parameter
        if variable == None:
            plt_label = 'Stability'
            plt.plot(episodes, res['policy_stable'], label=plt_label, color='green')
        else:
            plt_label = f'{variable.upper()} = {res[variable]}'
            plt.plot(episodes, res['policy_stable'], label=plt_label, alpha=0.85, marker=marker, markersize=2, linestyle=line)

    plt.xlabel("Episode")
    plt.ylabel("Stable fraction")
    plt.title("Policy Stability")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    

def main():
    lr_ = [0.05, 0.1, 0.3]
    df_ = [0.8, 0.95,0.99]
    er_ = [0.05,0.2, 0.5]
    p = -1
    lr_results, df_results, er_results = [], [], [] # Create empty lists to store different variable results
    
    num_episodes = 500 # number of episodes for training

    # Run online learning
    tik = time.time()

    for i in lr_:
        online_res = OnlineLearning(i, df_[1], er_[1], num_episodes,penalty=p)
        lr_results.append(online_res)

    for j in df_:
        online_res = OnlineLearning(lr_[1], j, er_[1], num_episodes,penalty=p)
        df_results.append(online_res)
            
    for k in er_:
        online_res = OnlineLearning(lr_[1], df_[1], k, num_episodes,penalty=p)
        er_results.append(online_res)

    tok = time.time()
    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')

    # Plot all results in one figure
    plot_metrics(lr_results, variable='lr', subtitle=f'Online - Penalty {p:.2f}')
    plot_metrics(df_results, variable='df', subtitle=f'Online - Penalty {p:.2f}')
    plot_metrics(er_results, variable='er', subtitle=f'Online - Penalty {p:.2f}')
    
    
    tik = time.time()
    # Parameters
    lr = 0.1 # learning rate example
    df = 0.9 # discount factor example
    er = 0.2 # exploration rate example
    p = -200
    
    # Run online learning
    results = []
    online_res = OnlineLearning(lr, df, er, num_episodes,penalty=p)
    results.append(online_res)

    tok = time.time()
    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')

    plot_metrics(results, variable=None, subtitle=f'Online - Penalty {p:.2f}')


    ################################################################################
    # Start of Offline Q-learning
    ################################################################################

    # Parameters for Offline Learning
    #lr_ = [0.05, 0.1, 0.3]
    lr_ = [0.3, 0.1, 0.05]
    #df_ = [0.8, 0.95,0.99]
    df_ = [0.99, 0.95,0.8]
    penalty = -1            # Value in penalty state
    episodes = 100          # Number of episodes for dataset creation (used to train off of)
    epoch = 100             # Number of times to run Qlearning for offline
    
    # i) Create grid world environment
    grid = g.GridWorld(penalty = penalty)
    
    # ii) Create Q-learning offline agent
    agent = ql_off.QLearningOffline(grid, epoch=epoch, learning_rate=lr, discount_factor=df)

    # iii) Create dataset to train on made up of a list of tuples containing the current state, action, reward, and next state
    # Set outside of Offline Learning function so all data is trained off the same dataset and reduces run time
    data_set = agent.generate_training_dataset(grid, episodes)

    lr_results, df_results = [], []

    # Time how long it takes to run offline learning
    tik = time.time()  # Start time

    for i in lr_:
        offline_res = OfflineLearning(grid, agent, data_set, epoch, i, df_[1], penalty=penalty)
        lr_results.append(offline_res)

    for j in df_:
        offline_res = OfflineLearning(grid, agent, data_set, epoch, lr_[1], j, penalty=penalty)
        df_results.append(offline_res)
            

    tok = time.time()  # End time

    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')

    # Plot all results in one figure
    plot_metrics(lr_results, variable='lr', subtitle=f'Offline - Penalty {penalty:.2f}')
    plot_metrics(df_results, variable='df', subtitle=f'Offline - Penalty {penalty:.2f}')


    ### Offline Learning Large Penalty #############################################
    
    # Parameters for Offline Learning
    lr = 0.1           # learning rate
    df = 0.9           # discount factor
    pen = -200         # Value in penalty state
    epoch = 100        # Number of times to run Qlearning for offline

    # i) Create grid world environment
    grid = g.GridWorld(penalty = penalty)
    
    # ii) Create Q-learning offline agent
    agent = ql_off.QLearningOffline(grid, epoch=epoch, learning_rate=lr, discount_factor=df)

    # iii) Create dataset to train on made up of a list of tuples containing the current state, action, reward, and next state
    # Set outside of Offline Learning function so all data is trained off the same dataset and reduces run time
    data_set = agent.generate_training_dataset(grid, episodes)

    # Time how long it takes to run offline learning
    tik = time.time()  # Start time
    results = []
    offline_res = OfflineLearning(grid, agent, data_set, epoch, lr, df, penalty=pen)  # Run online learning
    results.append(offline_res)
    tok = time.time()  # End time

    runtime = tok-tik
    print(f'Runtime: {runtime:.3f} seconds')

    plot_metrics(results, variable=None, subtitle=f'Offline - Penalty {pen:.2f}')


if __name__ == "__main__":
    main()

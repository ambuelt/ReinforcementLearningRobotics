################################################################################
# Assignment : HW1                                                             #
# Team #3                                                                      #
# ---------------------------------------------------------------------------  #
#                                                                              #
# Program initializes q_table values for offline learning                      #
# Displays the symbolic best policy                                            #
# Generates the data to train offline learning on                              #
# Runs Offline Learning Bellman Equations                                      #
# Calculates performance metrics to be graphed in main                         #
################################################################################

import numpy as np
from typing import Tuple

BREAK_CON = 0.00008

class QLearningOffline:
    def __init__(self, grid, epoch: int, learning_rate: float, discount_factor: float) -> None:
        """
        Initializes the Q-Learning Online agent.

        Args:
            grid (GridWorld): The environment to run the training
            epoch (int): Number of tests to run on generated dataset
            learning_rate (float): The rate at which the agent learns.
            discount_factor (float): The factor by which future rewards are discounted.

        """

        # Parameters for Offline Learning
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Parameters for metrics tracking
        self.q_val_delta = 0      # Mean of delta between Qtable iterations
        self.q_changes = []       # average |delta Q| per episode
        self.policy_stable = []   # fraction of states with unchanged greedy action
        self.prev_policy = None   # Tracks if the policy changes
        self.returns = []         # total reward per episode
        
    def initialize_q_table(self, grid) -> np.ndarray:
        """
        Creates a 2D array of size val = 0.0 states and number of actions (in this case 9x4) to act as the Qtable

        Qtable Example:

                        Action
                    0   1   2   3
        State
        1 = (0,0)   r   r   r   r
        2 = (0,1)   r   r   r   r
        3 = (0,3)   r   r   r   r
              .     .   .   .   .
              .     .   .   .   .
              .     .   .   .   .
        9 = (2,2)   r   r   r   r
    
        Returns:
            np.array: An empty 2D array to be filled by reward value in each state
        """

        q_table_rows = []    # Stores the number of rows in the final Qtable

        # Checks for each position in grid
        for row in range(grid._rows):
            for col in range(grid._cols):

                # Grabs value from current position
                val = grid.grid[row, col]

                # Checks to see if current position is viable for agent movement
                if (not np.isnan(val)) and (val == 0.0):
                    q_table_rows.append((row, col))
                else:
                    continue

        # Create dictionary of valid states to get 9 rows for Qtable
        # Each row in the Qtable will represent the specific coordinate
        self._valid_state_index = {state: i
                                   for i, state in enumerate(q_table_rows)
                                  }
        
        q_table = np.zeros((len(q_table_rows), len(grid.actions)), dtype = float)
        return q_table

    
    def display_policy(self, grid, q_table) -> list:
        """
        Creates the dataset for the offline learning algorithm to use to determine best action per state
    
        Args:
            q_table (np.array): 
                
        Returns:
            list: Contains final arrow directional for best action at positions in grid
        """

        # Create dictionary to visually display best policy
        symbol = {0: '↑', 
                  1: '→', 
                  2: '↓', 
                  3: '←'}
        
        symbol_grid = []   # Stores the final arrow, WALL, or value strings for policy display

        # Runs through all grid positions
        for row in range(grid._rows):
            disp_row = []   # Stores every row in the grid (acts as a reset)
            for col in range(grid._cols):

                # Grabs value from current state
                state = (row, col)
                val = grid.grid[state]

                # If state is terminal, stay there
                if grid.is_terminal(state):
                    if val == grid.penalty:
                        disp_row.append(str(val))    # display penalty state
                    else:
                        disp_row.append('+1')        # display final reward state

                # If state is an obstacle
                elif np.isnan(val):
                    disp_row.append('Wall')          # display wall state
                
                # If in state where you can move, determine best action (0-3) and add that symbol
                else:
                    state_index = self._valid_state_index[state]  # Grab r, c combination

                    # Get best action + add to row list
                    best_action_position = np.argmax(q_table[state_index, :])
                    disp_row.append(symbol[best_action_position])
            
            symbol_grid.append(disp_row)   # add current row to display output

        # Reverse grid so it is display like in word doc position-wise
        symbol_grid.reverse()

        return symbol_grid


    def generate_training_dataset(self, grid, trials: int = 100) -> list[Tuple[Tuple[int, int], int, float, Tuple[int, int]]]:
        """
        Creates the dataset for the offline learning algorithm to use to determine best action per state
    
        Args:
            grid (GridWorld): The environment to run the training
            trials (int): The number of complete paths from start state to terminal state
                
        Returns:
            list[Tuple[Tuple[int, int], int, float, Tuple[int, int]]]: A list of tuples containing the current state, action, reward, and next state
        """

        actions = grid.actions  # Set up directional choices: 0=North, 1=East, 2=South, 3=West   
        data_set = []           # Set up empty dataset to host training trials
    
        # Run all trials to create large dataset for offline q_learning
        for trial in range(trials):
            state = grid.reset()      # initialize state to starting state
            done = False              # since currently in starting state, not done
    
            # Run until termincal state (penalty or +1) is reached
            while not done:
                # i) Select random action
                action = int(np.random.choice(actions))
    
                # ii)  Takes a step in the grid environment
                # Tracks reward and next state
                next_state, reward, done = grid.step(action)
    
                # iii) Adds trial to list of tuple trials
                data_set.append((state, action, reward, next_state))
    
                # iv) Sets current state to next state (transistion)
                state = next_state
    
        return data_set
    
    
    def offline_q_learning(self, grid, data_set) -> np.ndarray:
        """
        Creates the dataset for the offline learning algorithm to use to determine best action per state
    
        Args:
            grid (GridWorld): The environment to run the training
            data_set (list[tuples]): The list of tuples containing (current state, action, reward, and next state) to test
                
        Returns:
            ndarray: A 2D array containing rewards in Qtable
        """

        q_table = self.initialize_q_table(grid)       # Initialize empty 9x4 Qtable
        valid_state_index = self._valid_state_index   # Get dictionary containing all viable agent spaces for indexing
        q_table_dim = np.size(q_table)                # Gets the size of q_table to use to calc mean
        
        # Run trials to test different paths and shuffle training data for less bias
        for trial in range(self.epoch):
            np.random.shuffle(data_set)   # Shuffle data for each pass to ensure better training

            total_reward = 0.0            # Track reward as Qtable is navigated per epoch
            deltas = []                   # Track Q-value changes in this episode
    
            # Run Q learning for each state within the data set to determine best action using Bellmans Equation
            for (state, action, reward, next_state) in data_set:
                
                # Don't neet to know best direction once terminal states are reached! You want to stay there!!
                if grid.is_terminal(state):
                    continue
    
                state_index = valid_state_index[state]
    
                # Determine current action-value
                q_sa = q_table[state_index, action]

                # Use as reference for delta (used to save runtime over copying full table)
                old_q_sa = q_sa
    
                # Determine if next step is going to be a wall, penalty, or final reward state
                if grid.is_terminal(next_state):
                    max_next = 0.0

                # Use Q(s, a) ← Q(s, a) + alpha * [ r + gamma max_{a'} Q(s', a') - Q(s, a) ] equation
                else:
                    next_state_index = valid_state_index[next_state]
                    max_next = float(np.max(q_table[next_state_index, :]))
                
                # [ r + gamma max_{a'} Q(s', a') - Q(s, a) ]
                target = (reward + (self.discount_factor * max_next)) - q_sa
        
                # Q(s, a) ← Q(s, a) + alpha [ target ]  <-- updates Qtable
                q_table[state_index, action] = q_sa + self.learning_rate * (target)
                new_q_sa = q_table[state_index, action]

                # Determine how much Qsa has changed upon updating and take average change across Qtable per best action step
                diff = np.abs(new_q_sa - old_q_sa)
                deltas.append(diff / q_table_dim)
            
            # At the end of each runthrough of the dataset, track metrics which fixes 200000 iterations (rewards still wrong)
            self.offline_calc_metrics(q_table, deltas)

            # Check to see if path converges
            if self.q_val_delta <= BREAK_CON:
                break

        return q_table
    

    def offline_calc_metrics(self, q_table, deltas) -> None:
        """
        Calculates the Average Q Value change, policy stability, and total reward per policy
    
        Args:
            grid (GridWorld): The environment to run the training
            q_table (ndarray): Holds the updated reward values
            deltas (list[floats]): Tracks the difference between previous Qtable and current Qtable
            total_reward (float): Tracks the total reward per policy path              
        
        """

        # Track average Q value change for convergence
        self.q_val_delta = np.mean(deltas)
        self.q_changes.append(self.q_val_delta)

        # Determine current iteration's best action per state
        greedy_policy = np.argmax(q_table, axis=1)

        # If first run, no policy to compare to
        if self.prev_policy is None:
            stable_frac = 0.0
        else:
            stable_frac = np.mean(greedy_policy == self.prev_policy)

        self.policy_stable.append(stable_frac)

        # Sets current best policy to be previous policy for iteration comparison
        self.prev_policy = greedy_policy.copy()



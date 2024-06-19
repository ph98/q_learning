# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


navigations = [ (0, 1), (0, -1), (1, 0), (-1, 0) ]


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    state_space_size = (env.height, env.width, 2)
    q_table = np.zeros(state_space_size + (env.action_space.n,))


    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------

    count = [0,0]
    for episode in range(no_episodes):
        state, info = env.reset()
        _,_, has_food = state


        count[has_food] += 1
        state = tuple(state)
        total_reward = 0


        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, info = env.step(action)
            # env.render()
            next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] += alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])

            # if(not has_food and has_food_next_state):
            #     break

            state = next_state
            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break

        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if(episode % 100 == 0):
            print(f"Episode {episode}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")

    print(count)


arrowOfAction = ['→', '←', '↓', '↑']

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(hell_state_coordinates=[(2, 1), (0, 4)],
                      goal_coordinates=(4, 4),
                      restourantPosition=(0, 4),
                      actions=["Right", "Left", "Down", "Up"],
                      q_values_path="q_table.npy",
                      treeArray=[]
                      ):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(3, 4, figsize=(20, 10))

        for j in range(2):
            for i, action in enumerate(actions):
                ax = axes[j][i]
                heatmap_data = q_table[:, :,j, i].copy()
                # Mask the goal state's Q-value for visualization:
                # ------------------------------------------------
                mask = np.zeros_like(heatmap_data, dtype=bool)
                mask[goal_coordinates] = True

                for i in range(len(hell_state_coordinates)):
                    mask[hell_state_coordinates[i]] = True

                for i in range(len(treeArray)):
                    mask[treeArray[i]] = True
                # mask[restourantPosition] = True

                sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                            ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

                temp_added_value = 0.5
                # Denote Goal and Hell states:
                # ----------------------------
                ax.text(goal_coordinates[1] + temp_added_value, goal_coordinates[0] + temp_added_value, 'C', color='green',
                        ha='center', va='center', weight='bold', fontsize=14)
                
                for i in range(len(hell_state_coordinates)):
                    ax.text(hell_state_coordinates[i][1] + temp_added_value, hell_state_coordinates[i][0] + temp_added_value, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

                for i in range(len(treeArray)):
                    ax.text(treeArray[i][1] + temp_added_value, treeArray[i][0] + temp_added_value, 'T', color='brown',
                        ha='center', va='center', weight='bold', fontsize=14)
                # ax.text(hell_state_coordinates[0][1] + temp_added_value, hell_state_coordinates[0][0] + temp_added_value, 'H', color='red',
                #         ha='center', va='center', weight='bold', fontsize=14)
                # ax.text(hell_state_coordinates[1][1] + temp_added_value, hell_state_coordinates[1][0] + temp_added_value, 'H', color='red',
                #         ha='center', va='center', weight='bold', fontsize=14)

                ax.text(restourantPosition[1] + temp_added_value, restourantPosition[0] , 'R', color='white',
                        ha='center', va='top', weight='bold', fontsize=12)
                
                ax.set_title(f'{j == 0 and 'Without food' or 'With food'}, Action: {action}')

     

        mask = np.zeros_like(q_table[:, :, 0, 0], dtype=bool)
        mask[goal_coordinates] = True
        for i in range(len(hell_state_coordinates)):
            mask[hell_state_coordinates[i]] = True
        for i in range(len(treeArray)):
            mask[treeArray[i]] = True
        optimal_policy = np.argmax(q_table, axis=3)
        for has_food_index in range(2):
            # show the plot for the final answer using arrows:
            optimal_policy_indices_array = [(0,0)]
            ax = axes[2][has_food_index]
            # find the index of the maximum Q-value for each state
            optimal_policy = np.argmax(q_table[:,:,has_food_index,:], axis=2)
            sns.heatmap(np.ones_like(optimal_policy), annot=False, fmt="", cmap="viridis",
                ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})
            # write the desired action on each state
            for i in range(optimal_policy.shape[0]):
                for j in range(optimal_policy.shape[1]):
                    if (i, j) == goal_coordinates:
                        ax.text(j + temp_added_value, i + temp_added_value, 'C', color='green',
                                ha='center', va='center', weight='bold', fontsize=14)
                    elif (i, j) in hell_state_coordinates:
                        ax.text(j + temp_added_value, i + temp_added_value, 'H', color='red',
                                ha='center', va='center', weight='bold', fontsize=14)
                    elif (i, j) in treeArray:
                        ax.text(j + temp_added_value, i + temp_added_value, 'T', color='brown',
                                ha='center', va='top', weight='bold', fontsize=10)
                    else:
                        color = 'white'
                        ax.text(j + temp_added_value, i + temp_added_value, arrowOfAction[optimal_policy[i, j]], color=color,
                                ha='center', va='center', weight='bold', fontsize=14)
                        
                    if (i, j) == restourantPosition:
                        ax.text(j + temp_added_value, i , 'R', color='white',
                                ha='center', va='top', weight='bold', fontsize=12)
                                
            
            title = f'Optimal Policy: {"without" if has_food_index == 0 else "with"} food'
            ax.set_title(title)
        
    
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

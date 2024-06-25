# Imports:
# --------
from assignment1Delivery import create_env
from q_learning import train_q_learning, visualize_q_table, render_result

# User definitions:
# -----------------
train = False
visualize_results = True
render_results = True

learning_rate = 0.2  # Learning rate
gamma = 0.9 # Discount factor
epsilon = 1 # Exploration rate
epsilon_min = 0.1 # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1_000  # Number of episodes

goal_coordinates = (8, 7)
restourantPosition = (1, 9)
# Define all hell state coordinates as a tuple within a list
hell_state_coordinates = [(1, 2), (4, 0)]

treesArray = [
            (1, 1), (2, 5),
            (3, 1), (2, 2), (3, 3), 
            (9, 4), (9, 5),
            (5,9), (6, 9),
            (1, 6), (1, 8),
            (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7)
        ]

# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(goal_coordinates=goal_coordinates,
                     restourantPosition=restourantPosition,
                     hell_state_coordinates=hell_state_coordinates,
                     tree_array=treesArray
                     )

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      restourantPosition=restourantPosition,
                      q_values_path="q_table.npy",
                      treeArray=treesArray
                      )
if render_results:
    # Render the results:
    # -------------------
        
    env = create_env(goal_coordinates=goal_coordinates,
                        restourantPosition=restourantPosition,
                        hell_state_coordinates=hell_state_coordinates,
                        tree_array=treesArray
                        )
    render_result(
        env=env,
        q_table_path="q_table.npy",
    )
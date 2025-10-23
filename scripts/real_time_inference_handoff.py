""""
This script is intended to preprocess a single scenario file and its 
corresponding map file for real-time prediction using advisory congestion model. 

---
Only for inference purposes, not for training!
---
"""

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
import ast
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import torch     
import argparse

# file_id = "room-32-32-4"
# file_id = "maze-32-32-4"
file_id = "random-32-32-20"
scen__ = "random-10"  # This can be set to a specific scenario for testing, e.g., "even-25", "odd-25", etc.
agents = 111 # This can be set to a specific agent number for testing, e.g., 1, 2, etc.



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help='Path to map file')
    parser.add_argument('--scen', type=str, required=True, help='Path to scenario file')
    parser.add_argument('--num', type=int, required=True, help='Number of agents')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save output files')
    args = parser.parse_args()

    map_file = args.map
    scen_file = args.scen
    agents = args.num
    save_dir = args.save_dir
    
    # scen_file = f'./data/raw/{file_id}/{file_id}-{scen__}.scen' #This scene file can be given as input from bash

    #This becomes static input for this map file -- Random 32x32-20 map
    # map_file is something like: data/raw/random-32-32-20/random-32-32-20.map
    file_id = os.path.basename(os.path.dirname(map_file))
    # random_32_32_df = pd.read_csv(f"./data/raw/{file_id}/{file_id}.csv")
    csv_path = f"./data/raw/{file_id}/{file_id}.csv"
    random_32_32_df = pd.read_csv(csv_path)
    random_32_32_df['unique_id'] = random_32_32_df['scen_type'] + '_' + random_32_32_df['type_id'].astype(str) + '_agent_' + random_32_32_df['agents'].astype(str)
    random_32_32_df['unique_id'] = random_32_32_df['unique_id'].astype(str)
    
    # grid, wdith, height = parse_map(f"./data/raw/{file_id}/{file_id}.map") 
    grid, wdith, height = parse_map(map_file) 
    
    # Parse the scenario file
    # This function reads the scenario file and returns a DataFrame with relevant columns.
    scen_df = parse_scen_file(scen_file)
    scen_id = scen_file.split(f'{file_id}-')[-1].split('.scen')[0]
    scen_id = scen_id.replace('-', '_')
    scen_df = preprocess_scen(scen_df, scen_id)
    scen_df = scen_df.iloc[:agents] # Limit to the specified number of agents

    # Create a reference DataFrame for start and goal locations
    # This DataFrame will be used to match the agents in the scenario file with the agents
    ref_loc_df = scen_df[['unique_id', 'start_location', 'goal_location']].copy()
    ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str) 
    ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str)

    ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
    ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)


    #Filter the random_32_32_df to only include unique_ids present in the scenario file
    # This ensures that we only work with the agents defined in the scenario file.
    random_32_32_df = random_32_32_df[random_32_32_df['unique_id'].isin(ref_loc_df['unique_id'])].iloc[-1]
    solution_plan = random_32_32_df['solution_plan'].split('\n')
    makespan = max([len(plan) for plan in solution_plan])

    # Transpose the grid for aligning with standard positional indexing
    # This is necessary because the grid is represented in a way that matches the scenario file.
    grid_1 = grid.T.copy()
    grid_2 = grid.T.copy()
    simulation_grid = grid.T.copy()

    start_locations = ref_loc_df.start_location.tolist()
    goal_locations = ref_loc_df.goal_location.tolist()

    for agent in range(1, agents+1):
        for timestep in range(makespan):
            curr_pos = calculate_curr_position(timestep, solution_plan[agent - 1], start_locations[agent-1])
        
            # Ensure position is within bounds
            if 0 <= curr_pos[0] < simulation_grid.shape[0] and 0 <= curr_pos[1] < simulation_grid.shape[1]:
                if simulation_grid[curr_pos] >= 0:
                    simulation_grid[curr_pos] += 1
                else:
                    print('skip counter initialised')
                    skip_counter += 1
        
            # Stop if agent reaches goal exactly at the end
            if timestep == len(solution_plan[agent - 1]):
                if goal_locations[agent-1] == curr_pos:
                    break

    # Apply Min-Max Scaling, preserving obstacles
    simulation_grid = min_max_scaling(simulation_grid)

    for start_loc, goal_loc in zip (start_locations, goal_locations):
        grid_1[start_loc] = 1.5
        grid_2[goal_loc] = 1.5

    start_time = time.time()
    x_field, y_field = create_aggregate_direction_fields(start_locations, goal_locations, grid_1.shape)

    simulation_grid = gaussian_filter(simulation_grid, sigma=0.6)  # Adjust sigma for smoothness
    congestions = simulation_grid.T
    grid_agent = grid_1.T
    grid_goal = grid_2.T
    x_field = x_field.T
    y_field = y_field.T# Normalize the grid to a range of 0 to 1

    # So until here, we have preprocessed the scenario file and the map file, and created the simulation grid. Ready for inference.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    # model = DualInputTopologyVectorFields(in_channels=2, hidden_dim=64).to(device)
    # model.load_state_dict(torch.load("training_logs/training_20250319-000457/best_model.pth")['model_state_dict'])
    # model = torch.load("model/best_model_cpp.pt")
    model = torch.jit.load("model/best_model_cpp.pt", map_location=device)  # Load the model with the correct device mapping
    model.eval() 

    with torch.no_grad(): 

        # Convert numpy arrays to torch tensors and add channel and batch dimensions
        grid_agent = torch.from_numpy(grid_agent).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 32)
        grid_goal = torch.from_numpy(grid_goal).float().unsqueeze(0).unsqueeze(0)    # (1, 1, 32, 32)
        x_field = torch.from_numpy(x_field).float().unsqueeze(0).unsqueeze(0)        # (1, 1, 32, 32)
        y_field = torch.from_numpy(y_field).float().unsqueeze(0).unsqueeze(0)        # (1, 1, 32, 32)

        # Concatenate along the channel dimension
        topology_input = torch.cat((grid_agent, grid_goal), dim=1)         # (1, 2, 32, 32)
        vector_field_input = torch.cat((x_field, y_field), dim=1)          # (1, 2, 32, 32)

        # Move to device
        topology_input = topology_input.to(device)
        vector_field_input = vector_field_input.to(device)
        congestions = torch.from_numpy(congestions).float().unsqueeze(0).unsqueeze(0).to(device)  # If needed

        print(topology_input.shape, vector_field_input.shape)
        reconstructed = model(topology_input, vector_field_input)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5.5))

        # Heatmap for the normal grid (raw occupancy map)
        sns.heatmap(grid_agent.cpu().squeeze((0,1)), cmap='Reds', linecolor='black', linewidth=0.4, ax=ax1)
        ax1.set_title("Original Grid (Raw Map)")

        # Heatmap for the simulation grid (processed congestion map)
        sns.heatmap(reconstructed.cpu().squeeze((0,1)), cmap='Reds', linecolor='black', linewidth=0.4, ax=ax2)
        ax2.set_title("Occupancy Model Output (Reconstructed)")

         # Heatmap for the simulation grid (processed congestion map)
        sns.heatmap(congestions.cpu().squeeze((0,1)), cmap='Reds', linecolor='black', linewidth=0.4, ax=ax3)
        ax3.set_title("Ground Truth Occupancy")

        # Adjust layout and show the figure
        plt.tight_layout()
        fig.savefig(f"{save_dir}/model_occupancy.png", bbox_inches='tight', dpi=300)

        # plt.show()
        plt.close(fig)

        # Save figure

    t = 1

    # Need to save the reconstructed grid and original grid as binary files for further processing or analysis.
    print(f"Saving reconstructed grid and original grid for file_id: {file_id}, scen_id: {scen_id}, agents: {agents}")
    # Save the reconstructed and original grid as binary files (.bin)
    reconstructed.cpu().numpy().tofile(f"{save_dir}/{scen_id}_agent_{agents}_reconstructed.bin")
    grid_agent.cpu().numpy().tofile(f"{save_dir}/{scen_id}_agent_{agents}_original.bin")



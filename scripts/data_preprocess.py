import pandas as pd 
import numpy as np
import glob, os
import ast
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt

def parse_scen_file(fp):
    scen_df = pd.DataFrame(columns=['unknown_var', 'map', 'map_height', 'map_width',
                           'start_location_x', 'start_location_y',
                           'goal_location_x', 'goal_location_y', 'unknown_heuristic'])
    with open(fp, 'r') as f:
        scen_file = f.readlines()[1:]
        
    for line in scen_file:
        line = line.strip().split('\t')
        scen_df.loc[len(scen_df)] = line
    
    return scen_df

def preprocess_scen(scen_df, scen_id):
    scen_df = scen_df.astype({
          'sif tart_location_x': int,
          'start_location_y': int,
          'goal_location_x': int,
          'goal_location_y': int,
          'unknown_heuristic': float
        })
    scen_df['start_location'] = scen_df.apply(lambda row: (row['start_location_x'], row['start_location_y']), axis=1)
    scen_df['goal_location'] = scen_df.apply(lambda row: (row['goal_location_x'], row['goal_location_y']), axis=1)

    scen_df['unique_id'] = [scen_id + '_agent_' + str(i+1)  for i in range(len(scen_df))]
    scen_df.drop(columns=['start_location_x',
       'start_location_y', 'goal_location_x', 'goal_location_y'], inplace=True)
    return scen_df

def build_consolidated(scen_file_list):
    consolidated_scen_df = pd.DataFrame()
    
    for file in scen_file_list:
        scen_id = file.split('random-32-32-20-')[-1].split('.scen')[0]
        scen_id = scen_id.replace('-', '_')
        scen_df = parse_scen_file(file)
        scen_df = preprocess_scen(scen_df, scen_id)
        consolidated_scen_df = pd.concat([consolidated_scen_df, scen_df])
        
    return consolidated_scen_df

def un_concat_paths(results_df):
    optimal_path = []
    for i in range(len(results_df)):
        agent_no = results_df.iloc[i]['agents'] - 1
        try:
            path = results_df['solution_plan'][i].split("\n")[agent_no]
        except IndexError:
            print('Total number of solution plans are: ',
                  len(results_df.iloc[i]['solution_plan'].split('\n')),
                  'but total number of agents are:', 
                  results_df.iloc[i]['agents']
                 )
            path = None # Assigning NA path for agents that have possibly reached goal state.
        optimal_path.append(path)
    results_df['solution_plan'] = optimal_path
    return results_df

def calculate_goal_location(df, plan_col='solution_plan', start_col='start_location'):
    calculated_goal_location = []
    for i in range(len(df)):
        row = df.iloc[i]
        x, y = row['start_location']
        plan = row['solution_plan']
        uid = row['unique_id']
        
        if plan == None:
            calculated_goal_location.append(None)
            continue
        x += plan.count('r') - plan.count('l')
        y += plan.count('u') - plan.count('d')
        
        calculated_goal_location.append((x, y))
        
    df['calculated_goal_location'] = calculated_goal_location
    
    return df


def calculate_curr_position(timestep, agent_solution_plan, start_loc):
    plan = agent_solution_plan[:timestep]
    x, y = start_loc
    x += plan.count('r') - plan.count('l')
    y += plan.count('u') - plan.count('d')
    return (x, y)

def parse_map(map_file):
    # Read the map file
    with open(map_file, 'r') as f:
        map_data = f.readlines()
    map_data = map_data[4:]  # Skip the map header

    width = len(map_data[0].strip())  # Infer width from the map
    height = len(map_data)

    # Initialize the grid representation
    grid = np.zeros((height, width))
    for row_idx in range(height):
        for col_idx in range(width):
            grid[row_idx, col_idx] = 0 if map_data[row_idx][col_idx] == '.' else 1

    return grid, width, height

def min_max_scaling(grid, mask_value=-10):
    # Create a mask to identify non-obstacle values
    mask = (grid != mask_value)
    
    # Extract only valid values
    grid_mask = grid[mask]
    
    # Compute min and max from non-obstacle values
    grid_min = grid_mask.min()
    grid_max = grid_mask.max()
    
    # Create a float copy of the grid (so obstacle values remain unchanged)
    scaled_grid = grid.astype(float)

    # Apply Min-Max Scaling only on non-obstacle values
    scaled_grid[mask] = (grid_mask - grid_min) / (grid_max - grid_min)
    
    return scaled_grid


def create_direction_field(start, goal, map_shape):
    """
    Creates a direction field showing the movement intention
    from start to goal across the entire map.
    
    Args:
        start: tuple (x, y) of start position
        goal: tuple (x, y) of goal position
        map_shape: tuple (height, width) of the map
    
    Returns:
        Tuple of (field_x, field_y) 2D numpy arrays with direction components
    """
    height, width = map_shape
    
    # Calculate the direction vector from start to goal
    dir_vector = np.array([goal[0] - start[0], goal[1] - start[1]])
    
    # Normalize if not zero
    norm = np.linalg.norm(dir_vector)
    if norm > 0:
        dir_vector = dir_vector / norm
    
    # Create coordinate matrices for vectorized operations
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate distances from each cell to the start position (vectorized)
    distances = np.sqrt((x_coords - start[0])**2 + (y_coords - start[1])**2)
    
    # Add a small constant to avoid division by zero
    distances = np.maximum(distances, 0.1)
    
    # Calculate influence (inversely proportional to distance squared)
    influence = 1.0 / (distances**2)
    
    # Compute field components using broadcasting
    field_x = dir_vector[0] * influence
    field_y = dir_vector[1] * influence  # Fixed: was using dir_vector[0] before
    
    # Normalize the fields
    x_max = np.max(np.abs(field_x))
    if x_max > 0:
        field_x = field_x / x_max
        
    y_max = np.max(np.abs(field_y))
    if y_max > 0:
        field_y = field_y / y_max
        
    return field_x, field_y

def create_aggregate_direction_fields(starts, goals, map_shape):
    """
    Creates aggregate direction fields for multiple agents.
    
    Args:
        starts: List of (x, y) tuples for start positions
        goals: List of (x, y) tuples for goal positions
        map_shape: tuple (height, width) of the map
        
    Returns:
        Tuple of (x_field, y_field) containing the aggregate directional information
    """
    height, width = map_shape
    x_field = np.zeros((height, width))
    y_field = np.zeros((height, width))
    
    # Process all agents in one go if there are many
    if len(starts) > 10:
        # Pre-allocate arrays for all agent fields
        all_x_fields = np.zeros((len(starts), height, width))
        all_y_fields = np.zeros((len(starts), height, width))
        
        # Calculate individual fields
        for i, (start, goal) in enumerate(zip(starts, goals)):
            all_x_fields[i], all_y_fields[i] = create_direction_field(start, goal, map_shape)
            
        # Sum across all agents
        x_field = np.sum(all_x_fields, axis=0)
        y_field = np.sum(all_y_fields, axis=0)
    else:
        # Original approach for smaller numbers of agents
        for start, goal in zip(starts, goals):
            field_x, field_y = create_direction_field(start, goal, map_shape)
            x_field += field_x
            y_field += field_y
    
    # Normalize the combined fields
    x_max = np.max(np.abs(x_field))
    if x_max > 0:
        x_field = x_field / x_max
        
    y_max = np.max(np.abs(y_field))
    if y_max > 0:
        y_field = y_field / y_max
        
    return x_field, y_field

if __name__ == "__main__":
    # Reading the raw random-32-32-20 scenario files, grid and results 
    scen_file_list = glob.glob("../data/raw/random-32-32-20/random-32-32-20/*.scen")
    random_32_32_df = pd.read_csv("../data/raw/random-32-32-20.csv")
    grid, wdith, height = parse_map("../data/raw/random-32-32-20/random-32-32-20.map") 
    
    random_32_32_df['unique_id'] = random_32_32_df['scen_type'] + '_' + random_32_32_df['type_id'].astype(str) + '_agent_' + random_32_32_df['agents'].astype(str)
   
    consolidated_df = build_consolidated(scen_file_list)
    random_32_32_df['unique_id'] = random_32_32_df['unique_id'].astype(str)
    consolidated_df['unique_id'] = consolidated_df['unique_id'].astype(str)
    merged_df = pd.merge(random_32_32_df, consolidated_df[['unique_id', 'start_location', 'goal_location', 'unknown_heuristic', 'unknown_var']],
                            on='unique_id')
    merged_df.to_csv("../data/processed/random_32_32_20_scen_results.csv", index=False)

    ref_loc_df = merged_df[['scen_type', 'type_id', 'agents', 'start_location', 'goal_location']]
    ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
    ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)

    ref_loc_df.to_csv("../data/processed/random_32_32_20_ref_locations.csv", index=False)

    # Model input data
    merged_df['makespan'] = [len(max(i.split('\n'), key=len)) for i in merged_df['solution_plan']]
    merged_df = merged_df.sample(len(merged_df)).reset_index()

    simulation_batches = []
    batch_idx = 0

    for idx, row in df.iterrows():
        try:
            agents = int(row['agents'])
            # if agents <= 2:
            #     continue
            makespan = int(row['makespan'])
            scen_type, type_id = row['scen_type'], row['type_id']
            solution_plans = row['solution_plan'].split('\n')
            # print(agents, makespan, scen_type, type_id)
            
            grid_1 = grid.T.copy()
            grid_2 = grid.T.copy()
            simulation_grid = grid.T.copy()
            empty_grid = np.zeros_like(grid_1)

            if sum(np.isnan(grid_1.flatten())) > 0:
                print('debug')

            # simulation_grid[grid_1 == 1] = -10  # Mark obstacles as -10
        
            start_locations = []
            goal_locations = []
            skip_counter = 0  # Move this outside the loop
        
            # Pre-fetch query before the loop
            df_query = ref_loc_df[
                (ref_loc_df['scen_type'] == scen_type) &
                (ref_loc_df['type_id'] == type_id)
            ]
        
            for agent in range(1, agents + 1):
                query = df_query[df_query['agents'] == agent]
                if query.empty:
                    continue  # Skip if no matching reference location
                
                start_loc = tuple(query['start_location'].values[0])
                start_locations.append(start_loc)

                goal_loc = tuple(query['goal_location'].values[0])
                goal_locations.append(goal_loc) 

                for timestep in range(makespan):
                    curr_pos = calculate_curr_position(timestep, solution_plans[agent - 1], start_loc)
        
                    # Ensure position is within bounds
                    if 0 <= curr_pos[0] < simulation_grid.shape[0] and 0 <= curr_pos[1] < simulation_grid.shape[1]:
                        if simulation_grid[curr_pos] >= 0:
                            simulation_grid[curr_pos] += 1
                        else:
                            skip_counter += 1
        
                    # Stop if agent reaches goal exactly at the end
                    if timestep == len(solution_plans[agent - 1]):
                        if tuple(query['goal_location'].values[0]) == curr_pos:
                            break
        
            # Apply Min-Max Scaling, preserving obstacles
            simulation_grid = min_max_scaling(simulation_grid)

            for start_loc, goal_loc in zip (start_locations, goal_locations):
                grid_1[start_loc] = 1.5
                grid_2[goal_loc] = 1.5

            x_field, y_field = create_aggregate_direction_fields(start_locations, goal_locations, grid_1.shape)
        
            simulation_grid = gaussian_filter(simulation_grid, sigma=0.6)  # Adjust sigma for smoothness
            simulation_grid = simulation_grid.T
            grid_1 = grid_1.T
            grid_2 = grid_2.T
            x_field = x_field.T
            y_field = y_field.T

            if sum(np.isnan(grid_1.flatten())) > 0:
                print('debug')
                
            stacked = np.stack((grid_1, x_field, y_field, grid_2, simulation_grid))
            simulation_batches.append(stacked)

            # # Create subplots
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))
        
            # # Heatmap for the normal grid (raw occupancy map)
            # sns.heatmap(grid_1, cmap='Reds', linewidth=0.4, ax=ax1)
            # ax1.set_title("Agent Map")

            # # Heatmap for the normal grid (raw occupancy map)
            # sns.heatmap(grid_2, cmap='Reds', linewidth=0.4, ax=ax2)
            # ax2.set_title("Goal Map")

            # # Heatmap for the simulation grid (processed congestion map)
            # sns.heatmap(simulation_grid, cmap='Reds', linewidth=0.4, ax=ax3)
            # ax3.set_title("Simulation Grid (Congestion Map)")
        
            # # Adjust layout and show the figure
            # plt.tight_layout()
            # plt.show()

            
            # Create subplots
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
            # # Heatmap for the normal grid (raw occupancy map)
            # sns.heatmap(grid_1, cmap='Reds', linewidth=0.4, ax=ax1)
            # ax1.set_title("Agent Map")

            # # Heatmap for the simulation grid (processed congestion map)
            # sns.heatmap(simulation_grid, cmap='Reds', linewidth=0.4, ax=ax2)
            # ax2.set_title("Simulation Grid (Congestion Map)")
        
            # # Adjust layout and show the figure
            # plt.tight_layout()
            # plt.show()
            
            
            if len(simulation_batches) % 32 == 0:
                # print(len(simulation_batches))
                #save
                np.save(rf"C:\Users\kanis\Documents\Monash\MAPF Research\MAPF-Machine_Learning\Data\Map_Encoding_Data5\batch_{batch_idx}_encoded.npy", simulation_batches)
                batch_idx += 1
                simulation_batches = []
                print('Batches saved', batch_idx)

        except IndexError:
            print('Index exception')
            continue
            
        np.save(rf"C:\Users\kanis\Documents\Monash\MAPF Research\MAPF-Machine_Learning\Data\Map_Encoding_Data5\batch_{batch_idx}_encoded.npy", simulation_batches)

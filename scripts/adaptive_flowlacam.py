# write the adaptive flow code idea
"""
Adaptive Flow LaCAM
=================
This script implements an adaptive flow algorithm for multi-agent pathfinding using the LaCAM framework. 
The algorithm dynamically adjusts the flow of agents based on their current positions and goals, 
optimizing their paths to minimize conflicts and travel time.

Steps:
1. Parse the map and scenario files to extract the grid layout and agent start/goal positions.
2. Use the model to make the directional congestion map predictions at t=0.
3. Initialize the LaCAM planner and run with the initial congestion paths.
4. Set the intermediate and intermediate_freq parameters to enable intermediate solutions.
5. During the planning process, periodically check the agents' positions and update the congestion map predictions.
6. Repeat steps 3-5 until all agents reach their goals or a maximum time limit is reached.
"""

#imports
import sys, os, time, glob
import subprocess
import tempfile
from matplotlib import lines
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train import OUTPUT_DIR
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
from src.utils_congestion_models import DualInputTopologyVectorFields
import ast
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import torch     
import argparse


SEED = 42

DATA_DIR = "./data/raw/"
MAP_DIR = "./data/raw/maps/"
RESULTS_DIR = "./data/raw/combined_results/"
SCEN_DIR = "./data/raw/scenarios/"
MAPS = sorted(glob.glob(MAP_DIR + "*.map"))
MAP_NAMES = [m.replace('\\', '/').split('/')[-1].split('.map')[0] for m in MAPS]

OUTPUT_DIR = "./data/raw/adaptive_flow_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# def generate_model_input(grid, scen_df_subset):
#     """
#     Generate model input (topology and vector field) for the given grid and scenario subset.
    
#     Args:
#         grid: Parsed map grid from parse_map()
#         scen_df_subset: DataFrame with agent start/goal positions (subset of scenarios)
#         width: Map width
#         height: Map height
        
#     Returns:
#         tuple: (topology_input, vector_field_input) as numpy arrays
#     """
    
#     # Extract start and goal locations from scenario dataframe
#     try:
#         ref_loc_df = scen_df_subset[['unique_id', 'start_location', 'goal_location']].copy()
        
#         # Convert string representations to tuples
#         ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str)
#         ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str)
#         ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
#         ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)

#         # Generate topology and vector field inputs
#         start_locations = ref_loc_df.start_location.tolist()
#         goal_locations = ref_loc_df.goal_location.tolist()
        
#     except Exception as e:
#         print(f"❌ Error parsing locations: {e}")
#         return None, None

#     # Work with copies - no transpose needed
#     grid_1 = grid.copy()
#     grid_2 = grid.copy()

#     # Set obstacles to 0.5
#     grid_1[grid_1 == 1] = grid_2[grid_2 == 1] = 0.5

#     # Set agent start and goal positions
#     for start_loc, goal_loc in zip(start_locations, goal_locations):
#         x_s, y_s = start_loc  # Extract x, y from (x, y) tuple
#         x_g, y_g = goal_loc   # Extract x, y from (x, y) tuple
        
#         grid_1[y_s, x_s] = 1.0  # Index grid as [y, x] to match row-major
#         grid_2[y_g, x_g] = 1.0  # Index grid as [y, x] to match row-major

#     # Generate aggregate direction fields
#     active_starts, active_goals = scen_df_subset[scen_df_subset['start_location'] != scen_df_subset['goal_location']][['start_location', 'goal_location']].values.T
#     x_field, y_field = create_aggregate_direction_fields(active_starts, active_goals, grid_1.shape)

#     # Make sure all arrays are contiguous copies - no transpose needed
#     grid_agent = np.ascontiguousarray(grid_1.copy()).astype(np.float32)
#     grid_goal = np.ascontiguousarray(grid_2.copy()).astype(np.float32)
#     x_field = np.ascontiguousarray(x_field.copy()).astype(np.float32)
#     y_field = np.ascontiguousarray(y_field.copy()).astype(np.float32)

#     # Stack along channel dimension
#     topology_input = np.stack((grid_agent, grid_goal), axis=0)  # Shape: (2, 32, 32)
#     vector_field_input = np.stack((x_field, y_field), axis=0)   # Shape: (2, 32, 32)
    
#     # Make sure everything is contiguous for safe processing
#     topology_input = np.ascontiguousarray(topology_input.copy())
#     vector_field_input = np.ascontiguousarray(vector_field_input.copy())

#     return topology_input, vector_field_input



def generate_model_input(grid, scen_df_subset, agents_reached_goal):
    """
    Generate model input (topology and vector field) for the given grid and scenario subset.
    
    Args:
        grid: Parsed map grid from parse_map()
        scen_df_subset: DataFrame with agent start/goal positions (subset of scenarios)
        width: Map width
        height: Map height
        
    Returns:
        tuple: (topology_input, vector_field_input) as numpy arrays
    """
    
    # Extract start and goal locations from scenario dataframe
    try:
            ref_loc_df = scen_df_subset[['unique_id', 'start_location', 'goal_location']].copy()
            
            # Convert string representations to tuples
            ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str)
            ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str)
            ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
            ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)

            # Generate topology and vector field inputs
            start_locations = ref_loc_df.start_location.tolist()
            goal_locations = ref_loc_df.goal_location.tolist()

            # active_starts, active_goals = scen_df_subset[~scen_df_subset['agent_id'].isin(agents_reached_goal)][['start_location', 'goal_location']].values.T

            # Work with copies - no transpose needed
            grid_1 = grid.copy()
            grid_2 = grid.copy()

            # Set obstacles to 0.5
            grid_1[grid_1 == 1] = grid_2[grid_2 == 1] = 0.5

            # Set agent start and goal positions
            for start_loc, goal_loc in zip(start_locations, goal_locations):
                x_s, y_s = start_loc  # Extract x, y from (x, y) tuple
                x_g, y_g = goal_loc   # Extract x, y from (x, y) tuple
                
                grid_1[y_s, x_s] = 1.0  # Index grid as [y, x] to match row-major
                grid_2[y_g, x_g] = 1.0  # Index grid as [y, x] to match row-major
            
            # Make sure all arrays are contiguous copies - no transpose needed
            grid_agent = np.ascontiguousarray(grid_1.copy()).astype(np.float32)
            grid_goal = np.ascontiguousarray(grid_2.copy()).astype(np.float32)
            topology_input = np.stack((grid_agent, grid_goal), axis=0)  # Shape: (2, 32, 32)
            topology_input = np.ascontiguousarray(topology_input.copy())
        
    except Exception as e:
        print(f"❌ Error parsing locations: {e}")
        return None, None        

    # Generate aggregate direction fields
    # x_field, y_field = create_aggregate_direction_fields(active_starts, active_goals)
    x_field, y_field = create_aggregate_direction_fields(start_locations, goal_locations)

    x_field = np.ascontiguousarray(x_field.copy()).astype(np.float32)
    y_field = np.ascontiguousarray(y_field.copy()).astype(np.float32)

    # Stack along channel dimension
    vector_field_input = np.stack((x_field, y_field), axis=0)   # Shape: (2, 32, 32)
    
    # Make sure everything is contiguous for safe processing
    vector_field_input = np.ascontiguousarray(vector_field_input.copy())

    return topology_input, vector_field_input


def save_intermediate_positions(historical_positions_df, solution_lines):
    """
    Save agent positions from LaCAM2 output to a DataFrame.
    
    Args:
        historical_positions_df: DataFrame to append positions
        solution_lines: Lines from LaCAM2 output containing the solution
    """

    intermediate_df = pd.DataFrame(columns=['timestep'] + [f'agent_{x}' for x in range(agents)])


    for line in solution_lines:
        timestep, path = line.split(':')
        path = list(ast.literal_eval(path))
        timestep = int(timestep.strip())
        intermediate_df.loc[len(intermediate_df)] = [timestep] + path
        
    historical_positions_df = pd.concat([historical_positions_df, intermediate_df], ignore_index=True)
    return historical_positions_df


def run_lacam_with_checkpoint(historical_positions_df, adj_matrix, map_file, scen_file, unique_id, num_agent, 
                             intermediate_freq=20, iteration=1, run_type='adaptive'):
    """
    Enhanced LaCAM2 execution with checkpoint support for adaptive algorithm.
    
    Args:
        adj_matrix: Single congestion prediction array (4, 32, 32)
        map_file: Path to .map file
        scen_file: Path to .scen file  
        unique_id: Unique identifier for this run
        num_agent: Number of agents
        intermediate_freq: Timestep interval for checkpoints (0=disabled)
        iteration: Current adaptive iteration number
        run_type: Type of run ('adaptive', 'baseline', etc.)
        
    Returns:
        dict: Results including success status, output files, and intermediate flag
    """
    # Detect if running on Windows (VS Code) or in WSL
    is_windows = os.name == 'nt' or 'WSL_DISTRO_NAME' not in os.environ

    output_file = os.path.join(OUTPUT_DIR, f"{unique_id}.txt")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create temporary adjacency matrix file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False, dir=OUTPUT_DIR) as tmp_adj_matrix_file:
        adj_matrix_file = tmp_adj_matrix_file.name
        
        # Handle tensor conversion with proper dtype management
        if torch.is_tensor(adj_matrix):
            if adj_matrix.requires_grad:
                adj_matrix_np = adj_matrix.detach().cpu().numpy().astype(np.float32)
            else:
                adj_matrix_np = adj_matrix.cpu().numpy().astype(np.float32)
        else:
            adj_matrix_np = adj_matrix.astype(np.float32)

        # Handle single datapoint - remove batch dimension if present
        if adj_matrix_np.ndim == 4 and adj_matrix_np.shape[0] == 1:
            adj_matrix_np = adj_matrix_np.squeeze(0)  # Remove batch dimension: (1,4,32,32) -> (4,32,32)
        
        # Verify final shape for single datapoint
        if adj_matrix_np.shape != (4, 32, 32):
            raise ValueError(f"Adjacency matrix has wrong shape: {adj_matrix_np.shape}, expected (4, 32, 32)")

        adj_matrix_np.tofile(adj_matrix_file)

        # Assert file was written and is non-empty
        file_size = os.path.getsize(adj_matrix_file)
        expected_size = adj_matrix_np.nbytes
        assert file_size == expected_size, f"File size mismatch: {file_size} != {expected_size}"

    try:
        if is_windows:
            # Convert Windows paths to WSL paths for the binary execution
            def win_to_wsl_path(path):
                if path.startswith('\\\\wsl.localhost\\Ubuntu'):
                    return path.replace('\\\\wsl.localhost\\Ubuntu', '').replace('\\', '/')
                elif path.startswith('./'):
                    return '/home/kanis/dev/Occupation_LaCAM2/' + path[2:].replace('\\', '/')
                else:
                    return path.replace('\\', '/')
            
            # Convert paths for WSL execution
            wsl_adj_matrix_file = win_to_wsl_path(adj_matrix_file)
            wsl_map_file = win_to_wsl_path(map_file)
            wsl_scen_file = win_to_wsl_path(scen_file)
            wsl_output_file = win_to_wsl_path(output_file)

            # Build command with checkpoint parameters
            cmd = [
                "wsl",
                "/home/kanis/dev/Occupation_LaCAM2/lacam2/build/main",
                "--map", wsl_map_file,
                "--scen", wsl_scen_file,
                "--num", str(num_agent),
                "--congestion", wsl_adj_matrix_file,
                "--output", wsl_output_file,
                "--verbose"
            ]
            
            # Add checkpoint parameters if intermediate_freq > 0
            if intermediate_freq > 0:
                cmd.extend([
                    "--intermediate",
                    "--intermediate_freq", str(intermediate_freq)
                ])
                
            # Running from Windows (VS Code) - use WSL to execute Linux binary
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        else:
            # Build command with checkpoint parameters
            cmd = [
                "lacam2/build/main",
                "--map", map_file,
                "--scen", scen_file,
                "--num", str(num_agent),
                "--congestion", adj_matrix_file,
                "--output", output_file,
                "--verbose"
            ]
            
            # Add checkpoint parameters if intermediate_freq > 0
            if intermediate_freq > 0:
                cmd.extend([
                    "--intermediate", 
                    "--intermediate_freq", str(intermediate_freq)
                ])
                
            # Running directly in WSL/Linux
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Verify output file was created and is non-empty
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file was not created: {output_file}")
        
        if os.path.getsize(output_file) == 0:
            raise ValueError(f"Output file is empty: {output_file}")
        
        # Parse the output to check if it's an intermediate solution
        is_intermediate = False
        solved = False
        makespan = 0
        
        solution_lines = []

        with open(output_file, 'r') as f:
            content = f.read()
        
        in_solution = False
        # Parse key metrics from output
        for line in content.split('\n'):
            if line.startswith('solved='):
                solved = line.split('=')[1].strip() == '1'
            elif line.startswith('makespan='):
                makespan = int(line.split('=')[1].strip())                
            elif line.startswith('solution='):
                in_solution = True
                continue
            elif in_solution and ':' in line:
                solution_lines.append(line)
            elif line.startswith('is_intermediate='):
                is_intermediate = line.split('=')[1].strip() == '1'


        current_locations = ast.literal_eval(solution_lines[-1].split(':')[-1])
        historical_positions_df = save_intermediate_positions(historical_positions_df, solution_lines)
            
        if not solution_lines:
            print(f"⚠️ No solution found in {output_file}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ LaCAM2 execution timed out for iteration {iteration} with {num_agent} agents")
        return {'success': False, 'error': 'timeout', 'iteration': iteration}
    except subprocess.CalledProcessError as e:
        print(f"❌ LaCAM2 execution failed: {e}")
        print(f"Error output: {e.stderr}")
        return {'success': False, 'error': f'execution_failed: {e}', 'iteration': iteration}
    except Exception as e:
        print(f"❌ Unexpected error in LaCAM2 execution: {e}")
        return {'success': False, 'error': f'unexpected: {e}', 'iteration': iteration}
    
    finally:
        # Ensure temporary file is cleaned up
        if os.path.exists(adj_matrix_file):
            os.remove(adj_matrix_file)

    # Return detailed results for adaptive algorithm
    result = {
        'success': True,
        'output_file': output_file,
        'is_intermediate': is_intermediate,
        'solved': solved,
        'makespan': makespan,
        'soc'
        'iteration': iteration,
        'agents': num_agent,
        'current_locations': current_locations,
        'historical_positions': historical_positions_df
    }
    return result

       

if __name__ == '__main__':
    TEST_SCEN_IDS = ["even-1", "even-25", 'even-16', "even-11", "even-7",
                    "random-16", "random-4",  "random-10", "random-7", "random-11"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = DualInputTopologyVectorFields(in_channels=2, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("./data/model/best_model_curriculum_huber_loss.pt"))
    model.eval()

    map_file = './data/raw/maps/random-32-32-10.map'
    grid, width, height = parse_map(map_file)
    scen_file = './data/raw/scenarios/random-32-32-10-random-7.scen'
    scen_id = scen_file.split(f'random-32-32-10-')[-1].split('.scen')[0]
    agents = 402

    scen_df = parse_scen_file(scen_file)
    scen_df = preprocess_scen(scen_df, scen_id)
    scen_df = scen_df.iloc[:agents] # Limit to the specified number of agents
    scen_df['agent_id'] = [f'agent_{i}' for i in range(len(scen_df))]
    check_agent_df = pd.DataFrame(columns=scen_df.columns)

    goal_locations = scen_df['goal_location'].tolist()
    start_locations = scen_df['start_location'].tolist()

    # Adaptive algorithm parameters
    max_iterations = 5
    checkpoint_interval = 20  # Checkpoint every 20 timesteps
    iteration = 1
    is_complete_solution = False
    total_timesteps = 0
    
    print(f"🚀 Starting Adaptive LaCAM2 for {agents} agents")
    print(f"📋 Checkpoint interval: {checkpoint_interval} timesteps")
    print(f"🎯 Max iterations: {max_iterations}")

    makespan, soc, sum_of_loss = 0, 0, 0

    historical_positions_df = pd.DataFrame(columns=['timestep'] + [f'agent_{x}' for x in range(agents)])
    agents_reached_goal = []


    while not is_complete_solution and iteration <= max_iterations:
        print(f"\n🔄 ITERATION {iteration}")

        is_complete_solution = len(scen_df) == len(check_agent_df) or len(scen_df) - len(check_agent_df) <= 50 or iteration+1 == max_iterations
        checkpoint_interval = 0 if is_complete_solution else checkpoint_interval
        
        print("is_complete_solution:", is_complete_solution)

        # Generate model inputs for current agent configuration
        topology_input, vector_field_input = generate_model_input(grid, scen_df, agents_reached_goal)

        if topology_input is None or vector_field_input is None:
            print(f"❌ Failed to generate model input for iteration {iteration}")
            break


        # Generate model output (congestion prediction)
        with torch.no_grad(): 
            # Convert numpy arrays to torch tensors and add batch dimension
            topology_input_tensor = torch.from_numpy(topology_input).float().unsqueeze(0).to(device)  # (1, 2, 32, 32)
            vector_field_input_tensor = torch.from_numpy(vector_field_input).float().unsqueeze(0).to(device)    # (1, 2, 32, 32)

            print(f"🧠 Model input shapes: {topology_input_tensor.shape}, {vector_field_input_tensor.shape}")
            direction_congestion_maps = model(topology_input_tensor, vector_field_input_tensor)

            print(f"🧠 Congestion prediction: max={direction_congestion_maps.max():.3f}, mean={direction_congestion_maps.mean():.3f}")

        # Run LaCAM2 with checkpoint
        unique_id_iter = f"{scen_id}_agents_{agents}_iter{iteration}"
        
        result = run_lacam_with_checkpoint(
            historical_positions_df, direction_congestion_maps, map_file, scen_file, unique_id_iter, agents,
            intermediate_freq=checkpoint_interval, iteration=iteration
        )

        historical_positions_df = result['historical_positions']
        
        for idx in range(len(scen_df)):
            agent_col = f'agent_{idx}'
            if agent_col not in agents_reached_goal and agent_col in historical_positions_df.columns:
                # Get the goal location for this specific agent
                agent_goal = scen_df.iloc[idx]['goal_location']
                scen_df.at[idx, 'start_location'] = historical_positions_df[agent_col].iloc[-1]
                # Check if this agent has reached their goal at any point
                if agent_goal in historical_positions_df[agent_col].iloc[checkpoint_interval * (iteration - 1): checkpoint_interval * iteration].values.tolist():
                    # Add this agent's row to check_agent_df
                    agent_row = scen_df.iloc[idx:idx+1].copy()  # Get single row as DataFrame
                    check_agent_df = pd.concat([check_agent_df, agent_row], ignore_index=True)
                    agents_reached_goal.append(agent_col)

        print(len(check_agent_df), 'of', len(scen_df), 'agents at goal')
        

        check_agent_df = check_agent_df.drop_duplicates()


        if not result['success']:
            print(f"❌ LaCAM2 failed at iteration {iteration}: {result.get('error', 'unknown')}")
            break
        
        print(f"✅ LaCAM2 completed: makespan={result['makespan']}, intermediate={result['is_intermediate']}")

        # Check if complete solution or checkpoint
        if result['solved'] and not result['is_intermediate'] or is_complete_solution:
            print("🎉 Complete solution found!")
            is_complete_solution = True
            total_timesteps += result['makespan']
            
        elif result['is_intermediate']:
            print(f"   🔄 Checkpoint reached at timestep {result['makespan']}")
            total_timesteps += result['makespan']
                    
        else:
            print("⚠️ LaCAM2 completed but no solution found - stopping")
            break
    
        iteration += 1
        
    
    # Final summary
    print(f"\n📊 ADAPTIVE ALGORITHM COMPLETED")
    print(f"   Total iterations: {iteration - 1}")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Success: {is_complete_solution}")
    print(f"   Makespan: {makespan}, SOC: {soc}, Sum of Loss: {sum_of_loss}")
    
    if is_complete_solution:
        print(f"   🎯 Solution found in {iteration - 1} adaptive iterations!")
    else:
        print(f"   ❌ No complete solution found within {max_iterations} iterations")

    historical_positions_df.to_csv(os.path.join(OUTPUT_DIR, f"{unique_id_iter}_historical_positions.csv"), index=False)





    
    
    # for map_name in MAP_NAMES:
    #     map_file = f'./data/raw/maps/{map_name}.map' #This scene file can be given as input from bash
    #     grid, width, height = parse_map(map_file)

    #     for scen_id in TEST_SCEN_IDS:
    #         scen_file = f'./data/raw/scenarios/{map_name}-{scen_id}.scen' #This scene file can be given as input from bash
    #         print(f"Running test scenario: {scen_id} on map: {map_name}")

    #         scen_df = parse_scen_file(scen_file)
    #         scen_df = preprocess_scen(scen_df, scen_id)

    #         agents = len(scen_df)

    #         for n_agents in range(100, agents+1, 10):
    #             scen_df_subset = scen_df.iloc[:n_agents] # Limit to the specified number of agents
    #             unique_id = f"{map_name}_{scen_id}_agent_{n_agents}"
                
    #             # Generate inputs for the model
    #             topology_input, vector_field_input = generate_model_input(
    #                 grid, scen_df_subset
    #             )
    #             # generate inputs for the model



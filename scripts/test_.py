import warnings, sys
import ast, os
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
from src.utils_congestion_models import DualInputTopologyVectorFields
from scripts.train import run_lacam_in_wsl
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

scen_repo= "/home/kanis/dev/Occupation_LaCAM2//data/raw/room-32-32-4/room-32-32-4-"
# scen_repo= "/home/kanis/dev/Occupation_LaCAM2//data/raw/maze-32-32-4/maze-32-32-4-"
TEMP_DIR = "/home/kanis/dev/Occupation_LaCAM2/data/temp/"
os.makedirs(TEMP_DIR, exist_ok=True)
OUTPUT_DIR = "/home/kanis/dev/Occupation_LaCAM2/data/test_random_4/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TEST_SCEN_IDS = ["even-1", "random-16", "random-4", "even-25", "random-10"]
TEST_SCEN_IDS = ['random-4', 'even-19', 'random-14', 'even-12']
NUM_WORKERS = 6  # Adjust based on your CPU cores





def prepare_model_input(start_locations, goal_locations, grid):
    """Prepares input for the neural network model"""
    # Work with copies - no transpose needed
    grid_1 = grid.copy()
    grid_2 = grid.copy()

    grid_1[grid_1 == 1] = grid_2[grid_2 == 1] = 0.5

    for start_loc, goal_loc in zip(start_locations, goal_locations):
        x_s, y_s = start_loc  # Extract x, y from (x, y) tuple
        x_g, y_g = goal_loc   # Extract x, y from (x, y) tuple
        
        grid_1[y_s, x_s] = 1.0  # Index grid as [y, x] to match row-major
        grid_2[y_g, x_g] = 1.0  # Index grid as [y, x] to match row-major

    x_field, y_field = create_aggregate_direction_fields(start_locations, goal_locations, grid_1.shape)

    # Make sure all arrays are contiguous copies - no transpose needed
    grid_agent = np.ascontiguousarray(grid_1.copy()).astype(np.float32)
    grid_goal = np.ascontiguousarray(grid_2.copy()).astype(np.float32)
    x_field = np.ascontiguousarray(x_field.copy()).astype(np.float32)
    y_field = np.ascontiguousarray(y_field.copy()).astype(np.float32)

    # Stack along channel dimension
    topology_input = np.stack((grid_agent, grid_goal), axis=0)
    vector_field_input = np.stack((x_field, y_field), axis=0)

    # Make sure everything is contiguous for safe serialization
    topology_input = np.ascontiguousarray(topology_input.copy())
    vector_field_input = np.ascontiguousarray(vector_field_input.copy())
    
    return topology_input, vector_field_input

def run_lacam_worker(args):
    """Worker function that can be safely passed to multiprocessing"""
    model_output, map_file, scen_file, scen_id, agents, output_dir, temp_dir = args
    try:
        result = run_lacam_in_wsl(model_output, map_file, scen_file, scen_id, agents, 
                               run_type='model', output_dir=output_dir, temp_dir=temp_dir)
        return f"{scen_id}_{agents}", result
    except Exception as e:
        return f"{scen_id}_{agents}", f"Error: {str(e)}"



if __name__ == "__main__":
    scen_files = [scen_repo + f + '.scen' for f in TEST_SCEN_IDS]

    # random_32_32_df = pd.read_csv("/home/kanis/dev/Occupation_LaCAM2/data/raw/random-32-32-20/random-32-32-20.csv")
    random_32_32_df = pd.read_csv("/home/kanis/dev/Occupation_LaCAM2/data/raw/room-32-32-4/room-32-32-4.csv")
    # random_32_32_df = pd.read_csv("/home/kanis/dev/Occupation_LaCAM2/data/raw/maze-32-32-4/maze-32-32-4.csv")
    random_32_32_df['unique_id'] = (random_32_32_df['scen_type'] + '_' + 
                                    random_32_32_df['type_id'].astype(str) + '_agent_' + 
                                    random_32_32_df['agents'].astype(str))
    random_32_32_df['scen_id'] = random_32_32_df['scen_type'] + '-' + random_32_32_df['type_id'].astype(str)

    # map_file = "/home/kanis/dev/Occupation_LaCAM2//data/raw/random-32-32-20/random-32-32-20.map"
    map_file = "/home/kanis/dev/Occupation_LaCAM2//data/raw/room-32-32-4/room-32-32-4.map"
    # map_file = "/home/kanis/dev/Occupation_LaCAM2//data/raw/maze-32-32-4/maze-32-32-4.map"
    grid, width, height = parse_map(map_file)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DualInputTopologyVectorFields(in_channels=2, hidden_dim=64).to(device)

    # Check if the saved model file exists before loading
    model_path = "/home/kanis/dev/Occupation_LaCAM2/data/model/best_model_curriculum.pt"
    if os.path.exists(model_path):
        print(f"Loading pre-trained weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Pre-trained weights loaded successfully!")
    else:
        print("No pre-trained weights found. Starting training from scratch.")

    # Process one scenario at a time to avoid memory explosion
    all_results = []
    
    for idx, scen_file in enumerate(scen_files):
        scen_id = TEST_SCEN_IDS[idx]
        scen_df = parse_scen_file(scen_file)
        scen_id_processed = scen_id.replace('-', '_')
        scen_df = preprocess_scen(scen_df, scen_id)
        num_agents = len(scen_df)
        
        print(f"\nProcessing scenario {scen_id} with {num_agents-10+1} agent configurations")
        lacam_jobs = []
        
        # Prepare model outputs for this scenario only
        for agents in tqdm(range(10, num_agents + 1), desc=f"Running model inference"):
            # Prepare data for this scenario/agent pair
            unique_id = scen_df[:agents].iloc[-1]['unique_id']

            ref_loc_df = scen_df[['unique_id', 'start_location', 'goal_location']].copy()[:agents]
            ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str)
            ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str)
            ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
            ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)

            # Generate topology and vector field inputs
            start_locations = ref_loc_df.start_location.tolist()
            goal_locations = ref_loc_df.goal_location.tolist()
            
            # Run model inference in main process
            topology_input, vector_field_input = prepare_model_input(start_locations, goal_locations, grid)
            model_output = model(torch.tensor(topology_input).unsqueeze(0).to(device), 
                              torch.tensor(vector_field_input).unsqueeze(0).to(device))
            model_output = model_output.squeeze(0).detach().cpu().numpy()
            
            # Store the job for parallel execution (for this scenario only)
            lacam_jobs.append((model_output, map_file, scen_file, scen_id_processed, 
                            agents, OUTPUT_DIR, TEMP_DIR))
        
        # Run LaCAM executions in parallel for just this scenario
        print(f"Running {len(lacam_jobs)} LaCAM jobs in parallel for {scen_id}...")
        scenario_results = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(run_lacam_worker, job) for job in lacam_jobs]
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"Running LaCAM for {scen_id}"):
                try:
                    result = future.result()
                    scenario_results.append(result)
                except Exception as e:
                    print(f"Error in LaCAM execution: {e}")
        
        # Clear memory for next scenario
        del lacam_jobs
        del scenario_results
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    


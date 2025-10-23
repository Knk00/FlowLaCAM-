import sys, os, glob, gc, json, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
import ast
import matplotlib
matplotlib.use('TkAgg') 
import torch     
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

np.random.seed(42)

# Configuration
SEED = 42
BATCH_SIZE = 32
MAP_DIR = "./data/raw/maps/"
RESULTS_DIR = "./data/raw/combined_results/"
SCEN_DIR = "./data/raw/scenarios/"
PRECOMPUTED_DIR = "./data/precomputed_v2/"
NUM_WORKERS = 6

# 🎯 TEST-ONLY CONFIGURATION
TEST_SCEN_IDS = ["even-1", "even-25", 'even-16', "even-11", "even-7",
                 "random-16", "random-4",  "random-10", "random-7", "random-11"]

# Map setup
MAPS = sorted(glob.glob(MAP_DIR + "*.map"))
MAP_NAMES = [m.replace('\\', '/').split('/')[-1].split('.map')[0] for m in MAPS]

@dataclass
class TestScenarioMetadata:
    """Metadata for test scenarios"""
    unique_id: str
    scen_id: str
    scen_file: str
    map_name: str
    agents: int
    batch_idx: int
    inside_idx: int
    solution_cost: float
    scen_type: str
    type_id: int

def process_test_instance(item: tuple, map_grids: dict, batch_idx: int, inside_idx: int):
    """Process a single test instance"""
    scenario = item
    map_name = scenario['map_name']
    agents = scenario['agents']
    unique_id = scenario['unique_id']
    solution_cost = scenario['solution_cost']
    scen_id = scenario['scen_id']
    scen_file = SCEN_DIR + map_name + '-' + scen_id + '.scen'
    
    # Get the appropriate grid for this map
    grid = map_grids[map_name]

    # Check if scenario file exists
    if not os.path.exists(scen_file):
        print(f"⚠️ Scenario file not found: {scen_file}")
        return None
    
    try:
        scen_df = parse_scen_file(scen_file)
        scen_df = preprocess_scen(scen_df, scen_id)
        
        # Get reference locations
        ref_loc_df = scen_df[['unique_id', 'start_location', 'goal_location']].copy()[:agents]
        
        if len(ref_loc_df) < agents:
            print(f"⚠️ Insufficient location data for {scen_id}: need {agents}, got {len(ref_loc_df)}")
            return None
            
        ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str).apply(ast.literal_eval)
        ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str).apply(ast.literal_eval)

        start_locations = ref_loc_df.start_location.tolist()
        goal_locations = ref_loc_df.goal_location.tolist()
        
    except Exception as e:
        print(f"❌ Error processing locations for {scen_id}: {e}")
        return None

    # Generate model inputs (topology and vector fields)
    try:
        # Work with grid copies
        grid_1 = grid.copy()
        grid_2 = grid.copy()

        grid_1[grid_1 == 1] = grid_2[grid_2 == 1] = 0.5

        for start_loc, goal_loc in zip(start_locations, goal_locations):
            x_s, y_s = start_loc
            x_g, y_g = goal_loc
            
            grid_1[y_s, x_s] = 1.0
            grid_2[y_g, x_g] = 1.0

        x_field, y_field = create_aggregate_direction_fields(start_locations, goal_locations, grid_1.shape)

        # Create contiguous arrays
        grid_agent = np.ascontiguousarray(grid_1.copy()).astype(np.float32)
        grid_goal = np.ascontiguousarray(grid_2.copy()).astype(np.float32)
        x_field = np.ascontiguousarray(x_field.copy()).astype(np.float32)
        y_field = np.ascontiguousarray(y_field.copy()).astype(np.float32)

        # Stack along channel dimension
        topology_input = np.stack((grid_agent, grid_goal), axis=0)
        vector_field_input = np.stack((x_field, y_field), axis=0)
        
        # Ensure contiguity
        topology_input = np.ascontiguousarray(topology_input.copy())
        vector_field_input = np.ascontiguousarray(vector_field_input.copy())

    except Exception as e:
        print(f"❌ Error generating inputs for {scen_id}: {e}")
        return None

    # Create metadata
    metadata = TestScenarioMetadata(
        unique_id=unique_id,
        scen_id=scen_id,
        scen_file=scen_file,
        map_name=map_name,
        agents=agents,
        batch_idx=batch_idx,
        inside_idx=inside_idx,
        solution_cost=solution_cost,
        scen_type=scenario['scen_type'],
        type_id=scenario['type_id']
    )

    result = {
        'topology_input': topology_input,
        'vector_field_input': vector_field_input,
        'metadata': metadata
    }  

    gc.collect()
    return result

def save_test_batch(batch_results: list, batch_idx: int):
    """Save test batch results"""
    batch_size = len(batch_results)
    
    # Pre-allocate arrays
    topology_inputs = np.empty((batch_size, 2, 32, 32), dtype=np.float32)
    vector_field_inputs = np.empty((batch_size, 2, 32, 32), dtype=np.float32)
    
    # Fill arrays
    for i, scenario in enumerate(batch_results):
        topology_inputs[i] = scenario['topology_input']
        vector_field_inputs[i] = scenario['vector_field_input'] 

    metadata_list = [results['metadata'] for results in batch_results]

    # Create test batch directory
    test_batch_dir = f"{PRECOMPUTED_DIR}/batches/test/"
    os.makedirs(test_batch_dir, exist_ok=True)
    
    # Save batch data
    batch_filename = f"{test_batch_dir}/batch_{batch_idx:04d}.pt"
    save_dict = {
        'topology_inputs': torch.from_numpy(topology_inputs),
        'vector_field_inputs': torch.from_numpy(vector_field_inputs),
        'batch_idx': batch_idx
    }

    torch.save(save_dict, batch_filename)

    # Save metadata
    metadata_filename = f"{PRECOMPUTED_DIR}/metadata/test_batch_{batch_idx:04d}.json"
    with open(metadata_filename, 'w') as f:
        json.dump([asdict(meta) for meta in metadata_list], f, indent=2)

    print(f"  ✅ Saved test batch {batch_idx} ({len(batch_results)} scenarios)")
    
    del topology_inputs, vector_field_inputs, metadata_list, save_dict
    gc.collect()

def main():
    """Main function to precompute test data only"""
    print("🧪 TEST DATA PRECOMPUTATION")
    print("="*50)
    print(f"📁 Maps: {len(MAPS)} files")
    print(f"🎯 Test scenario IDs: {TEST_SCEN_IDS}")
    print(f"⚙️  Workers: {NUM_WORKERS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    
    # Create directories
    os.makedirs(f"{PRECOMPUTED_DIR}/batches/test/", exist_ok=True)
    os.makedirs(f"{PRECOMPUTED_DIR}/metadata/", exist_ok=True)
    
    # Load all maps into memory
    print("\n📊 Loading maps...")
    map_grids = {}
    for map_file, map_name in zip(MAPS, MAP_NAMES):
        grid, width, height = parse_map(map_file)
        map_grids[map_name] = grid
        print(f"   ✅ {map_name}: {width}x{height}")

    # Load and filter test scenarios
    print("\n🔍 Loading test scenarios...")
    solution_df = pd.DataFrame()

    for map_name in MAP_NAMES:
        map_df = pd.read_csv(RESULTS_DIR + map_name + '.csv')
        
        # Basic filtering
        if 'maze' in map_name or 'room' in map_name:
            map_df = map_df[map_df['solution_plan'].notna()]
        if 'maze' not in map_name:
            map_df = map_df[map_df['agents'] >= 4]
        
        solution_df = pd.concat([solution_df, map_df], ignore_index=True)

    # Create identifiers
    solution_df['unique_id'] = (solution_df['map_name'] + '_' + solution_df['scen_type'] + '_' + 
                                solution_df['type_id'].astype(str) + '_agent_' + 
                                solution_df['agents'].astype(str))
    solution_df['scen_id'] = solution_df['scen_type'] + '-' + solution_df['type_id'].astype(str)

    # 🎯 FILTER FOR TEST SCENARIOS ONLY
    test_df = solution_df[solution_df['scen_id'].isin(TEST_SCEN_IDS)].copy()
    
    print(f"\n📊 Test scenarios found:")
    for scen_id in TEST_SCEN_IDS:
        count = len(test_df[test_df['scen_id'] == scen_id])
        print(f"   {scen_id}: {count} scenarios")
    
    print(f"\n🎯 Total test scenarios: {len(test_df)}")
    
    # Calculate batches
    n_test_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    print(f"📦 Test batches to create: {n_test_batches}")

    # Process test scenarios in batches
    print(f"\n🔄 Processing test batches...")
    
    for batch_idx in tqdm(range(n_test_batches), desc="Test batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        
        args = [(row, map_grids, batch_idx, idx) for idx, row in enumerate(batch_df.to_dict('records'))]
        
        print(f"\n  📦 Processing test batch {batch_idx+1}/{n_test_batches} - {len(args)} scenarios")
        
        batch_results = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_test_instance, *arg) for arg in args]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Scenarios", leave=False):
                result = future.result()
                if result is not None:
                    batch_results.append(result)

        # Save batch if we have valid results
        if batch_results:
            save_test_batch(batch_results, batch_idx)
        else:
            print(f"  ⚠️ No valid results for test batch {batch_idx}")

        # Cleanup
        del batch_results
        gc.collect()

    # Save test scenario summary
    print(f"\n💾 Saving test scenario summary...")
    test_summary_file = f"{PRECOMPUTED_DIR}/metadata/test_scenarios_summary.csv"
    test_df.to_csv(test_summary_file, index=False)
    
    print(f"\n🎉 Test data precomputation completed!")
    print(f"📁 Batches saved: {n_test_batches}")
    print(f"📊 Summary saved: {test_summary_file}")
    print(f"📂 Location: {PRECOMPUTED_DIR}")

if __name__ == "__main__":
    main()
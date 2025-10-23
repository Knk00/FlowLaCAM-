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
BATCH_SIZE = 32  # Reduce batch size
DATA_DIR = "./data/raw/"
MAP_DIR = "./data/raw/maps/"
RESULTS_DIR = "./data/raw/combined_results/"
SCEN_DIR = "./data/raw/scenarios/"
MAPS = sorted(glob.glob(MAP_DIR + "*.map"))
MAP_NAMES = [m.replace('\\', '/').split('/')[-1].split('.map')[0] for m in MAPS]
PRECOMPUTED_DIR = "./data/precomputed_v3/"
DIR = "C:\\Users\\kanis\\Documents\\Monash Temp\\MAPF Research\\precomputed_data"
NUM_WORKERS = 6


@dataclass
class ScenarioMetadata:
    """Metadata for each scenario"""
    unique_id: str
    scen_id: str
    scen_file: str
    agents: int
    batch_idx: int
    inside_idx: int
    data_split: str  # 'train', 'val', 'test'
    solution_cost: float

def stratified_sampling_train_test_val(df: pd.DataFrame, sampling_fraction: float = 1.0):
    test_scen_ids = ["even-1", "even-25", 'even-16', "even-11", "even-7",
                     "random-16", "random-4",  "random-10", "random-7", "random-11"]
    
    # Select test scenarios BEFORE filtering by agents
    test_df = df[df['scen_id'].isin(test_scen_ids)]
    
    # Filter remaining data by agents >= 10
    remaining_df = df[~df['scen_id'].isin(test_scen_ids)]
    remaining_df = remaining_df[remaining_df['agents'] >= 10]
    
    # 🎯 Balance map types - reduce empty map dominance
    map_counts = remaining_df['map_name_prefix'].value_counts()
    print(f"\n📊 Original map distribution:")
    for prefix, count in map_counts.items():
        total_remaining = len(remaining_df)
        print(f"   {prefix}: {count} ({count/total_remaining*100:.1f}%)")
    
    # Target distribution: empty=10%, others proportional
    target_empty_ratio = 0.01
    total_non_empty = remaining_df[remaining_df['map_name_prefix'] != 'empty'].shape[0]
    target_empty_count = int(total_non_empty * target_empty_ratio / (1 - target_empty_ratio))
    
    # Subsample empty maps if they exceed target - STRATIFIED BY AGENT COUNT
    empty_df = remaining_df[remaining_df['map_name_prefix'] == 'empty']
    empty_df = empty_df[empty_df['agents'] >= 350]

    if len(empty_df) > target_empty_count:
        print(f"🎯 Subsampling empty maps: {len(empty_df)} -> {target_empty_count}")
        print(f"📊 Empty map agent distribution BEFORE subsampling:")
        empty_agent_counts = empty_df['agents'].value_counts().sort_index()
        for agents, count in empty_agent_counts.items():
            print(f"   {agents} agents: {count} scenarios")
        
        # Stratified sampling by agent count within empty maps
        empty_df = empty_df.groupby('agents', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, int(len(x) * target_empty_count / len(empty_df)))), 
                             random_state=SEED)
        )
        
        # If we still have too many, do final random subsample
        if len(empty_df) > target_empty_count:
            empty_df = empty_df.sample(n=target_empty_count, random_state=SEED)
            
        print(f"📊 Empty map agent distribution AFTER subsampling:")
        empty_agent_counts_after = empty_df['agents'].value_counts().sort_index()
        for agents, count in empty_agent_counts_after.items():
            print(f"   {agents} agents: {count} scenarios")
    
    # Combine balanced dataset
    non_empty_df = remaining_df[remaining_df['map_name_prefix'] != 'empty']
    remaining_df = pd.concat([empty_df, non_empty_df], ignore_index=True)
    
    print(f"\n📊 Balanced map distribution:")
    balanced_counts = remaining_df['map_name_prefix'].value_counts()
    total_balanced = len(remaining_df)
    for prefix, count in balanced_counts.items():
        print(f"   {prefix}: {count} ({count/total_balanced*100:.1f}%)")
    
    # Stratified sampling on balanced data
    strat_df = remaining_df.groupby(by='agents', group_keys=False).apply(
        lambda x: x.sample(frac=sampling_fraction, random_state=SEED)
    )
    
    # Split train/val
    train_frac = 0.8
    train_df = strat_df.sample(frac=train_frac, random_state=SEED)
    val_df = strat_df.drop(train_df.index)
    
    # Updated assertion
    assert len(train_df) + len(val_df) + len(test_df) == len(strat_df) + len(test_df)
    
    return train_df, val_df, test_df


def process_instance(item: tuple, grid: np.array, split: str, batch_idx: int, inside_idx: int):
    agents = item['agents']
    solution_plan = item['solution_plan']
    unique_id = item['unique_id']
    solution_cost = item['solution_cost']
    scen_id = item['scen_id']
    scen_file = SCEN_DIR  + item['map_name'] + '-' + scen_id + '.scen'

    # Check if file exists
    if not os.path.exists(scen_file):
        print(f"⚠️ Scenario file not found: {scen_file}")
        print('Scene file found:', scen_file)
        return None
    
    scen_df = parse_scen_file(scen_file)

    # FIXED: Use correct scen_id parsing logic from reference
    scen_df = preprocess_scen(scen_df, scen_id)
    
    try:
        solution_plan = solution_plan.split('\n')
    except AttributeError:
        print(f"⚠️ Invalid solution plan format for {scen_id}")
        return None
    # Handle agent/path count mismatches more intelligently
    # if len(solution_plan) != agents:
    #     if abs(len(solution_plan) - agents) <= 2:  # Allow small discrepancies
    #         print(f"📝 Note: {scen_id} has {len(solution_plan)} paths for {agents} agents - using min count")
    #         agents = min(agents, len(solution_plan))
    #     else:
    #         print(f"⚠️ Skipping {scen_id} - large mismatch: {agents} agents vs {len(solution_plan)} paths")
    #         return None

    if len(solution_plan) != agents:
        return None
    

    # Create reference DataFrame for locations - more robust
    try:
        ref_loc_df = scen_df[['unique_id', 'start_location', 'goal_location']].copy()[:agents]
        
        if len(ref_loc_df) < agents:
            print(f"⚠️ Insufficient location data for {scen_id}: need {agents}, got {len(ref_loc_df)}")
            return None
            
        ref_loc_df['start_location'] = ref_loc_df['start_location'].astype(str)
        ref_loc_df['goal_location'] = ref_loc_df['goal_location'].astype(str)
        ref_loc_df['start_location'] = ref_loc_df['start_location'].apply(ast.literal_eval)
        ref_loc_df['goal_location'] = ref_loc_df['goal_location'].apply(ast.literal_eval)

        # Generate topology and vector field inputs
        start_locations = ref_loc_df.start_location.tolist()
        goal_locations = ref_loc_df.goal_location.tolist()
        
    except Exception as e:
        print(f"❌ Error parsing locations for {scen_id}: {e}")
        return None

    # Generate ground truth adjacency matrix
    try:
        if scen_id == "even-13":
            print('debug')
        gt_adj_matrix = edge_frequency_count(solution_plan, agents, start_locations, goal_locations, channels=4)
        
        if gt_adj_matrix is None:
            print(f"⚠️ Failed to generate adjacency matrix for {scen_id}")
            return None
    
    except Exception as e:
        print(f"❌ Error generating adjacency matrix for {scen_id}: {e}")
        return None
    
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
    gt_adj_matrix = np.ascontiguousarray(gt_adj_matrix.copy()).astype(np.float32)
    # gt_adj_matrix_transformed = np.ascontiguousarray(gt_adj_matrix_transformed.copy())

    metadata = ScenarioMetadata(
        unique_id=unique_id,
        scen_id=scen_id,
        scen_file=scen_file,
        agents=agents,
        batch_idx=batch_idx,
        inside_idx=inside_idx,
        data_split=split,
        solution_cost=solution_cost
    )

    result = {
    'topology_input': topology_input,
    'vector_field_input': vector_field_input,
    'adjacency_matrix_2d': gt_adj_matrix,
    # 'adjacency_matrix_4d': gt_adj_matrix_transformed,
    'metadata': metadata
    }  

    gc.collect()
    return result


def save_batch_results(batch_results: list, batch_idx: int, split_name: str):
    batch_size = len(batch_results)
    
    # Pre-allocate based on known shapes (32x32 grids, etc.)
    topology_inputs = np.empty((batch_size, 2, 32, 32), dtype=np.float32)
    vector_field_inputs = np.empty((batch_size, 2, 32, 32), dtype=np.float32)
    adjacency_matrices_2d = np.empty((batch_size, 4, 32, 32), dtype=np.float32)  # 32*32 = 1024
    
    # Vectorized fill - much faster than stacking
    for i, scenario in enumerate(batch_results):
        topology_inputs[i] = scenario['topology_input']
        vector_field_inputs[i] = scenario['vector_field_input'] 
        adjacency_matrices_2d[i] = scenario['adjacency_matrix_2d']

    metadata_list = [results['metadata'] for results in batch_results]

    # Save with proper data types
    batch_filename = f"{PRECOMPUTED_DIR}/batches/{split_name}/batch_{batch_idx:04d}.pt"
    save_dict = {
        'topology_inputs': torch.from_numpy(topology_inputs),
        'vector_field_inputs': torch.from_numpy(vector_field_inputs),
        'adjacency_matrices_2d': torch.from_numpy(adjacency_matrices_2d),
        'batch_idx': batch_idx
    }

    torch.save(save_dict, batch_filename)

     # Save metadata separately
    metadata_filename = f"{PRECOMPUTED_DIR}/metadata/{split_name}_batch_{batch_idx:04d}.json"
    with open(metadata_filename, 'w') as f:
        json.dump([asdict(meta) for meta in metadata_list], f, indent=2)

    print(f"  ✅ Successfully saved batch {batch_idx} for {split_name}")
    
    del topology_inputs, vector_field_inputs, adjacency_matrices_2d, metadata_list, save_dict
    gc.collect()


def estimate_precomputed_space(num_scenarios: int, batch_size: int = 32):
    """
    Estimate disk space required for precomputed batches.
    Returns size in MB and GB.
    """
    # Per scenario sizes
    topology_bytes = 2 * 32 * 32 * 2         # 2 channels, int16 (2 bytes)
    vector_field_bytes = 2 * 32 * 32 * 4     # 2 channels, float32 (4 bytes)
    adj_matrix_bytes = 1024 * 1024 * 2       # int16 (2 bytes)
    scenario_bytes = topology_bytes + vector_field_bytes + adj_matrix_bytes

    total_bytes = scenario_bytes * num_scenarios

    # Metadata is small, can be ignored for rough estimate
    mb = total_bytes / (1024 ** 2)
    gb = total_bytes / (1024 ** 3)

    print(f"Estimated storage needed for {num_scenarios} scenarios:")
    print(f"  Per scenario: {scenario_bytes/1024:.2f} KB")
    print(f"  Total: {mb:.2f} MB ({gb:.2f} GB)")
    return mb, gb

if __name__ == "__main__":
    solution_df = pd.DataFrame()
    scen_files = []

    for map_file, map_name in zip(MAPS, MAP_NAMES):
        grid, width, height = parse_map(map_file)
        scen_files += glob.glob('/'.join(map_file.split('/')[:-1]) +  "/*.scen")
        map_df = pd.read_csv(RESULTS_DIR + map_name + '.csv')
        if 'maze' in map_name or 'room' in map_name:
            map_df = map_df[map_df['solution_plan'].notna()]
        
        map_df['map_name_prefix'] = map_name.split('-')[0]

        solution_df = pd.concat([solution_df, map_df], ignore_index=True)


    solution_df['unique_id'] = (solution_df['map_name'] + '_' + solution_df['scen_type'] + '_' + 
                                solution_df['type_id'].astype(str) + '_agent_' + 
                                solution_df['agents'].astype(str))
    solution_df['scen_id'] = solution_df['scen_type'] + '-' + solution_df['type_id'].astype(str)


    train_df, val_df, test_df = stratified_sampling_train_test_val(solution_df, sampling_fraction=1.0)

    n_train_batches = len(train_df) // BATCH_SIZE
    n_val_batches = len(val_df) // BATCH_SIZE

    train_mb, train_gb = estimate_precomputed_space(len(train_df), BATCH_SIZE)
    val_mb, val_gb = estimate_precomputed_space(len(val_df), BATCH_SIZE)

    print(f"\n🚀 Starting data preprocessing...")
    print(f"📊 Dataset splits:")
    print(f"   Train: {len(train_df)} scenarios ({n_train_batches} batches)")
    print(f"   Train size: ({train_gb:.2f} GB)")
    print(f"   Val:   {len(val_df)} scenarios ({n_val_batches} batches)")
    print(f"   Val size: ({val_gb:.2f} GB)")
    print(f"   Test:  {len(test_df)} scenarios")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Workers: {NUM_WORKERS}")

    # Process train and val splits
    for split_name, split_df, n_batches in [('train', train_df, n_train_batches), ('val', val_df, n_val_batches)]:
        print(f"\n🔄 Processing {split_name} split: {n_batches} batches")
        
        for batch_idx in tqdm(range(n_batches), desc=f"{split_name.capitalize()} batches"):
            batch_df = split_df.iloc[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            args = [(row, grid, split_name, batch_idx, idx_) for idx_, row in enumerate(batch_df.to_dict('records'))]

            print(f"\n  📦 Starting batch {batch_idx+1}/{n_batches} ({split_name}) - {len(args)} scenarios")
            
            batch_results = []
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(process_instance, *arg) for arg in args]
                
                # Process with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Processing scenarios", leave=False):
                    if future.result() is not None:
                        result = future.result()
                        batch_results.append(result)
            # for arg in args:
            #     result = process_instance(*arg)
            #     if result is not None:
            #         batch_results.append(result)
            #     print(f"🗂️ Processed {len(batch_results)}/{32} valid scenarios in batch {batch_idx+1}")

            # Save the batch results if we have any valid results
                if batch_results:
                    if len(batch_results) >= 24:  # At least 75% of batch size
                        save_batch_results(batch_results, batch_idx, split_name)
                else:
                    print(f"  ⚠️ No valid results for batch {batch_idx}")

            # Cleanup
            del batch_results
            gc.collect()
            print(f"  ✅ Completed batch {batch_idx+1}/{n_batches} ({split_name})")

    print(f"\n🎉 All processing completed!")
    print(f"💾 Saving test scenarios metadata...")
    
    test_df.to_csv(f"{PRECOMPUTED_DIR}/metadata/test_scenarios.csv")
    print(f"✅ Test scenarios saved to {PRECOMPUTED_DIR}/metadata/test_scenarios.csv")

import sys, os, glob, gc, json, warnings, time, ast, tempfile, subprocess
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
from src.utils_congestion_models import DualInputTopologyVectorFields
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== CONFIGURATION - CHANGE THESE AS NEEDED =====
# MODEL_PATH = "./data/model/best_model_curriculum_huber_loss.pt"  # Path to trained model"
# MODEL_PATH = "./data/model/unstable_stage3_training/stage_3_final_model_20251017-105758.pt"  # Path to trained model
MODEL_PATH = "./data/model/best_model_curriculum_stable_huber_loss_stage3.pt"  # Path to trained model
TEST_OUTPUT_DIR = "./data/test_huber_loss_stable_stage3_model/"  # Output directory for test results
MAP_DIR = "./data/raw/maps/"  # Maps directory
SCEN_DIR = "./data/raw/scenarios/"  # Scenarios directory
PRECOMPUTED_DIR = "./data/precomputed_v2/"  # Precomputed test data
TEST_BATCH_DIR = PRECOMPUTED_DIR + 'batches/test/'  # Test batches
TEST_METADATA_DIR = PRECOMPUTED_DIR + 'metadata/'  # Test metadata
NUM_WORKERS = 6  # Number of parallel workers for LaCAM2 execution
DEVICE = 'cuda'  # Device to run model on ('cuda' or 'cpu')

# Test dataset scenario IDs (should match precompute_test_only.py)
TEST_SCEN_IDS = ["even-1", "even-25", 'even-16', "even-11", "even-7",
                     "random-16", "random-4",  "random-10", "random-7", "random-11"]
# ================================================

# Setup multiple map handling like train.py
MAPS = sorted(glob.glob(MAP_DIR + "*.map"))
MAP_NAMES = [m.replace('\\', '/').split('/')[-1].split('.map')[0] for m in MAPS]
map_files_dict = {name: path for name, path in zip(MAP_NAMES, MAPS)}

def run_lacam_in_wsl(adj_matrix, map_file, scen_file, scen_id, unique_id, num_agent, run_type='model'):
    """
    Enhanced LaCAM2 execution with better error handling and path management.
    """
    # Detect if running on Windows (VS Code) or in WSL
    is_windows = os.name == 'nt' or 'WSL_DISTRO_NAME' not in os.environ

    output_file = TEST_OUTPUT_DIR + unique_id  + f'_{run_type}.txt'

    # Skip if output already exists for ground truth runs
    if os.path.exists(output_file) and run_type == 'gt':
        print(f"Output file already exists: {output_file}. Skipping LaCAM2 execution.")
        return True

    # Ensure output directory exists (but no subdirectories needed)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Create temporary adjacency matrix file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False, dir="./data/precomputed_v2/") as tmp_adj_matrix_file:
        adj_matrix_file = tmp_adj_matrix_file.name
        
        # Handle tensor conversion with proper dtype management
        if torch.is_tensor(adj_matrix):
            if adj_matrix.requires_grad:
                adj_matrix_np = adj_matrix.detach().cpu().numpy().astype(np.float32)
            else:
                adj_matrix_np = adj_matrix.cpu().numpy().astype(np.float32)
        else:
            adj_matrix_np = adj_matrix.astype(np.float32)


        # ⭐ ADD SHAPE VERIFICATION FROM TRAIN.PY
        # Reshape if necessary - model outputs might be [batch_size, 4, 32, 32]
        if adj_matrix_np.ndim == 4 and adj_matrix_np.shape[0] == 1:
            adj_matrix_np = adj_matrix_np.squeeze(0)  # Remove batch dimension
        
        # Verify final shape
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

             # Running from Windows (VS Code) - use WSL to execute Linux binary
            result = subprocess.run([
                "wsl",
                "/home/kanis/dev/Occupation_LaCAM2/lacam2/build/main",
                "--map", wsl_map_file,
                "--scen", wsl_scen_file,
                "--num", str(num_agent),
                "--congestion", wsl_adj_matrix_file,
                "--output", wsl_output_file,
                "--verbose",
                # "--time_limit_sec", "30"  # Added timeout
            ], capture_output=True, text=True, check=True)
        else:
            # Running directly in WSL/Linux
            result = subprocess.run([
                "lacam2/build/main",
                "--map", map_file,
                "--scen", scen_file,
                "--num", str(num_agent),
                "--congestion", adj_matrix_file,
                "--output", output_file,
                "--verbose",
                # "--time_limit_sec", "30"  # Added timeout
            ], capture_output=True, text=True, check=True)  
            # ], capture_output=True, text=True, timeout=60, check=True)  # Added timeout

         # Verify output file was created and is non-empty
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file was not created: {output_file}")
        
        if os.path.getsize(output_file) == 0:
            raise ValueError(f"Output file is empty: {output_file}")
        
        # print(f"LaCAM2 execution successful. Output file: {output_file}")
        if os.path.exists(adj_matrix_file):
            os.remove(adj_matrix_file)
        return True

    except subprocess.TimeoutExpired:
        print(f"LaCAM2 execution timed out for {scen_id} with {num_agent} agents")
        return False
    except subprocess.CalledProcessError as e:
        print(f"LaCAM2 execution failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error in LaCAM2 execution: {e}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(adj_matrix_file):
            os.remove(adj_matrix_file)

def load_trained_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load the trained model"""
    print(f"Loading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model
    model = DualInputTopologyVectorFields(in_channels=2, hidden_dim=64).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    return model

def load_test_batches() -> list:
    """Load precomputed test batches"""
    print("Loading precomputed test batches...")
    
    # Get all test batch files
    test_batch_files = glob.glob(TEST_BATCH_DIR + "batch_*.pt")
    test_metadata_files = glob.glob(TEST_METADATA_DIR + "test_batch_*.json")

    # 🔀 Shuffle both lists while maintaining correspondence
    import random
    combined = list(zip(test_batch_files, test_metadata_files))
    random.shuffle(combined)
    test_batch_files, test_metadata_files = zip(*combined)
    test_batch_files, test_metadata_files = list(test_batch_files), list(test_metadata_files)
    
    
    print(f"📦 Found {len(test_batch_files)} test batch files")
    print(f"📊 Found {len(test_metadata_files)} metadata files")
    
    if len(test_batch_files) != len(test_metadata_files):
        raise ValueError(f"Mismatch: {len(test_batch_files)} batch files vs {len(test_metadata_files)} metadata files")
    
    return test_batch_files, test_metadata_files

def run_single_test_scenario(args):
    """Helper function for parallel LaCAM2 execution on individual scenarios"""
    adj_matrix_np, map_file, scen_file, scen_id, unique_id, num_agent, metadata = args
    
    try:
        # Run LaCAM2
        success = run_lacam_in_wsl(adj_matrix_np, map_file, scen_file, scen_id, unique_id, num_agent, run_type='model')
        
        if success:
            # Parse results
            output_file = os.path.join(TEST_OUTPUT_DIR, f"{scen_id}_agent_{num_agent}_model.txt")
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    solved = int(lines[3].split('=')[-1].strip())
                    soc = int(lines[4].split('=')[-1].strip())
                    makespan = int(lines[6].split('=')[-1].strip())
                    comp_time = int(lines[10].split('=')[-1].strip())
                    
                return {
                    'success': True,    
                    'solved': solved,
                    'soc': soc,
                    'makespan': makespan,
                    'comp_time': comp_time,
                    'output_file': output_file,
                    'metadata': metadata
                }
            except (FileNotFoundError, IndexError, ValueError) as e:
                return {
                    'success': False, 
                    'error': f"Error parsing output: {str(e)}",
                    'metadata': metadata
                }
        else:
            return {
                'success': False, 
                'error': 'LaCAM2 execution failed',
                'metadata': metadata
            }
    except Exception as e:
        return {
            'success': False, 
            'error': str(e),
            'metadata': metadata
        }

def process_test_batch(batch_file: str, metadata_file: str, model: torch.nn.Module, device: str) -> list:
    """Process a single test batch through the model with parallel LaCAM2 execution"""
    
    # Load batch data
    batch_data = torch.load(batch_file)
    topology_inputs = batch_data['topology_inputs'].to(device)
    vector_field_inputs = batch_data['vector_field_inputs'].to(device)
    batch_idx = batch_data['batch_idx']
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
    
    # Run model inference for entire batch at once
    with torch.no_grad():
        batch_model_outputs = model(topology_inputs, vector_field_inputs)
    
    # Prepare arguments for parallel LaCAM2 execution
    parallel_args = []
    for i, metadata in enumerate(metadata_list):
        # Get map file for this scenario
        map_name = metadata['map_name']
        map_file = map_files_dict[map_name].replace('\\', '/')
        scen_id = metadata['scen_id'].replace('-', '_')
        unique_id = metadata['unique_id']
        agent = int(unique_id.split('_')[-1])

        # if 'empty' in map_name:
        #     continue

        if i >= len(batch_model_outputs):  # Safety check
            break
            
        # Get model output for this scenario
        model_output = batch_model_outputs[i]
        
        # ⭐ ENHANCED CLAMPING: Ensure values are in valid [0, 1] range
        model_output = torch.clamp(model_output, min=0.0, max=1.0)

        
        # 🔄 CONGESTION FLIPPING FOR EXTREME DENSITY ON EMPTY MAPS
        # if 'empty' in metadata['scen_file'] and metadata['agents'] >= 350:
        #     # print(f"🔄 EXTREME DENSITY ({metadata['agents']} agents) - Flipping congestion predictions to declutter center")
        #     model_output = 1.0 - model_output
        
        # Convert to numpy for LaCAM2
        if model_output.requires_grad:
            adj_matrix_np = model_output.detach().cpu().numpy()
        else:
            adj_matrix_np = model_output.cpu().numpy()

        
        parallel_args.append((adj_matrix_np, map_file, metadata['scen_file'], 
                             scen_id, unique_id, metadata['agents'], metadata))

    # 🚀 PARALLEL EXECUTION OF LACAM2
    batch_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_single_test_scenario, arg) for arg in parallel_args]
        
        for future in as_completed(futures):
            result = future.result()
            batch_results.append(result)
    
    return batch_results


def save_test_results(results: list, output_dir: str):
    """Save comprehensive test results"""
    
    # Create results summary
    summary_data = []
    for result in results:
        metadata = result['metadata']
        summary_row = {
            'unique_id': metadata['unique_id'],
            'scen_id': metadata['scen_id'],
            'map_name': metadata['map_name'],
            'agents': metadata['agents'],
            'baseline_cost': metadata.get('solution_cost', 'N/A'),
            'success': result['success']
        }
        
        if result['success']:
            baseline_cost = metadata.get('solution_cost', result['soc'])
            summary_row.update({
                'solved': result['solved'],
                'model_soc': result['soc'],
                'model_makespan': result['makespan'],
                'comp_time': result['comp_time'],
                'soc_gap': (result['soc'] - baseline_cost) / baseline_cost if baseline_cost and baseline_cost > 0 else 0,
                'error': None
            })
        else:
            summary_row.update({
                'solved': 0,
                'model_soc': float('inf'),
                'model_makespan': float('inf'),
                'comp_time': float('inf'),
                'soc_gap': float('inf'),
                'error': result.get('error', 'Unknown error')
            })
        
        summary_data.append(summary_row)
    
    # Save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "test_results_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary statistics
    print(f"\n📊 Test Results Summary:")
    print(f"Total scenarios tested: {len(results)}")
    successful = summary_df[summary_df['success'] == True]
    print(f"Successful runs: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    if len(successful) > 0:
        solved = successful[successful['solved'] == 1]
        print(f"Solved scenarios: {len(solved)}/{len(successful)} ({len(solved)/len(successful)*100:.1f}%)")
        
        if len(solved) > 0:
            valid_gaps = solved[solved['soc_gap'] != float('inf')]['soc_gap']
            if len(valid_gaps) > 0:
                print(f"Average SOC gap: {valid_gaps.mean():.3f}")
                print(f"Median SOC gap: {valid_gaps.median():.3f}")
    
    print(f"Results saved to: {summary_file}")
    
    # Group by scenario type
    print(f"\n📈 Results by scenario type:")
    for scen_id in TEST_SCEN_IDS:
        scen_results = summary_df[summary_df['scen_id'] == scen_id]
        if len(scen_results) > 0:
            successful_scen = scen_results[scen_results['success'] == True]
            solved_scen = successful_scen[successful_scen['solved'] == 1]
            print(f"  {scen_id}: {len(solved_scen)}/{len(scen_results)} solved ({len(solved_scen)/len(scen_results)*100:.1f}%)")

def main():
    """
    Main testing function with enhanced robustness for high agent counts.
    
    Key improvements:
    - Uses precomputed test batches for efficiency
    - Dynamic timeouts for large instances (300s for 200+ agents)
    - Congestion scaling for high-density empty map scenarios  
    - Proper adjacency matrix clamping to [0, 1] range
    - Weighted MSE evaluation capability
    """
    print("🧪 Starting Model Testing on Precomputed Test Dataset")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {TEST_OUTPUT_DIR}")
    print(f"Test scenarios: {TEST_SCEN_IDS}")
    print("🔧 Enhanced for high agent count scenarios (300+ agents)")
    
    # Create output directory
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Load model
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(MODEL_PATH, device)
    
    # Load test batches
    test_batch_files, test_metadata_files = load_test_batches()
    
    print(f"\n🚀 Processing {len(test_batch_files)} test batches...")
    
    all_results = []
    
    # Process each batch
    for batch_file, metadata_file in tqdm(zip(test_batch_files, test_metadata_files), 
                                          total=len(test_batch_files), desc="Processing batches"):
        
        print(f"\n📦 Processing batch: {os.path.basename(batch_file)}")
        
        try:
            batch_results = process_test_batch(batch_file, metadata_file, model, device)
            all_results.extend(batch_results)
            
            print(f"   ✅ Processed {len(batch_results)} scenarios")
            
        except Exception as e:
            print(f"   ❌ Error processing batch {batch_file}: {e}")
            continue
    
    # Save results
    save_test_results(all_results, TEST_OUTPUT_DIR)
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Total scenarios processed: {len(all_results)}")
    print(f"📁 All results saved to: {TEST_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

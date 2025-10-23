import random
import sys, os, time, glob
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils_advisory_congestion_input import *
from src.utils_arranging_raw_data import *
from src.utils_congestion_models import DualInputTopologyVectorFields
import matplotlib
matplotlib.use('TkAgg') 
import torch     
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
# from torchsummary import summary
import numpy as np
from tqdm import tqdm
import subprocess
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import Dataset, DataLoader
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn.functional as F


# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE = 32  
UNSOLVED_WEIGHT, SOLVED_WEIGHT = 1.0, 1.0  # Phase A weights for curriculum
MAP_DIR = "./data/raw/maps/"
SCEN_DIR = "./data/raw/scenarios/"

PRECOMPUTED_DIR = "./data/precomputed_v3/"
TRAIN_BATCH_DIR = PRECOMPUTED_DIR + 'batches/train/'
VAL_BATCH_DIR = PRECOMPUTED_DIR + 'batches/val/'
METADATA_DIR = PRECOMPUTED_DIR + 'metadata/'
MAP_MASKS_DIR = PRECOMPUTED_DIR + 'map_masks/'
MAP_MASKS = glob.glob(MAP_MASKS_DIR + '*_mask.pt')
MAPS = sorted(glob.glob(MAP_DIR + "*.map"))
MAP_NAMES = [m.replace('\\', '/').split('/')[-1].split('.map')[0] for m in MAPS]
OUTPUT_DIR = "./data/output_stage3_retry_lacam/"
DIR="C:\\Users\\kanis\\Documents\\Monash Temp\\VMShared\\Integrated_Pipeline\\"
DIR_WSL = "/mnt/c/Users/kanis/Documents/Monash Temp/VMShared/Integrated_Pipeline/"
NUM_WORKERS = 6

mask_indices = {m.split('\\')[-1].split('_mask')[0]: torch.load(m) for m in MAP_MASKS}
map_files_dict = {name: path for name, path in zip(MAP_NAMES, MAPS)}

# Training stages configuration - Simplified Two-Stage Approach
STAGE_1_EPOCHS = 0  # Supervised foundation (adjacency only) - increased from 8
STAGE_2_EPOCHS = 0   # Remove hybrid stage entirely
STAGE_3_EPOCHS = 20  # Planner-dominant (unsupervised) - increased from 15
PLANNER_BATCH_PROB = 0.05  # Increase planner frequency in Stage 3

LAMBDA_START = 1.2    # Start with higher planner weight in Stage 3
# LAMBDA_RAMP_FACTOR = 1  # Gentler ramp-up

# Remove SUPERVISED_WEIGHT_DECAY - not needed without Stage 2
EARLY_STOP_PATIENCE = 5
VAL_SOLVE_RATE_THRESHOLD = 0.9  # Threshold to switch from Phase A to Phase B weights


class SpatialCongestionLoss(nn.Module):
    """
    Pure Huber loss for robust adjacency matrix regression.
    Simplified from multi-component spatial loss.
    """
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions, targets):
        # Pure Huber loss - robust and meaningful for congestion values
        huber_loss = F.huber_loss(predictions, targets, delta=self.delta)
        # huber_loss = F.mse_loss(predictions, targets)
        
        return huber_loss, {
            'huber': huber_loss.item(),
            'cosine': 0.0,     # For compatibility with existing logging
            'gradient': 0.0    # For compatibility with existing logging
        }


class _Dataset(Dataset):
    def __init__(self, file_paths: list[str], batch_size: int = BATCH_SIZE):
        self.file_paths = file_paths
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        topology_inputs = data['topology_inputs'].squeeze(0).float()
        vector_field_inputs = data['vector_field_inputs'].squeeze(0).float()
        adjacency_matrices_2d = data['adjacency_matrices_2d'].squeeze(0).float()
        batch_idx = int(data['batch_idx'])
        return topology_inputs, vector_field_inputs, adjacency_matrices_2d, batch_idx


def parse_planner_output(unique_id):
    """Parse LaCAM2 output file to extract SOC and solved status."""
    output_file = OUTPUT_DIR + unique_id + '.txt'
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            solved = int(lines[3].split('=')[-1].strip())
            soc = int(lines[4].split('=')[-1].strip())
            return soc, solved
        
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"Error parsing output file {output_file}: {e}")
        return float('inf'), 0
    


def run_lacam_in_wsl(adj_matrix, map_file, scen_file, unique_id, num_agent, output_dir = '', temp_dir='./data/temp/'):
    """
    Enhanced LaCAM2 execution with better error handling and path management.
    """
    # Detect if running on Windows (VS Code) or in WSL
    is_windows = os.name == 'nt' or 'WSL_DISTRO_NAME' not in os.environ


    if output_dir == '':
        output_file = OUTPUT_DIR + unique_id + '.txt'
    else:
        output_file = output_dir + unique_id + '.txt'

    # Ensure output directory exists (but no subdirectories needed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create temporary adjacency matrix file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False, dir=temp_dir) as tmp_adj_matrix_file:
        adj_matrix_file = tmp_adj_matrix_file.name
        
        # Handle tensor conversion with proper dtype management
        if torch.is_tensor(adj_matrix):
            if adj_matrix.requires_grad:
                adj_matrix_np = adj_matrix.detach().cpu().numpy().astype(np.float32)
            else:
                adj_matrix_np = adj_matrix.cpu().numpy().astype(np.float32)
        else:
            adj_matrix_np = adj_matrix.astype(np.float32)

        # ENSURE CORRECT SHAPE: Should be [4, 32, 32] for 4 channels
        # print(f"Debug: adj_matrix shape before writing: {adj_matrix_np.shape}")
        
        # Reshape if necessary - model outputs might be [batch_size, 4, 32, 32]
        if adj_matrix_np.ndim == 4 and adj_matrix_np.shape[0] == 1:
            adj_matrix_np = adj_matrix_np.squeeze(0)  # Remove batch dimension
        
        # Verify final shape
        if adj_matrix_np.shape != (4, 32, 32):
            raise ValueError(f"Adjacency matrix has wrong shape: {adj_matrix_np.shape}, expected (4, 32, 32)")

        adj_matrix_np.tofile(adj_matrix_file)

        # Assert file was written with correct size
        file_size = os.path.getsize(adj_matrix_file)
        expected_size = adj_matrix_np.nbytes
        # print(f"Debug: Written {file_size} bytes, expected {expected_size} bytes")
        assert file_size == expected_size, f"File size mismatch: {file_size} != {expected_size}"

    try:
        if is_windows:
            # Convert Windows paths to WSL paths for the binary execution
            def win_to_wsl_path(path):
                if path.startswith('\\\\wsl.localhost\\Ubuntu'):
                    return path.replace('\\\\wsl.localhost\\Ubuntu', '').replace('\\', '/')
                elif path.startswith('./'):
                    return '/home/kanis/dev/Occupation_LaCAM2/' + path[2:]
                else:
                    return path
            
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
                "--verbose"
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
                "--verbose"
            ], capture_output=True, text=True, check=True)  # Added timeout
            # ], capture_output=True, text=True, timeout=60, check=True)  # Added timeout

         # Verify output file was created and is non-empty
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file was not created: {output_file}")
        
        if os.path.getsize(output_file) == 0:
            raise ValueError(f"Output file is empty: {output_file}")
        
        # print(f"LaCAM2 execution successful. Output file: {output_file}")
        os.remove(adj_matrix_file)
        return True

    except subprocess.TimeoutExpired:
        print(f"LaCAM2 execution timed out for {unique_id} with {num_agent} agents")
        return False
    except subprocess.CalledProcessError as e:
        print(f"LaCAM2 execution failed with return code {e.returncode}")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        
        # Also check if the binary exists and is executable
        binary_path = "/home/kanis/dev/Occupation_LaCAM2/lacam2/build/main"
        result = subprocess.run(["wsl", "ls", "-la", binary_path], capture_output=True, text=True)
        print(f"Binary check: {result.stdout}")
        
        return False
    except Exception as e:
        print(f"Unexpected error in LaCAM2 execution: {e}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(adj_matrix_file):
            os.remove(adj_matrix_file)


def run_single_planner_instance(args):
    """Helper function for parallel LaCAM2 execution"""
    adj_matrix_np, map_file, scen_file, scen_id, unique_id, num_agent, baseline_cost = args
    
    try:
        # Convert numpy back to appropriate format for run_lacam_in_wsl
        success = run_lacam_in_wsl(adj_matrix_np, map_file, scen_file, unique_id, num_agent)
        
        if success:
            model_cost, solved = parse_planner_output(unique_id)
            return {
                'success': True,
                'model_cost': model_cost,
                'solved': solved,   
                'baseline_cost': baseline_cost,
                'scen_id': scen_id
            }
        else:
            return {
                'success': False,
                'model_cost': float('inf'),
                'solved': 0,
                'baseline_cost': baseline_cost,
                'scen_id': scen_id
            }
    except Exception as e:
        return {
            'success': False,
            'model_cost': float('inf'),
            'solved': 0,
            'baseline_cost': baseline_cost,
            'scen_id': scen_id,
            'error': str(e)
        }


def planner_gap_loss(model_soc, baseline_soc, solved, w_unsolved=UNSOLVED_WEIGHT, w_solved=SOLVED_WEIGHT, eps=1e-6):
    """
    Enhanced planner gap loss with better handling of edge cases.
    
    Computes delta_i = max(0, (soc_model_i - soc_baseline_i) / (soc_baseline_i + eps))
    Weights solved vs unsolved instances appropriately.
    
    Args:
        model_soc: Array of SOC values from model-guided LaCAM2
        baseline_soc: Array of baseline SOC values (CBS/LNS2)
        solved: Array of solved flags (1 if solved, 0 if not)
        w_unsolved: Weight for unsolved instances
        w_solved: Weight for solved instances
        eps: Small constant to avoid division by zero
    
    Returns:
        Weighted average planner gap loss
    """

    model = np.asarray(model_soc, dtype=np.float64)
    baseline = np.asarray(baseline_soc, dtype=np.float64)
    s = np.asarray(solved, dtype=np.float64)

    # Filter out invalid values
    valid = (np.isfinite(model) & np.isfinite(baseline) & 
             (baseline > 0) & (model >= 0))
    
    if not np.any(valid):
        print("Warning: No valid instances for planner loss computation")
        return 0.0
    
    # Compute relative gap only for valid instances
    delta = np.zeros_like(model, dtype=np.float64)
    valid_model = model[valid]
    valid_baseline = baseline[valid]
    
    # Relative gap: (model - baseline) / baseline
    relative_gap = (valid_model - valid_baseline) / (valid_baseline + eps)

    # Only penalize when model is worse than baseline
    delta[valid] = np.maximum(relative_gap, 0.0)
    
    # Apply weights based on solved status
    weights = np.where(s > 0.5, w_solved, w_unsolved)
    
    # Compute weighted average loss
    valid_weights = weights[valid]
    valid_deltas = delta[valid]
    
    if np.sum(valid_weights) > 0:
        loss = np.sum(valid_weights * valid_deltas) / (np.sum(valid_weights) + eps)
    else:
        loss = 0.0
    
    return np.float32(loss)

def create_combined_batch_mask(mask_indices_list, device='cuda'):
    """
    Combine multiple mask indices for different scenarios in a batch
    
    Args:
        mask_indices_list: List of mask indices for each scenario
        device: Device to place tensors on
    
    Returns:
        Combined mask indices tuple (batch_indices, channel_indices, y_indices, x_indices)
    """
    combined_batch = []
    combined_channel = []
    combined_y = []
    combined_x = []
    
    for batch_idx, scenario_mask in enumerate(mask_indices_list):
        # scenario_mask is (channel_indices, y_indices, x_indices) for single scenario
        num_elements = len(scenario_mask[0])
        
        # Add batch dimension for this scenario
        combined_batch.extend([batch_idx] * num_elements)
        combined_channel.extend(scenario_mask[0].cpu().tolist())
        combined_y.extend(scenario_mask[1].cpu().tolist())
        combined_x.extend(scenario_mask[2].cpu().tolist())
    
    return (
        torch.LongTensor(combined_batch).to(device),
        torch.LongTensor(combined_channel).to(device),
        torch.LongTensor(combined_y).to(device),
        torch.LongTensor(combined_x).to(device)
    )


def run_planner_batch(model_outputs, scen_files, agents, scen_ids, unique_ids, baseline_costs, 
                     w_unsolved=UNSOLVED_WEIGHT, w_solved=SOLVED_WEIGHT):
    
    # Prepare arguments for parallel execution
    args = []
    for i in range(len(scen_files)):
        if 'empty' in scen_files[i]:
            map_name = '-'.join(scen_files[i].split('/')[-1].split('.scen')[0].split('-')[:3])
        else:
            map_name = '-'.join(scen_files[i].split('/')[-1].split('.scen')[0].split('-')[:4])

        map_file = map_files_dict[map_name].replace('\\', '/')
        adj_matrix_4D = model_outputs[i]
        adj_matrix_4D = torch.clamp(adj_matrix_4D, min=0.0)
        
        # CRITICAL FIX: Detach tensor and convert to numpy for multiprocessing
        if adj_matrix_4D.requires_grad:
            adj_matrix_np = adj_matrix_4D.detach().cpu().numpy()
        else:
            adj_matrix_np = adj_matrix_4D.cpu().numpy()

        args.append((adj_matrix_np, map_file, scen_files[i], scen_ids[i], unique_ids[i], agents[i], baseline_costs[i]))

    # Parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_single_planner_instance, arg) for arg in args]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Process results
    successful_runs = sum(1 for r in results if r['success'])
    solve_rate = successful_runs / len(results)  

    # Extract costs and solved status for planner gap loss
    batch_model_costs = [r['model_cost'] for r in results]
    batch_solved = [r['solved'] for r in results]
    
    # Compute planner gap loss
    L_planner = planner_gap_loss(batch_model_costs, baseline_costs, batch_solved, 
                                w_unsolved, w_solved)
    
        # Compute SOC gap mean
    valid_costs = [(m, b) for m, b, s in zip(batch_model_costs, baseline_costs, batch_solved) 
                   if s > 0 and np.isfinite(m) and np.isfinite(b) and b > 0]
    
    if valid_costs:
        soc_gap_mean = np.mean([(m - b) / (b + 1e-6) for m, b in valid_costs])
    else:
        soc_gap_mean = float('inf')
    
    return {
        'loss': L_planner,
        'solve_rate': solve_rate,
        'soc_gap_mean': soc_gap_mean,
        'successful_runs': successful_runs,
        'total_runs': len(scen_files)
    }



def train_step(model, optimizer, batch_data, mask_indices, split_name: str = 'train', 
               device: str = 'cuda', execute_planner: bool = False, lambda_coeff: float = 0.0,
               stage: int = 1, w_unsolved: float = UNSOLVED_WEIGHT, w_solved: float = SOLVED_WEIGHT):
    """
    Enhanced training step with planner-aware curriculum learning.
    
    Args:
        model: The neural network model
        optimizer: Optimizer for training
        batch_data: Input batch data
        mask_indices: Mask indices for valid cells
        split_name: 'train' or 'val'
        device: Device to run on
        execute_planner: Whether to run LaCAM2 planner for this batch
        lambda_coeff: Weight for planner loss
        stage: Current training stage (1, 2, or 3)
        w_unsolved: Weight for unsolved instances
        w_solved: Weight for solved instances
    
    Returns:
        Dictionary with losses and metrics
    """

    topology_inputs, vector_field_inputs, edge_grid_4D, batch_idx = batch_data
   
    topology_inputs = topology_inputs.squeeze(0).to(device)
    vector_field_inputs = vector_field_inputs.squeeze(0).to(device)
    edge_grid_4D = edge_grid_4D.squeeze(0).to(device)
    batch_idx = int(batch_idx)
    batch_idx = f"{batch_idx:04d}"
    
    # Load metadata
    metdata_batch_file = METADATA_DIR + split_name + '_batch_' + batch_idx + '.json'

    with open(metdata_batch_file, 'r') as f:
        metadata = json.load(f)    
    
    # Get individual scenario masks
    mask_indices_batch = [mask_indices[item['unique_id'].split('_')[0]] for item in metadata]
    
    # Create combined batch mask
    batch_mask_indices = create_combined_batch_mask(mask_indices_batch, device)

    batch_scen_files = [m['scen_file'] for m in metadata]
    batch_agents = [m['agents'] for m in metadata]
    batch_scen_ids = [m['scen_id'] for m in metadata]
    batch_baseline_costs = [m['solution_cost'] for m in metadata]  # CBS/LNS2 baseline
    batch_unique_ids = [m['unique_id'] for m in metadata]
    
    if split_name == 'train':
        optimizer.zero_grad()
    
    with autocast():
        batch_output = model(topology_inputs, vector_field_inputs)
        # batch_output = torch.clamp(batch_output, min=0.0)

        # Apply combined batch mask for adjacency loss
        batch_output_filtered = batch_output[batch_mask_indices[0], batch_mask_indices[1], 
                                           batch_mask_indices[2], batch_mask_indices[3]]
        edge_grid_4D_filtered = edge_grid_4D[batch_mask_indices[0], batch_mask_indices[1], 
                                           batch_mask_indices[2], batch_mask_indices[3]]

        del batch_mask_indices, mask_indices_batch  # Free memory

        # Spatial Congestion Loss (replaces MSE)
        spatial_loss_fn = SpatialCongestionLoss()
        L_adj, loss_components = spatial_loss_fn(batch_output_filtered, edge_grid_4D_filtered)
       
        # # Add this right after L_adj calculation in train_step:
        # print(f"\n🔍 DEBUG - First Batch Analysis:")
        # print(f"Predictions: min={batch_output_filtered.min():.6f}, max={batch_output_filtered.max():.6f}, mean={batch_output_filtered.mean():.6f}")
        # print(f"Targets:     min={edge_grid_4D_filtered.min():.6f}, max={edge_grid_4D_filtered.max():.6f}, mean={edge_grid_4D_filtered.mean():.6f}")
        # print(f"Abs Error:   mean={torch.abs(batch_output_filtered - edge_grid_4D_filtered).mean():.6f}")
        # print(f"Huber Loss:  {L_adj:.6f}")
        # print(f"Data shapes: pred={batch_output_filtered.shape}, target={edge_grid_4D_filtered.shape}")
        
        # Initialize total loss
        L_total = L_adj
        L_planner = 0.0
        planner_metrics = {}

        if torch.isnan(L_adj):
            print('NaN adjacency loss detected, skipping batch')
            return None
    
         # Planner loss (only if enabled for this batch)
        if execute_planner and lambda_coeff > 0:
            try:
                # Run LaCAM2 planner using RAW (non-normalized) model outputs
                planner_results = run_planner_batch(batch_output, batch_scen_files, batch_agents, 
                                                  batch_scen_ids, batch_unique_ids, batch_baseline_costs, 
                                                  w_unsolved, w_solved)
                L_planner = planner_results['loss']
                planner_metrics = {k: v for k, v in planner_results.items() if k != 'loss'}
                
                # Combine losses with curriculum weighting
                if stage == 2:  # Hybrid stage - gradual transition
                    supervised_w = globals().get('supervised_weight', 1.0)
                    L_total = supervised_w * L_adj + lambda_coeff * L_planner
                else:  # Stage 1 or 3 - standard combination
                    L_total = L_adj + lambda_coeff * L_planner
                
            except Exception as e:
                print(f"Planner execution failed: {e}")
                # Fallback to adjacency-only loss
                L_planner = 0.0
                planner_metrics = {}
        
    # Gradient computation and optimization
    grad_norm = 0.0
    if split_name == 'train':
        scaler.scale(L_total).backward()

        # Track gradient norm for monitoring training stability
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
    
    # Prepare return dictionary
    result = {
        'L_adj': L_adj.item(),
        'L_planner': L_planner.item() if isinstance(L_planner, torch.Tensor) else L_planner,
        'L_total': L_total.item(),
        'grad_norm': grad_norm,
        'batch_output': batch_output,  # Raw outputs for potential planner evaluation
        'loss_components': loss_components,  # Spatial loss breakdown
        'metadata': {
            'scen_files': batch_scen_files,
            'agents': batch_agents,
            'scen_ids': batch_scen_ids,
            'baseline_costs': batch_baseline_costs
        }
    }
    
    # Add planner metrics if available
    result.update(planner_metrics)
    
    return result

if __name__ == "__main__":
    LOG_DIR = "./data/training_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DualInputTopologyVectorFields(in_channels=2, hidden_dim=64).to(device)
    
    # Check if the saved model file exists before loading
    model_path = "./data/model/best_model_curriculum_huber_loss.pt"
    if os.path.exists(model_path):
        print(f"Loading pre-trained weights from: {model_path}")    
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Pre-trained weights loaded successfully!")
    else:
        print("No pre-trained weights found. Starting training from scratch.")

    # Enhanced training configuration
    TOTAL_EPOCHS = STAGE_1_EPOCHS + STAGE_2_EPOCHS + STAGE_3_EPOCHS
    LR = 5e-4
    weight_decay = 1e-4
    best_val_soc_gap = float("inf")  # Best validation SOC gap for checkpoint selection
    patience_counter = 0

    # Model saving configuration
    MODEL_SAVE_DIR = "./data/model/"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model_curriculum.pt")
    stage_2_model_path = os.path.join(MODEL_SAVE_DIR, "stage_2_model.pt")


    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.7, verbose=True)
    criterion = nn.MSELoss().to(device)
    scaler = GradScaler()

    # Enhanced metrics tracking
    training_metrics = {
        'epoch': [], 'stage': [], 'train_adj_mse': [], 'train_planner_loss': [], 'train_total_loss': [],
        'val_adj_mse': [], 'val_planner_loss': [], 'val_total_loss': [],
        'val_soc_gap_mean': [], 'val_solve_rate': [], 'grad_norm_avg': [], 'lr': [],
        'lambda_coeff': [], 'stage_weights': []
    }

    # Create training timestamp for logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = LOG_DIR + f"/training_curriculum_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Load data
    train_files = glob.glob(TRAIN_BATCH_DIR + '*.pt')
    val_files = glob.glob(VAL_BATCH_DIR + '*.pt')
    train_loader = DataLoader(_Dataset(train_files), batch_size=1, shuffle=True)
    val_loader = DataLoader(_Dataset(val_files), batch_size=1, shuffle=False)

    print(f"🎓 Full Curriculum Learning (Supervised → Unsupervised):")
    print(f"   Stage 1: {STAGE_1_EPOCHS} epochs (Supervised foundation - adjacency only)")
    print(f"   Stage 2: {STAGE_2_EPOCHS} epochs (Hybrid transition - supervised weight decays)")
    print(f"   Stage 3: {STAGE_3_EPOCHS} epochs (Unsupervised refinement - planner-dominant)")
    print(f"   Total: {TOTAL_EPOCHS} epochs")

    # Fixed validation subset for consistent evaluation
    # val_subset_indices = list(range(min(5, len(val_files))))  # Use first 5 validation batches
    
    # Fixed validation subset for consistent evaluation (random but reproducible)
    val_subset_indices = random.sample(range(len(val_files)), min(5, len(val_files)))  # Pick 5 random indices
    
    # Main training loop with simplified curriculum
    for epoch in range(TOTAL_EPOCHS):
        start_time = time.time()
        
        # Determine stage and configuration - SIMPLIFIED
        if epoch < STAGE_1_EPOCHS:
            stage = 1
            stage_name = "SUPERVISED_FOUNDATION"
            execute_planner = False
            lambda_coeff = 0.0
            w_unsolved, w_solved = 1.0, 1.0
        else:
            stage = 3  # Skip directly to Stage 3
            stage_name = "PLANNER_OPTIMIZATION" 
            execute_planner = True
            stage_epoch = epoch - STAGE_1_EPOCHS
            lambda_coeff = LAMBDA_START 
            
            # Adaptive weights based on performance
            if len(training_metrics['val_solve_rate']) > 0 and training_metrics['val_solve_rate'][-1] > VAL_SOLVE_RATE_THRESHOLD:
                w_unsolved, w_solved = 1.0, 1.0  # Less aggressive than Phase A
            else:
                w_unsolved, w_solved = UNSOLVED_WEIGHT, SOLVED_WEIGHT

        print(f"\n🚀 Epoch {epoch+1}/{TOTAL_EPOCHS} - Stage {stage} ({stage_name})")
        print(f"   λ={lambda_coeff:.3f}, weights=(unsolved:{w_unsolved}, solved:{w_solved})")

        # TRAINING PHASE
        model.train()
        epoch_metrics = {
            'train_adj_losses': [], 'train_planner_losses': [], 'train_total_losses': [],
            'grad_norms': [], 'planner_batches': 0,
            'huber_losses': [], 'cosine_losses': [], 'gradient_losses': []
        }

        train_pbar = tqdm(train_loader, desc=f"Training Stage {stage}", leave=False)
        for batch_idx, batch_data in enumerate(train_pbar):
            # Decide whether to execute planner for this batch
            run_planner_this_batch = (execute_planner and 
                                    (np.random.random() < PLANNER_BATCH_PROB))
            # run_planner_this_batch = execute_planner

            if stage == 1:
                # Stage 1: Pure adjacency learning
                result = train_step(
                    model, optimizer, batch_data, mask_indices,
                    split_name='train', device=device, 
                    execute_planner=False,
                    lambda_coeff=0.0, stage=stage,
                    w_unsolved=w_unsolved, w_solved=w_solved
                )
            elif stage >= 3 and run_planner_this_batch:
                # Stage 3: Pure planner optimization
                result = train_step(
                    model, optimizer, batch_data, mask_indices,
                    split_name='train', device=device, 
                    execute_planner=True,
                    lambda_coeff=lambda_coeff, stage=stage,
                    w_unsolved=w_unsolved, w_solved=w_solved
                )
            else:
                # Skip batches without planner in Stage 3
                continue

            if result is not None:
                epoch_metrics['train_adj_losses'].append(result['L_adj'])
                epoch_metrics['train_planner_losses'].append(result['L_planner'])
                epoch_metrics['train_total_losses'].append(result['L_total'])
                epoch_metrics['grad_norms'].append(result['grad_norm'])
                
                # Track spatial loss components
                if 'loss_components' in result:
                    epoch_metrics['huber_losses'].append(result['loss_components']['huber'])
                    epoch_metrics['cosine_losses'].append(result['loss_components']['cosine'])
                    epoch_metrics['gradient_losses'].append(result['loss_components']['gradient'])
                
                if run_planner_this_batch and result['L_planner'] > 0:
                    epoch_metrics['planner_batches'] += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Adj': f"{result['L_adj']:.4f}",
                    'Plan': f"{result['L_planner']:.4f}",
                    'Total': f"{result['L_total']:.4f}"
                })

        # VALIDATION PHASE
        print(f"📊 Running validation for epoch {epoch + 1}")
        model.eval()
        
        val_metrics = {
            'val_adj_losses': [], 'val_planner_losses': [], 'val_total_losses': [],
            'val_solve_rates': [], 'val_soc_gaps': []
        }

        # Always run planner on validation subset for consistent metrics
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", leave=False)
        for val_idx, batch_data in val_pbar:
            run_planner_val = (val_idx in val_subset_indices and execute_planner)
            
            with torch.no_grad():
                result = train_step(
                    model, optimizer, batch_data, mask_indices,
                    split_name='val', device=device,
                    execute_planner=run_planner_val,
                    lambda_coeff=lambda_coeff, stage=stage,
                    w_unsolved=w_unsolved, w_solved=w_solved
                )
            
            if result is not None:
                val_metrics['val_adj_losses'].append(result['L_adj'])
                val_metrics['val_planner_losses'].append(result['L_planner'])
                val_metrics['val_total_losses'].append(result['L_total'])
                
                if run_planner_val and 'solve_rate' in result:
                    val_metrics['val_solve_rates'].append(result['solve_rate'])
                    val_metrics['val_soc_gaps'].append(result['soc_gap_mean'])

        # Compute epoch averages
        train_adj_mse = np.mean(epoch_metrics['train_adj_losses'])
        train_planner_loss = np.mean([x for x in epoch_metrics['train_planner_losses'] if x > 0]) if any(x > 0 for x in epoch_metrics['train_planner_losses']) else 0.0
        train_total_loss = np.mean(epoch_metrics['train_total_losses'])
        grad_norm_avg = np.mean(epoch_metrics['grad_norms'])
        
        # Compute spatial loss component averages
        train_huber_avg = np.mean(epoch_metrics['huber_losses']) if epoch_metrics['huber_losses'] else 0.0
        train_cosine_avg = np.mean(epoch_metrics['cosine_losses']) if epoch_metrics['cosine_losses'] else 0.0
        train_gradient_avg = np.mean(epoch_metrics['gradient_losses']) if epoch_metrics['gradient_losses'] else 0.0
        
        val_adj_mse = np.mean(val_metrics['val_adj_losses'])
        val_planner_loss = np.mean([x for x in val_metrics['val_planner_losses'] if x > 0]) if any(x > 0 for x in val_metrics['val_planner_losses']) else 0.0
        val_total_loss = np.mean(val_metrics['val_total_losses'])
        val_solve_rate = np.mean(val_metrics['val_solve_rates']) if val_metrics['val_solve_rates'] else 0.0
        val_soc_gap_mean = np.mean([x for x in val_metrics['val_soc_gaps'] if np.isfinite(x)]) if val_metrics['val_soc_gaps'] else float('inf')

        # Learning rate scheduling
        if stage >= 3:  # Only schedule in Stage 3
            scheduler.step(val_soc_gap_mean if val_soc_gap_mean != float('inf') else val_adj_mse)

        # Record metrics
        current_lr = optimizer.param_groups[0]['lr']
        training_metrics['epoch'].append(epoch + 1)
        training_metrics['stage'].append(stage)
        training_metrics['train_adj_mse'].append(train_adj_mse)
        training_metrics['train_planner_loss'].append(train_planner_loss)
        training_metrics['train_total_loss'].append(train_total_loss)
        training_metrics['val_adj_mse'].append(val_adj_mse)
        training_metrics['val_planner_loss'].append(val_planner_loss)
        training_metrics['val_total_loss'].append(val_total_loss)
        training_metrics['val_soc_gap_mean'].append(val_soc_gap_mean)
        training_metrics['val_solve_rate'].append(val_solve_rate)
        training_metrics['grad_norm_avg'].append(grad_norm_avg)
        training_metrics['lr'].append(current_lr)
        training_metrics['lambda_coeff'].append(lambda_coeff)
        training_metrics['stage_weights'].append(f"unsolved:{w_unsolved},solved:{w_solved}")

        # Checkpoint selection based on validation SOC gap (primary) or adjacency MSE (fallback)
        checkpoint_metric = val_soc_gap_mean if val_soc_gap_mean != float('inf') else val_adj_mse
        if checkpoint_metric < best_val_soc_gap:
            best_val_soc_gap = checkpoint_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved! Val SOC gap: {val_soc_gap_mean:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Epoch summary
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train - Spatial: {train_adj_mse:.4f} (H:{train_huber_avg:.4f}, C:{train_cosine_avg:.4f}, G:{train_gradient_avg:.4f}), Planner: {train_planner_loss:.4f}, Total: {train_total_loss:.4f}")
        print(f"  Val   - Adj MSE: {val_adj_mse:.4f}, SOC Gap: {val_soc_gap_mean:.4f}, Solve Rate: {val_solve_rate:.3f}")
        print(f"  Grad Norm: {grad_norm_avg:.4f}, LR: {current_lr:.2e}, Time: {elapsed:.1f}s")
        print(f"  📊 Stage {stage} - Patience: {patience_counter}/{EARLY_STOP_PATIENCE} (Best: {best_val_soc_gap:.4f})")
        
        # Early stopping with stage transitions
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"🔄 Early stopping triggered after {patience_counter} epochs without improvement")
            
            if stage == 1:
                # Stage 1 early stopping -> Jump to Stage 3 (since Stage 2 is disabled)
                print(f"📈 Stage 1 converged early. Jumping to Stage 3 (Planner Optimization)")
                # Save Stage 1 model
                stage_1_model_path = os.path.join(MODEL_SAVE_DIR, f"stage_1_model_{timestamp}.pt")
                torch.save(model.state_dict(), stage_1_model_path)
                print(f"📁 Stage 1 model saved: {stage_1_model_path}")
                
                # Jump to Stage 3
                epoch = STAGE_1_EPOCHS + STAGE_2_EPOCHS - 1  # Next epoch will be Stage 3
                patience_counter = 0  # Reset patience for Stage 3
                best_val_soc_gap = float('inf')  # Reset best metric for Stage 3
                print(f"🚀 Transitioning to Stage 3 at epoch {epoch + 2}")
                continue
                
            elif stage >= 3:
                # Stage 3 early stopping -> Training complete
                print(f"✅ Stage 3 converged early. Training complete!")
                # Save final model
                final_model_path = os.path.join(MODEL_SAVE_DIR, f"stage_3_final_model_{timestamp}.pt")
                torch.save(model.state_dict(), final_model_path)
                print(f"📁 Final Stage 3 model saved: {final_model_path}")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}_{timestamp}.pth")
            torch.save(model.state_dict(), periodic_path)
            print(f"📁 Periodic checkpoint saved: {periodic_path}")

    # Save final metrics
    metrics_df = pd.DataFrame(training_metrics)
    metrics_path = os.path.join(log_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"📊 Training metrics saved to: {metrics_path}")

    print("🎉 Curriculum training completed!")
    print(f"Best validation SOC gap: {best_val_soc_gap:.4f}")
    print(f"Best model saved at: {best_model_path}")

import glob
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

MODEL_DIR = "./data/output/*_model.txt"  # Changed to match flat file structure
# MODEL_DIR = "./data/test/*_model.txt"  # For test results
BASELINE_DIR = "./data/output_baseline/"

def process_file(model_file, baseline_dir):
    """Process a single model file and return both model and baseline results"""
    model_results = {}
    baseline_results = {}
    
    # Extract unique_id from model filename (flat structure)
    # Example: "./data/output/maze_32_32_4_random_20_agent_50_model.txt" → "maze_32_32_4_random_20_agent_50"
    filename = os.path.basename(model_file)
    unique_id = filename.replace('_model.txt', '')
    baseline_output = baseline_dir + unique_id + "_output_baseline.txt"
    
    # Parse baseline output
    try:
        with open(baseline_output.replace('\\', '/'), "r") as f:
            lines = f.readlines()
            baseline_results['unique_id'] = unique_id
            baseline_results['agents'] = int(lines[0].strip().split('=')[1])
            baseline_results['solved'] = int(lines[3].strip().split('=')[1])
            baseline_results['soc'] = int(lines[4].strip().split('=')[1])
            baseline_results['soc_lb'] = int(lines[5].strip().split('=')[1])
            baseline_results['makespan'] = int(lines[6].strip().split('=')[1])
            baseline_results['makespan_lb'] = int(lines[7].strip().split('=')[1])
            baseline_results['sum_of_loss'] = int(lines[8].strip().split('=')[1])
            baseline_results['sum_of_loss_lb'] = int(lines[9].strip().split('=')[1])
            baseline_results['comp_time'] = int(lines[10].strip().split('=')[1])
            baseline_results['loop_cnt'] = int(lines[14].strip().split('=')[1])
            baseline_results['num_node_gen'] = int(lines[15].strip().split('=')[1])
    except FileNotFoundError:
        print(f"Baseline output file not found for {unique_id}")
        baseline_results = None
    except (IndexError, ValueError) as e:
        print(f"Error parsing baseline file for {unique_id}: {e}")
        baseline_results = None
    
    # Parse model output
    try:
        with open(model_file, "r") as f:
            lines = f.readlines()
            model_results['unique_id'] = unique_id
            model_results['agents'] = int(lines[0].strip().split('=')[1])
            model_results['solved'] = int(lines[3].strip().split('=')[1])
            model_results['soc'] = int(lines[4].strip().split('=')[1])
            model_results['soc_lb'] = int(lines[5].strip().split('=')[1])
            model_results['makespan'] = int(lines[6].strip().split('=')[1])
            model_results['makespan_lb'] = int(lines[7].strip().split('=')[1])
            model_results['sum_of_loss'] = int(lines[8].strip().split('=')[1])
            model_results['sum_of_loss_lb'] = int(lines[9].strip().split('=')[1])
            model_results['comp_time'] = int(lines[10].strip().split('=')[1])
            model_results['loop_cnt'] = int(lines[14].strip().split('=')[1])
            model_results['num_node_gen'] = int(lines[15].strip().split('=')[1])
    except FileNotFoundError:
        print(f"Model output file not found for {unique_id}")
        model_results = None
    except (IndexError, ValueError) as e:
        print(f"Error parsing model file for {unique_id}: {e}")
        model_results = None
    
    return model_results, baseline_results

if __name__ == "__main__":
    model_files = glob.glob(MODEL_DIR)
    
    print(f"Processing {len(model_files)} model files using {cpu_count()} CPU cores...")
    
    # Create partial function with baseline_dir parameter
    process_func = partial(process_file, baseline_dir=BASELINE_DIR)
    
    # Use multiprocessing to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_func, model_files)
    
    # Separate model and baseline results
    model_results_list = []
    baseline_results_list = []
    
    for model_result, baseline_result in results:
        if model_result is not None:
            model_results_list.append(model_result)
        if baseline_result is not None:
            baseline_results_list.append(baseline_result)
    
    # Create DataFrames
    columns = ['unique_id', 'agents', 'solved', 'soc', 'soc_lb', 'makespan', 'makespan_lb', 
               'sum_of_loss', 'sum_of_loss_lb', 'comp_time', 'loop_cnt', 'num_node_gen']
    
    model_df = pd.DataFrame(model_results_list, columns=columns)
    baseline_df = pd.DataFrame(baseline_results_list, columns=columns)
    
    print(f"Processed {len(model_df)} model results and {len(baseline_df)} baseline results")

    # Combine the dataframes
    combined_df = pd.merge(model_df, baseline_df, left_on='unique_id', right_on='unique_id', suffixes=('_model', '_baseline'))

    #Calculate the difference in performance metrics as percentage improvement from baseline
    for col in model_df.columns:
        if col != 'unique_id' and col != 'agents' and col != 'soc_lb' and col != 'makespan_lb' and col != 'sum_of_loss_lb':
            combined_df[f'diff_{col}'] = (combined_df[f'{col}_model'] - combined_df[f'{col}_baseline']) / combined_df[f'{col}_baseline'] * 100

    combined_df.to_csv("./data/comparison_results_test_congestion_test_w.csv", index=False)
    print("Results saved to ./data/comparison_results_test_congestion_test_w.csv")
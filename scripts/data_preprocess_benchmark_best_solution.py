import pandas as pd
import argparse
import os,re 
import numpy as np

def node_frequency_count(solution_path, makespan):
    grid = np.zeros((32, 32), dtype=np.float32)

    for timestep in range(1, makespan + 1):
        path = solution_path.split('\n')[timestep].strip()
        agent_locations = [(int(x), int(y)) for x, y in path]

        for loc in agent_locations:
            x, y = loc
            if 0 <= x < 32 and 0 <= y < 32:
                grid[x, y] += 1

    return grid


def edge_frequency_count(solution_path, makespan):
    # For 32x32 grid, we have 1024 nodes (32*32)
    # Create adjacency matrix: 1024 x 1024
    grid_size = 32
    num_nodes = grid_size * grid_size
    edge_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    def coord_to_index(x, y):
        return x * grid_size + y
    
    def index_to_coord(idx):
        return idx // grid_size, idx % grid_size
    
    for timestep in range(1, makespan + 1):
        path = solution_path.split('\n')[timestep].strip()
        agent_locations = [(int(x), int(y)) for x, y in path]
        
        for i in range(len(agent_locations) - 1):
            x1, y1 = agent_locations[i]
            x2, y2 = agent_locations[i + 1]
            
            if (0 <= x1 < grid_size and 0 <= y1 < grid_size and 
                0 <= x2 < grid_size and 0 <= y2 < grid_size):
                
                idx1 = coord_to_index(x1, y1)
                idx2 = coord_to_index(x2, y2)
                
                # For unordered pairs, ensure consistent ordering
                min_idx, max_idx = min(idx1, idx2), max(idx1, idx2)
                edge_matrix[min_idx, max_idx] += 1
    
    return edge_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default='data/benchmark_with_best_solution/even_1_agent_60_CBSH2_RTC_solution.csv')
    args = parser.parse_args()
    csv_file = args.csv_file
    csv_file = 'data/benchmark_with_best_solution/even_1_agent_60_CBSH2_RTC_solution.csv'
    file_name = os.path.basename(csv_file)
    file_name = file_name.replace('.csv', '_grid_bin.npy')

    df = pd.read_csv(csv_file)
    num_agents = df['agents'].iloc[0]
    solution_path = df['solution_path'].iloc[0]
    makespan = max(len(solution_path.split('\n')), 0)
    df['makespan'] = makespan

    df.to_csv(csv_file, index=False)

    grid = node_frequency_count(solution_path, makespan)
    np.save(f'data/benchmark_with_best_solution/{file_name}', grid)

    edge_grid = edge_frequency_count(solution_path, makespan)
    np.save(f'data/benchmark_with_best_solution/{file_name.replace("_grid_bin.npy", "_edge_grid_bin.npy")}', edge_grid)

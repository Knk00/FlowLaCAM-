import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_file', type=str, required=True, help='Path to solution file')
    parser.add_argument('--output_bin', type=str, required=False, default=None, help='Path to save occupancy .bin file')
    parser.add_argument('--reference', type=str, required=False, default=None, help='Path to get models bin file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save output files')
    args = parser.parse_args()
    solution_file = args.solution_file
    output_bin = args.output_bin
    reference = args.reference
    save_dir = args.save_dir

    # solution_file = "/home/kanishk/Downloads/lacam2-dev/data/output/random-32-32-20/random_25_agent_200.txt"
    # reference = False
    # Assume 32x32 grid (can be parameterized)
    grid_size = 32
    occupancy = np.zeros((grid_size, grid_size), dtype=np.float32)

    with open(solution_file, 'r') as f:
        lines = f.readlines()
    
    agents = int(lines[0].strip().split('=')[1]) # Get agent IDs from the first line
    makespan = int(lines[6].strip().split('=')[1]) # Get makespan from the sixth line

    agent_paths = [[] for _ in range(agents)]
    for timestep in range(1, makespan + 1):
        line = lines[18 + timestep].strip()
        path = re.findall(r'\((\d+),(\d+)\)', line)
        agent_locations = [(int(x), int(y)) for x, y in path]

        for agent in range(agents):
            agent_paths[agent].append(agent_locations[agent])

    # Remove the mode of locations of agents
    for agent in range(agents):
        if agent_paths[agent]:
            mode_location = max(set(agent_paths[agent]), key=agent_paths[agent].count)
            agent_paths[agent] = [loc for loc in agent_paths[agent] if loc != mode_location]
            agent_paths[agent].append(mode_location)  # Ensure the last location is included

    occupancy_matrix = np.zeros((grid_size, grid_size), dtype=np.float32)
    all_positions = [(x, y) for path in agent_paths for (x, y) in path]

    # Build occupancy grid
    for (x, y) in all_positions:
        if 0 <= x < grid_size and 0 <= y < grid_size:
                occupancy_matrix[x, y] += 1

    #Read reference occupancy if provided
    if reference:
        #It is a binary file with 32x32 float32 occupancy
        ref_occupancy = np.fromfile(reference, dtype=np.float32).reshape((grid_size, grid_size))

    #Plot heatmaps together
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Plot reference heatmap
    sns.heatmap(ref_occupancy, cmap='Reds', linecolor='black', linewidth=0.4, ax=ax[0])
    ax[0].set_title("Model's Occupancy Heatmap")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    # Plot heatmap
    sns.heatmap(occupancy_matrix.T, cmap='Reds', linecolor='black', linewidth=0.4, ax=ax[1])
    ax[1].set_title("Occupancy Heatmap after integration of LaCAM2 and Occupancy model")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    plt.tight_layout()
    # plt.show()
    fig.savefig(f"{save_dir}/lacam2_occupancy.png", bbox_inches='tight', dpi=300)

    plt.close(fig)
    


    # Save as .bin (float32, row-major)
    if output_bin is None:
        output_bin = solution_file.replace('.txt', '_occupancy.bin')
    occupancy.astype(np.float32).tofile(output_bin)
    print(f"Occupancy matrix saved to {output_bin}")

    # plt.plot(figsize=(12, 5))
    # sns.heatmap(occupancy_matrix.T, cmap='Reds', linecolor='black', linewidth=0.4)
    # plt.title("LaCAM2 Occupancy Heatmap")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.tight_layout()
    # plt.show()
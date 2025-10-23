"""
Utility functions to parse scenario files and maps for the arranging domain.
"""
import pandas as pd
import numpy as np


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
          'start_location_x': int,
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
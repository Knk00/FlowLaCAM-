""""
This file contains utility functions for processing advisory congestion input data.
"""

import pandas as pd
import numpy as np
import torch
import re

def expand_run_length_encoded_path(rle_path):
    """
    Efficiently converts run-length encoded path to expanded format.
    
    Examples:
        '3r3u5lu8l' -> 'rrrruuullllulllllll'  
        '2r4u' -> 'rruuuu'
        'r3u2l' -> 'ruuull'
    
    Args:
        rle_path: String with run-length encoded moves (e.g., '3r2u5l')
        
    Returns:
        String with expanded moves (e.g., 'rrruuulllll')
    """
    if not rle_path or rle_path is None:
        return ""
    
    # Use regex to find all number+direction pairs
    # Pattern: optional number (defaults to 1) followed by direction letter
    pattern = r'(\d*)([urdl])'
    matches = re.findall(pattern, rle_path)
    
    expanded_moves = []
    for count_str, direction in matches:
        # If no number prefix, default to 1
        count = int(count_str) if count_str else 1
        expanded_moves.append(direction * count)
    
    return ''.join(expanded_moves)

def calculate_goal_location(df, plan_col='solution_plan', start_col='start_location'):
    calculated_goal_location = []
    for i in range(len(df)):
        row = df.iloc[i]
        x, y = row['start_location']
        plan = row['solution_plan']
        uid = row['unique_id']
        
        if plan == None:
            calculated_goal_location.append(None)
            continue
            
        # Expand run-length encoded path first
        expanded_plan = expand_run_length_encoded_path(plan)
        x += expanded_plan.count('r') - expanded_plan.count('l')
        y += expanded_plan.count('u') - expanded_plan.count('d')
        
        calculated_goal_location.append((x, y))
        
    df['calculated_goal_location'] = calculated_goal_location
    
    return df


def calculate_curr_position(timestep, agent_solution_plan, start_loc):
    # First expand the run-length encoded path
    expanded_plan = expand_run_length_encoded_path(agent_solution_plan)
    
    # Then take the slice up to timestep
    plan = expanded_plan[:timestep]
    x, y = start_loc
    x += plan.count('r') - plan.count('l')
    y += plan.count('u') - plan.count('d')
    return (x, y)

def min_max_scaling(grid, mask_value=-10):
    # Create a mask to identify non-obstacle values
    mask = (grid != mask_value)
    
    # Extract only valid values
    grid_mask = grid[mask]
    
    # Compute min and max from non-obstacle values
    grid_min = grid_mask.min()
    grid_max = grid_mask.max()
    
    # Create a float copy of the grid (so obstacle values remain unchanged)
    scaled_grid = grid.astype(float)

    # Apply Min-Max Scaling only on non-obstacle values
    scaled_grid[mask] = (grid_mask - grid_min) / (grid_max - grid_min)
    
    return scaled_grid


def create_direction_field(start, goal, map_shape):
    """
    Creates a direction field showing the movement intention
    from start to goal across the entire map.
    
    Args:
        start: tuple (x, y) of start position
        goal: tuple (x, y) of goal position
        map_shape: tuple (height, width) of the map
    
    Returns:
        Tuple of (field_x, field_y) 2D numpy arrays with direction components
    """
    height, width = map_shape
    
    # Calculate the direction vector from start to goal
    dir_vector = np.array([goal[0] - start[0], goal[1] - start[1]])
    
    # Normalize if not zero
    norm = np.linalg.norm(dir_vector)
    if norm > 0:
        dir_vector = dir_vector / norm
    
    # Create coordinate matrices for vectorized operations
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate distances from each cell to the start position (vectorized)
    distances = np.sqrt((x_coords - start[0])**2 + (y_coords - start[1])**2)
    
    # Add a small constant to avoid division by zero
    distances = np.maximum(distances, 0.1)
    
    # Calculate influence (inversely proportional to distance squared)
    influence = 1.0 / (distances**2)
    
    # Compute field components using broadcasting
    field_x = dir_vector[0] * influence
    field_y = dir_vector[1] * influence  # Fixed: was using dir_vector[0] before
    
    # Normalize the fields
    x_max = np.max(np.abs(field_x))
    if x_max > 0:
        field_x = field_x / x_max
        
    y_max = np.max(np.abs(field_y))
    if y_max > 0:
        field_y = field_y / y_max
        
    return field_x, field_y

def create_aggregate_direction_fields(starts, goals, map_shape= (32, 32)):
    """
    Creates aggregate direction fields for multiple agents.
    
    Args:
        starts: List of (x, y) tuples for start positions
        goals: List of (x, y) tuples for goal positions
        map_shape: tuple (height, width) of the map
        
    Returns:
        Tuple of (x_field, y_field) containing the aggregate directional information
    """
    height, width = map_shape
    x_field = np.zeros((height, width))
    y_field = np.zeros((height, width))


    # Process all agents in one go if there are many
    if len(starts) > 10:
        # Pre-allocate arrays for all agent fields
        all_x_fields = np.zeros((len(starts), height, width))
        all_y_fields = np.zeros((len(starts), height, width))
        
        # Calculate individual fields
        for i, (start, goal) in enumerate(zip(starts, goals)):
            all_x_fields[i], all_y_fields[i] = create_direction_field(start, goal, map_shape)
            
        # Sum across all agents
        x_field = np.sum(all_x_fields, axis=0)
        y_field = np.sum(all_y_fields, axis=0)
    else:
        # Original approach for smaller numbers of agents
        for start, goal in zip(starts, goals):
            field_x, field_y = create_direction_field(start, goal, map_shape)
            x_field += field_x
            y_field += field_y
    
    # Normalize the combined fields
    x_max = np.max(np.abs(x_field))
    if x_max > 0:
        x_field = x_field / x_max
        
    y_max = np.max(np.abs(y_field))
    if y_max > 0:
        y_field = y_field / y_max
        
    return x_field, y_field


def coord_to_index(x, y, grid_size=32):
    # Convert coordinates to linear index
    # Consistent with C++ implementation: xu = index % W, yu = index / W
    # Therefore: index = y * grid_size + x
    return y * grid_size + x


def index_to_coord(idx, grid_size=32):
    # Convert linear index back to coordinates
    # Consistent with C++ implementation: xu = index % W, yu = index / W
    x = idx % grid_size     # xu = n->index % W
    y = idx // grid_size    # yu = n->index / W
    return x, y


def edge_frequency_count(solution_path, num_agents, start_locations, goal_locations, channels: int=4):
    grid_size = 32
    channels = 4 
    """
    Channel representations:
    0: North
    1: South
    2: East
    3: West
    """
    edge_matrix = np.zeros((channels, grid_size, grid_size), dtype=np.int32)
    skipped_agents = 0
    for agent in range(num_agents):
        start_location = start_locations[agent]
        goal_location = goal_locations[agent]
        path = solution_path[agent]
        
        if start_location == goal_location:
            continue

        # Expand run-length encoded path first
        expanded_path = expand_run_length_encoded_path(path)

        agent_locations = [calculate_curr_position(i, expanded_path, start_location) for i in range(len(expanded_path) + 1)]

        try:
            assert agent_locations[0] == start_location, f"Agent {agent} start location mismatch: {agent_locations[0]} vs {start_location}"
            assert agent_locations[-1] == goal_location, f"Agent {agent} goal location mismatch: {agent_locations[-1]} vs {goal_location}"
        except AssertionError as e:
            print(f"⚠️ Skipping agent {agent}: {e}")
            skipped_agents += 1
            continue
            
        if skipped_agents > num_agents * 0.3:
            print("⚠️ Too many agents skipped due to path inconsistencies. Aborting edge frequency calculation.")
            return None
        
        try:
            for i in range(len(agent_locations)-1):
                x1, y1 = agent_locations[i]
                x2, y2 = agent_locations[i + 1]
                # print(agent_locations[i], 'to', agent_locations[i+1])

                dx = x2-x1; dy=y2-y1

                if dx==dy==0:
                    continue

                # channel_selector = {(1,0): 0, (-1,0): 1, (0, 1): 2, (0,-1): 3}
                channel_selector = {(1,0): 2, (-1,0): 3, (0,1): 1, (0,-1): 0}  # East, West, South, North
                channel = channel_selector[(dx, dy)]
                edge_matrix[channel, y2, x2] += 1
        except Exception as e:
            print(f"Error processing agent {agent} path: {e}")
            continue

    #min max norm
    edge_matrix = (edge_matrix - np.min(edge_matrix)) / (np.max(edge_matrix) - np.min(edge_matrix))

    return edge_matrix

  
def create_dynamic_mask(grid, height, width, batch_size):
    """Create mask for the actual batch size"""
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[grid == 1] = 1.0
    mask_extended = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 4, -1, -1)
    return torch.where(mask_extended != 1.0)

def weighted_mse_loss(pred, target, weight_nonzero=3.0, weight_zero=1.0, threshold=0.15):
    """
    Emphasize learning on high-congestion areas while still learning low-congestion patterns.
    """
    # Create weights based on target values
    weights = torch.where(target > threshold, weight_nonzero, weight_zero)
    
    # Compute weighted MSE
    squared_errors = (pred - target) ** 2
    weighted_errors = weights * squared_errors
    
    return torch.mean(weighted_errors)


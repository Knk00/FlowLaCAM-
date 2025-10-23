#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "planner.hpp"
#include "post_processing.hpp"
#include "utils.hpp"

// main function  
Solution solve(const Instance& ins, std::string& additional_info, bool& is_intermediate_solution,
               const int verbose = 0, const Deadline* deadline = nullptr,
               std::mt19937* MT = nullptr, const Objective objective = OBJ_NONE,
               const float restart_rate = 0.001, 
               const std::string& congestion_path = "data/processed/random-32-32-20/even_11_agent_11_reconstructed.bin",
               const std::string& highway_dist_output = "", bool intermediate = false, 
               int intermediate_freq = 0);

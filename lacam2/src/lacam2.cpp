#include "../include/lacam2.hpp"

Solution solve(const Instance& ins, std::string& additional_info, bool& is_intermediate_solution,
               const int verbose, const Deadline* deadline, std::mt19937* MT,
               const Objective objective, const float restart_rate, const std::string& congestion_path,
               const std::string& highway_dist_output, bool intermediate, int intermediate_freq)
{
  auto planner = Planner(&ins, deadline, MT, verbose, objective, restart_rate, congestion_path, intermediate, intermediate_freq);
  auto solution = planner.solve(additional_info, is_intermediate_solution);
  
  return solution;
}

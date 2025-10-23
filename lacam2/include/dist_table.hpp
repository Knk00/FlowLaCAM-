/*
 * distance table with lazy evaluation, using BFS
 */
#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

struct DistTable {
  const uint V_size;  // number of vertices
  std::vector<std::vector<uint> >
      table;          // distance table, index: agent-id & vertex-id
  std::vector<std::queue<Vertex*> > OPEN;  // search queue

  inline uint get(uint i, uint v_id);      // agent, vertex-id
  uint get(uint i, Vertex* v);             // agent, vertex

  DistTable(const Instance& ins);
  DistTable(const Instance* ins);

  void setup(const Instance* ins);  // initialization
};


struct HighwayDistTable{
  const uint V_size;  // number of vertices
  std::vector<std::vector<float>> table;  // distance table, index: agent-id & vertex-id
  std::vector<std::queue<Vertex*>> OPEN; // search queue

  inline float get(uint i, uint v_id);   // agent, vertex-id
  float get(uint i, Vertex* v);          // agent, vertex

  std::vector<std::vector<std::vector<float>>> congestion_grid;
  const Instance* ins; // instance

  // Dijkstra structures to avoid reinitialization
  using QueueElement = std::pair<float, uint>; // {distance, vertex_id}
  std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>> pq;
  std::vector<float> distances;

  HighwayDistTable(const Instance* ins, const std::string& congestion_path);

  void setup(const Instance* ins, const std::string& congestion_path);
 
};


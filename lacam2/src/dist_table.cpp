#include "../include/dist_table.hpp"
// For direction mapping
#include <unordered_map>

// Hash function for std::pair<int, int>
struct pair_hash {
  std::size_t operator()(const std::pair<int, int>& p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
  }
};

float epsilon = 1e-6f; // Small value to avoid division by zero

std::vector<std::vector<std::vector<float>>> edgeMatrix(const std::string& path) {
    size_t channels = 4;
    size_t height = 32;
    size_t width = 32;
    size_t expected_bytes = channels * height * width * sizeof(float);

    std::ifstream infile(path, std::ios::binary | std::ios::ate);
    if (!infile) throw std::runtime_error("Cannot open file: " + path);

    std::streamsize file_size = infile.tellg();
    
    
    if (file_size != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("Adjacency Matrix size mismatch: expected " +
            std::to_string(expected_bytes) + " bytes, got " + std::to_string(file_size));
    }
    
    infile.seekg(0, std::ios::beg);

    // Properly initialize a 3D vector [channels][height][width]
    std::vector<std::vector<std::vector<float>>> grid(
        channels,
        std::vector<std::vector<float>>(
            height,
            std::vector<float>(width)
        )
    );

    // Read data for each channel, one row at a time
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            infile.read(reinterpret_cast<char*>(grid[c][y].data()), width * sizeof(float));
            if (infile.fail()) {
                throw std::runtime_error("Failed to read adjacency matrix data at channel " + 
                    std::to_string(c) + ", row " + std::to_string(y));
            }
        }
    }
    
    return grid;
}


DistTable::DistTable(const Instance& ins)
    : V_size(ins.G.V.size()), table(ins.N, std::vector<uint>(V_size, V_size))
{
  setup(&ins);
}

DistTable::DistTable(const Instance* ins)
    : V_size(ins->G.V.size()), table(ins->N, std::vector<uint>(V_size, V_size))
{
  setup(ins);
}

HighwayDistTable::HighwayDistTable(const Instance* ins, const std::string& congestion_path)
    : V_size(ins->G.V.size()), table(ins->N, std::vector<float>(V_size, std::numeric_limits<float>::max())),
      ins(ins), distances(V_size, std::numeric_limits<float>::infinity())
{
  setup(ins, congestion_path);
}


void DistTable::setup(const Instance* ins)
{
  for (size_t i = 0; i < ins->N; ++i) {
    OPEN.push_back(std::queue<Vertex*>());
    auto n = ins->goals[i];
    OPEN[i].push(n);
    table[i][n->id] = 0;
  }
}


void HighwayDistTable::setup(const Instance* ins, const std::string& congestion_path)
{
  if (!congestion_path.empty()) {
    // Load adjacency matrix with highway multipliers (already float)
    congestion_grid = edgeMatrix(congestion_path);
  }

  for (size_t i = 0; i < ins->N; ++i) {
    OPEN.push_back(std::queue<Vertex*>());
    auto n = ins->goals[i];
    OPEN[i].push(n);
    table[i][n->id] = 0;
  }
  this->ins = ins;
}

uint DistTable::get(uint i, uint v_id)
{
  if (table[i][v_id] < V_size) return table[i][v_id];

  /*
   * BFS with lazy evaluation
   * c.f., Reverse Resumable A*
   * https://www.aaai.org/Papers/AIIDE/2005/AIIDE05-020.pdf
   *
   * sidenote:
   * tested RRA* but lazy BFS was much better in performance
   */

  while (!OPEN[i].empty()) {
    auto&& n = OPEN[i].front();
    OPEN[i].pop();
    const int d_n = table[i][n->id];
    for (auto&& m : n->neighbor) {
      const int d_m = table[i][m->id];
      if (d_n + 1 >= d_m) continue;
      table[i][m->id] = d_n + 1;
      OPEN[i].push(m);
    }
    if (n->id == v_id) return d_n;
  }
  return V_size;
}

uint DistTable::get(uint i, Vertex* v) { return get(i, v->id); }

// Backward Dijkstra with congestion-aware edge costs
float HighwayDistTable::get(uint i, uint v_id) {
    const uint W = ins->G.width;

    // If already computed, return cached distance
    if (table[i][v_id] < V_size) {
        return table[i][v_id];
    }

    // Clear and reinitialize Dijkstra structures
    while (!pq.empty()) pq.pop();
    std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::infinity());

    // Start from the goal
    uint goal_id = ins->goals[i]->id;
    distances[goal_id] = 0.0f;
    pq.push({0.0f, goal_id});

    while (!pq.empty()) {
        auto [current_dist, current_id] = pq.top();
        pq.pop();

        // Early exit if we reach the target
        if (current_id == v_id) {
            table[i][v_id] = current_dist; // cache result
            return current_dist;
        }

        // Skip if the distance is outdated
        if (current_dist > distances[current_id]) {
            continue;
        }

        Vertex* n = ins->G.V[current_id];

        // Expand neighbors (backward Dijkstra: incoming edges)
        for (Vertex* m : n->neighbor) {
            uint neighbor_id = m->id;
            
            // Skip self-edges (staying in same location)
            if (neighbor_id == current_id) continue;
            
            // Calculate edge cost with highway multipliers
            float edgeCost = 1.0f; // Base movement cost
            if (!congestion_grid.empty()) {
                uint n_idx = n->index;
                // FIX 1: Correct coordinate conversion
                uint ny = n_idx / W, nx = n_idx % W;  // y = row, x = col

                uint m_idx = m->index;
                uint my = m_idx / W, mx = m_idx % W;  // y = row, x = col

                int dx = (int)mx - (int)nx; // neighbor_x - current_x
                int dy = (int)my - (int)ny; // neighbor_y - current_y

                // FIX 2: Correct channel mapping (match Python)
                std::unordered_map<std::pair<int, int>, int, pair_hash> edge_costs;
                edge_costs[{1, 0}] = 0;   // East: dx=+1 → Channel 2
                edge_costs[{-1, 0}] = 1;  // West: dx=-1 → Channel 3
                edge_costs[{0, 1}] = 3;   // South: dy=+1 → Channel 1
                edge_costs[{0, -1}] = 2;  // North: dy=-1 → Channel 0

                // Apply highway (1.0 - congestion_value for cost reduction)
                auto it = edge_costs.find({dx, dy});
                if (it != edge_costs.end()) {
                    int channel = it->second;
                    float congestion = congestion_grid[channel][my][mx]; // destination cell
                    edgeCost += congestion; // congestion aware
                }
            }
            
            float new_dist = current_dist + edgeCost;
      
            if (new_dist < distances[neighbor_id]) {
                distances[neighbor_id] = new_dist;
                pq.push({new_dist, neighbor_id});
              
            }
        }
    }

    // If unreachable
    return std::numeric_limits<float>::infinity();
}

float HighwayDistTable::get(uint i, Vertex* v) {
    return get(i, v->id);
}


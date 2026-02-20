# (AL-generated) Codex Node Importance Analysis for `examples/london-transportation`
`IMPORTANT`: This file is AI-generated and not checked by human yet.

## 1. Objective
Identify the most effective metric to represent node importance for the graph used in:

- `examples/london-transportation/main.py`
- data file: `examples/london-transportation/Urban Geography.graphml`

The candidate metrics implemented in the code are:

- Degree centrality
- Betweenness centrality
- Closeness centrality
- PageRank

## 2. Graph Characteristics
The GraphML file was inspected with NetworkX.

- Graph type: `MultiDiGraph` (directed)
- Nodes: `1024`
- Edges: `2028`
- Undirected connected components: `8`
- Largest connected component (LCC): `964` nodes
- Undirected density: `0.003871884164222874`

Because distance-based metrics can be unstable across disconnected parts, metric comparison focused on the largest connected component of the undirected projection.

## 3. Evaluation Design
### 3.1 Why simple top-k lists are not enough
Comparing only “top nodes” does not prove which metric captures structurally critical nodes best.

### 3.2 Robustness-based comparison
For each metric:

1. Rank nodes by metric score (descending).
2. Remove top nodes sequentially (`K = 10, 20, 50, 100`).
3. After each removal, measure the fraction of nodes remaining in the largest connected component (`LCC fraction`).
4. Compute area under the LCC-fraction curve (`AUC_LCC`) using trapezoidal integration.

Interpretation:

- Lower `AUC_LCC` means faster structural collapse under targeted removal.
- Therefore, lower `AUC_LCC` indicates a metric that more effectively identifies high-impact nodes.

## 4. Quantitative Results
### 4.1 Attack simulation summary (lower is better)

| K (removed nodes) | PageRank | Degree | Betweenness | Closeness |
|---|---:|---:|---:|---:|
| 10  | **7.498** | 7.533 | 7.681 | 7.938 |
| 20  | **14.219** | 14.398 | 14.488 | 16.519 |
| 50  | **23.250** | 24.182 | 28.551 | 41.182 |
| 100 | **26.849** | 27.728 | 35.864 | 72.425 |

### 4.2 LCC fraction at the final removal step

| K | PageRank | Degree | Betweenness | Closeness |
|---|---:|---:|---:|---:|
| 10  | 0.724 | 0.743 | 0.758 | 0.863 |
| 20  | 0.631 | 0.643 | 0.610 | 0.854 |
| 50  | 0.168 | 0.109 | 0.230 | 0.749 |
| 100 | 0.038 | 0.047 | 0.048 | 0.537 |

Observation:

- For `K = 10` and `K = 20`, PageRank consistently gives the strongest early collapse (best practical behavior when only a small number of critical nodes can be targeted).
- For larger K, Degree becomes competitive in endpoint fragmentation, but PageRank remains best or near-best by cumulative collapse (`AUC_LCC`).
- Closeness performs clearly worse for this graph.

## 5. Metric Correlation (Spearman rank)

| Pair | Spearman rho |
|---|---:|
| Degree vs PageRank | 0.867 |
| Degree vs Betweenness | 0.695 |
| Betweenness vs PageRank | 0.654 |
| Betweenness vs Closeness | 0.462 |
| Degree vs Closeness | 0.296 |
| Closeness vs PageRank | 0.160 |

Interpretation:

- Degree and PageRank are strongly aligned, but not identical.
- Closeness is weakly aligned with PageRank and Degree, consistent with its weaker robustness performance here.

## 6. Top Nodes (examples)
Top 5 by each metric on the LCC:

### Degree
1. elsevier bv
2. taylor & francis
3. landscape and urban planning
4. sage publishing
5. wiley-blackwell

### Betweenness
1. elsevier bv
2. landscape and urban planning
3. taylor & francis
4. wiley-blackwell
5. urban geography

### Closeness
1. elsevier bv
2. landscape and urban planning
3. applied geography
4. applied energy
5. sustainable cities and society

### PageRank
1. elsevier bv
2. taylor & francis
3. landscape and urban planning
4. wiley-blackwell
5. sage publishing

## 7. Conclusion
For this dataset, **PageRank is the most effective single metric** for node importance if the goal is to identify a limited number of high-impact nodes that most quickly degrade global connectivity.

Recommended default in `examples/london-transportation/main.py`:

- Use `metric_name = "pagerank"` as the primary importance metric.

Contextual note:

- If the objective is large-scale progressive dismantling with many removals, Degree can be a strong secondary metric to compare.

## 8. Reproducibility Notes
- Library used: NetworkX (plus SciPy for rank correlation in post-check)
- Graph projection for comparison: undirected largest connected component
- Main robustness score: AUC of LCC fraction under targeted removal

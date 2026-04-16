# (AI-generated) Codex Node Importance Analysis for `examples/london-transportation`
`CAUTION`: This file is AI-generated and not checked by human yet.

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

## 7. Extended Search Beyond the Original Four Metrics
To test whether better alternatives exist, additional metrics were computed and evaluated with the same targeted-removal protocol.

### 7.1 Additional candidates
- Eigenvector centrality
- Harmonic centrality
- Load centrality
- Core number
- Collective Influence (`CI`, radius `l=2` and `l=3`)
- HITS hub score
- HITS authority score
- Objective-driven structural scores: `component_increase`, `lcc_drop_1step`, and `articulation_binary`

### 7.2 Technical normalization for fair comparison
Some metrics are not implemented on multigraphs in NetworkX, so the comparison graph was normalized to:

- Largest connected component of the undirected projection
- Converted from `MultiGraph` to simple `Graph`
- Resulting comparison graph: `964` nodes, `1203` edges

This affects absolute values slightly but keeps cross-metric ranking fair.

### 7.3 Expanded metric ranking by robustness (AUC_LCC, lower is better)

#### K = 10
| Metric | AUC_LCC | LCC@10 |
|---|---:|---:|
| PageRank | **7.627** | 0.765 |
| Degree | 7.646 | 0.765 |
| Load | 7.675 | 0.769 |
| Betweenness | 7.681 | 0.758 |
| CI (l=2) | 7.732 | 0.767 |
| CI (l=3) | 7.844 | 0.807 |

#### K = 20
| Metric | AUC_LCC | LCC@20 |
|---|---:|---:|
| PageRank | **14.111** | 0.585 |
| Degree | 14.272 | 0.596 |
| Load | 14.479 | 0.610 |
| Betweenness | 14.488 | 0.610 |
| CI (l=2) | 14.583 | 0.620 |
| CI (l=3) | 14.733 | 0.592 |

#### K = 50
| Metric | AUC_LCC | LCC@50 |
|---|---:|---:|
| CI (l=3) | **22.718** | 0.163 |
| Degree | 25.635 | 0.144 |
| PageRank | 26.982 | 0.132 |
| CI (l=2) | 26.992 | 0.189 |
| Load | 28.199 | 0.230 |
| Betweenness | 28.551 | 0.230 |

#### K = 100
| Metric | AUC_LCC | LCC@100 |
|---|---:|---:|
| Degree | **29.050** | 0.026 |
| CI (l=3) | 29.655 | 0.078 |
| PageRank | 30.038 | 0.034 |
| CI (l=2) | 32.662 | 0.053 |
| Load | 35.780 | 0.048 |
| Betweenness | 35.864 | 0.048 |

### 7.4 Aggregate view (across K = 10, 20, 50, 100)
Using rank-sum over AUC across all K:

1. Degree
2. PageRank
3. Collective Influence (l=3)
4. Load
5. Collective Influence (l=2)
6. Betweenness

Lower-performing metrics on this graph:
- Closeness
- Eigenvector
- HITS (hub/authority)
- Core number

### 7.5 Interpretation
- If the objective is **early-stage disruption** (small intervention budget, e.g., top 10-20 nodes), **PageRank** remains the best single choice.
- If the objective is **mid-scale dismantling** (around top 50 removals), **Collective Influence (l=3)** outperforms the original four metrics.
- For **large-scale removals** (around top 100), **Degree** is strongest, with CI(l=3) and PageRank close behind.
- Structure-only one-step scores (`lcc_drop_1step`, `component_increase`) are informative diagnostics but underperform as standalone ranking metrics for sequential removal.

## 8. Updated Recommendation
There is no universal winner independent of objective. A practical policy is:

- Default metric in `examples/london-transportation/main.py`: `pagerank`
- Secondary metric for dismantling analysis: `collective_influence_l3`
- Keep `degree` as a high-quality baseline for large-K stress tests

If only one metric can be shipped for general use, keep **PageRank** due to its strongest early-impact behavior and stable overall performance.

## 9. Reproducibility Notes
- Library used: NetworkX (SciPy only for rank-correlation in prior section)
- Graph source: `examples/london-transportation/Urban Geography.graphml`
- Primary evaluation graph: undirected largest connected component
- Extended search normalization: simple `Graph` projection to support all candidate metrics
- Main robustness score: AUC of LCC fraction under targeted node removal

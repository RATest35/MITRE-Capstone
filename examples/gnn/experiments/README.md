# Experiments Overview

This directory stores results of many experiments for the GNN model that predicts composite_score.

Each folder usually has these files.

- composite_score_gnn.pt
- composite_score_predictions.csv
- metrics.json

Notes:

- New experiments store config in metrics.json.
- Some old experiments do not have config.
- If config is missing, it is estimated from the folder name and past experiment history.

## Experiment List

| Folder                        | Description                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| output_prod_mps_fast        | Early production training run. Old baseline model. Details are estimated because config is missing. |
| quick_1hop                  | Early 1-hop baseline. No config.                                                                  |
| quick_1hop_n32              | 1-hop baseline with neighbor size 32.                                                             |
| quick_1hop_n48              | 1-hop baseline with neighbor size 48.                                                             |
| quick_1hop_h512             | 1-hop baseline with hidden_dim=512.                                                               |
| quick_1hop_h512_lr1e4       | Same as quick_1hop_h512 but learning rate is 1e-4.                                                |
| quick_1hop_top5             | 1-hop experiment. Likely focused on top 5%, but no config.                                        |
| search_e1                   | One of the early search runs. Details unknown.                                                        |
| ext_n32                     | Baseline with extended features. /24 split, linear weight, no ranking loss.                     |
| ext_16                      | extended features with group_by_prefix=16.                                                        |
| ext_quad                    | extended features with quadratic weight and weight_scale=6.0.                                   |
| ext_rank                    | extended features with ranking_loss_weight=0.2. Previous best setting.                            |
| ext_rank_01                 | Same as ext_rank but ranking loss weight is 0.1.                                                  |
| ext_rank_03                 | Same as ext_rank but ranking loss weight is 0.3.                                                  |
| ext_rank_ndcg1              | ext_rank with selection_metric=ndcg_1pct.                                                         |
| ext_rank_quad               | extended + quadratic weight + ranking loss.                                                       |
| exp_t02_ext24_linear_rank02 | extended, /24 split, linear weight, ranking_loss_weight=0.2. Similar to ext_rank.           |
| focus_quad_sel1             | extended, quadratic weight, selection_metric=ndcg_1pct, no ranking loss.                        |
| seed7_ext24_linear          | extended, /24 split, linear weight, seed 7, no ranking loss.                                  |
| seed7_ext24_linear_rank     | Same as above but with ranking loss (0.1).                                                          |
| seed7_ext24_quad            | Seed 7, quadratic weight, no ranking loss.                                                        |
| seed7_ext24_quad_rank       | Seed 7, quadratic weight, with ranking loss.                                                      |
| feature_groups_smoke        | Smoke test for feature switching. Uses only node_flow, node_degree, ip.                         |
| fs_drop_balance             | Feature reduction: removed flow_balance.                                                            |
| fs_drop_ip                  | Feature reduction: removed ip.                                                                      |
| fs_drop_ip_balance_degree   | Removed ip, flow_balance, neighbor_degree. Base of reduced_core.                              |
| reduced_core_rerun          | Re-run using reduced_core features.                                                                 |
| feature_ablation_local_loo  | Leave-one-out feature ablation. Each subfolder removes one feature group.                             |
| feature_ablation_c          | Only-one feature ablation. Each subfolder keeps only one feature group.                               |
| feature_seed_stability      | Compare full, drop_ip, reduced_core across multiple seeds.                                      |
| tune_top5_baseline          | Helper directory with only run.log.                                                                 |

## Meaning of Feature Reduction

### reduced_core

This setting removes these feature groups from extended.

- ip
- flow_balance
- neighbor_degree

Remaining feature groups:

- node_flow
- node_degree
- neighbor_flow
- neighbor_total_flow
- two_hop
- hub_ratio

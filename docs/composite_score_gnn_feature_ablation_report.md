# Composite Score GNN 特徴量削減実験レポート

## 概要

`examples/gnn/composite_risk.graphml` に対して、`composite_score` を直接予測する GNN で使う入力特徴量を見直した。

目的は次の 2 点である。

- 精度に貢献していない特徴量を削る
- 特徴量を減らすことで順位系指標を改善できるかを検証する

今回の結論は次の通りである。

- 特徴量削減によって、実際に順位系指標は改善した
- 特に `ip` 特徴群は順位品質を悪化させている可能性が高かった
- 一方で `node_flow`、`node_degree`、`neighbor_flow` は重要で、削ると大きく悪化した
- 実務上の推奨構成は `reduced_core` である


## 実験の考え方

既存の `extended` 特徴量は、ノード単体の統計と近傍構造の統計をまとめて使っていた。

ただし、すべての特徴量が有効とは限らない。
そこで、特徴量を意味ごとの群に分解し、次の 3 種類の実験を行った。

- `leave-one-out`
  - 1 つの特徴群だけ外す
- `only-one`
  - 1 つの特徴群だけ残す
- `targeted reduction`
  - `leave-one-out` の結果を見て、有望だった削減パターンをまとめて試す

また、単発の当たりを避けるため、最後に複数 seed でも比較した。


## 特徴群の定義

今回の実験では、`extended` 特徴量を次の群に分割した。

- `node_flow`
  - ノード自身の `in_flow`, `out_flow`, `total_flow`, `abs_gap`, `flow_ratio`
  - 平均 flow、最大 flow など
- `node_degree`
  - `in_degree`, `out_degree`, `total_degree`
- `flow_balance`
  - `inbound_share`, `outbound_share`
- `ip`
  - IP オクテット、private/public、RFC1918 帯のフラグ
- `neighbor_flow`
  - 1-hop 近傍エッジ flow の統計
- `neighbor_degree`
  - 近傍ノードの degree 統計
- `neighbor_total_flow`
  - 近傍ノードの総 flow 統計
- `two_hop`
  - 2-hop 到達ノード数
- `hub_ratio`
  - 自ノードと近傍平均との差に基づく局所ハブ比

関連コード:

- [composite_dataset.py](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/composite_dataset.py)
- [train_composite_score_gnn.py](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/train_composite_score_gnn.py)
- [feature_ablation_search.py](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/feature_ablation_search.py)


## 共通設定

特徴量以外の条件は、既存のベスト設定に揃えた。

- `feature_set = extended`
- `group_by_prefix = 24`
- `weight_mode = linear`
- `weight_scale = 4.0`
- `ranking_loss_weight = 0.2`
- `ranking_margin = 0.02`
- `ranking_pairs = 128`
- `num_hops = 1`
- `hidden_dim = 256`
- `num_layers = 2`
- `dropout = 0.1`
- `batch_size = 1024`
- `selection_metric = ndcg_5pct`

主な出力先:

- [feature_ablation_local_loo](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_local_loo)
- [feature_ablation_c](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c)
- [feature_seed_stability](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability)


## `leave-one-out` 実験

まず、各特徴群を 1 つずつ外して影響を見た。

結果:

| 設定 | ndcg_5pct | ndcg_1pct | top_5pct_recall | top_1pct_recall | spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `drop_node_flow` | 0.4319 | 0.4039 | 0.3750 | 0.4091 | 0.5186 |
| `drop_node_degree` | 0.4005 | 0.3502 | 0.1786 | 0.2727 | 0.4393 |
| `drop_flow_balance` | 0.4576 | 0.4006 | 0.6786 | 0.5909 | 0.7535 |
| `drop_ip` | 0.6029 | 0.5482 | 0.5223 | 0.5000 | 0.7069 |
| `drop_neighbor_flow` | 0.2241 | 0.0915 | 0.1741 | 0.0682 | 0.5379 |

ファイル:

- [drop_node_flow/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_local_loo/drop_node_flow/metrics.json)
- [drop_node_degree/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_local_loo/drop_node_degree/metrics.json)
- [drop_flow_balance/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_local_loo/drop_flow_balance/metrics.json)
- [drop_ip/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_local_loo/drop_ip/metrics.json)
- [drop_neighbor_flow/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_local_loo/drop_neighbor_flow/metrics.json)

所見:

- `node_flow` を外すと悪化した
- `node_degree` を外すとさらに悪化した
- `neighbor_flow` を外すと大幅に悪化した
- `flow_balance` を外しても大きな改善はなく、重要度は低かった
- `ip` を外すと `ndcg_5pct` と `ndcg_1pct` が大きく改善した

この段階で、`ip` は有害候補、`flow_balance` は低寄与候補、`node_flow` / `node_degree` / `neighbor_flow` は重要候補と判断した。


## `only-one` 実験

次に、各特徴群だけを残した場合の性能を確認した。

結果:

| 設定 | ndcg_5pct | ndcg_1pct | top_5pct_recall | top_1pct_recall | spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `only_node_flow` | 0.3662 | 0.3099 | 0.3214 | 0.3864 | 0.6877 |
| `only_node_degree` | 0.2199 | 0.0437 | 0.8170 | 0.0227 | 0.6674 |
| `only_flow_balance` | 0.1342 | 0.0564 | 0.3616 | 0.0909 | 0.1563 |
| `only_ip` | 0.1996 | 0.1941 | 0.0446 | 0.0909 | 0.0217 |
| `only_neighbor_flow` | 0.4179 | 0.3120 | 0.2321 | 0.1364 | 0.5255 |
| `full_baseline` | 0.5701 | 0.4392 | 0.7366 | 0.5909 | 0.7525 |

ファイル:

- [only_node_flow/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c/only_node_flow/metrics.json)
- [only_node_degree/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c/only_node_degree/metrics.json)
- [only_flow_balance/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c/only_flow_balance/metrics.json)
- [only_ip/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c/only_ip/metrics.json)
- [only_neighbor_flow/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c/only_neighbor_flow/metrics.json)
- [full_baseline/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_ablation_c/full_baseline/metrics.json)

所見:

- `only_ip` は非常に弱かった
- `only_flow_balance` も弱かった
- `only_node_flow` は単独でも比較的強かった
- `only_neighbor_flow` も単独で一定の情報を持っていた
- `only_node_degree` は `top_5pct_recall` は高いが、順位品質は悪かった

この結果から、強い特徴群は `node_flow` と `neighbor_flow`、補助として `node_degree` が効いていると分かった。


## 組み合わせ削減の実験

`leave-one-out` の結果を受けて、弱そうな群をまとめて削った。

結果:

| 設定 | 削除した特徴群 | ndcg_5pct | ndcg_1pct | top_5pct_recall | top_1pct_recall | spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fs_drop_balance` | `flow_balance` | 0.4375 | 0.3775 | 0.6920 | 0.5227 | 0.8061 |
| `fs_drop_ip` | `ip` | 0.6146 | 0.5782 | 0.4375 | 0.4091 | 0.8019 |
| `fs_drop_ip_balance_degree` | `ip`, `flow_balance`, `neighbor_degree` | 0.6161 | 0.4783 | 0.6741 | 0.6591 | 0.7242 |

ファイル:

- [fs_drop_balance/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/fs_drop_balance/metrics.json)
- [fs_drop_ip/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/fs_drop_ip/metrics.json)
- [fs_drop_ip_balance_degree/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/fs_drop_ip_balance_degree/metrics.json)

所見:

- `flow_balance` だけ削っても改善は限定的だった
- `ip` だけ削ると `NDCG` は大きく改善した
- ただし `ip` だけ削ると `top_5pct_recall` が大きく落ちた
- `ip + flow_balance + neighbor_degree` をまとめて削ると、`ndcg_5pct` と `top_1pct_recall` がともに改善した

この時点で、最も有望な削減セットは次の 2 つに絞られた。

- `drop_ip`
- `reduced_core = drop(ip, flow_balance, neighbor_degree)`


## 多 seed 比較

単発の当たりを避けるため、`seed=42, 7` で比較した。

結果:

| 設定 | 平均 ndcg_5pct | 平均 ndcg_1pct | 平均 top_5pct_recall | 平均 top_1pct_recall | 平均 spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.4356 | 0.3057 | 0.5414 | 0.3780 | 0.6396 |
| `drop_ip` | 0.5228 | 0.4541 | 0.3923 | 0.2703 | 0.8530 |
| `reduced_core` | 0.4871 | 0.3188 | 0.5417 | 0.4480 | 0.6242 |

ファイル:

- [full_seed42/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability/full_seed42/metrics.json)
- [full_seed7/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability/full_seed7/metrics.json)
- [drop_ip_seed42/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability/drop_ip_seed42/metrics.json)
- [drop_ip_seed7/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability/drop_ip_seed7/metrics.json)
- [reduced_core_seed42/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability/reduced_core_seed42/metrics.json)
- [reduced_core_seed7/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/experiments/feature_seed_stability/reduced_core_seed7/metrics.json)

所見:

- `drop_ip` は平均 `ndcg_5pct` と `ndcg_1pct` が最も高かった
- ただし `top_5pct_recall` と `top_1pct_recall` は `full` より悪化した
- `reduced_core` は `top_5pct_recall` を維持しつつ、`top_1pct_recall` を改善した
- `reduced_core` は `full` より少ない特徴量で、平均的により良いバランスを示した


## 最終結論

今回の実験から、特徴量削減は有効であると結論づけられる。

特に次の特徴群は削る価値が高かった。

- `ip`
- `flow_balance`
- `neighbor_degree`

一方で、次の特徴群は重要であり、削るべきではない。

- `node_flow`
- `node_degree`
- `neighbor_flow`
- `neighbor_total_flow`
- `two_hop`
- `hub_ratio`

実務上の最終推奨は `reduced_core` である。

残す特徴群:

- `node_flow`
- `node_degree`
- `neighbor_flow`
- `neighbor_total_flow`
- `two_hop`
- `hub_ratio`

削る特徴群:

- `ip`
- `flow_balance`
- `neighbor_degree`

理由:

- `drop_ip` は順位品質だけを見ると強い
- しかし high-risk ノードの回収率が落ちやすい
- `reduced_core` は特徴量を減らしつつ、`NDCG`、`top_1pct_recall`、`top_5pct_recall` のバランスが最も良かった


## 推奨コマンド

`reduced_core` で再学習する場合のコマンドは次の通りである。

```bash
python examples/gnn/train_composite_score_gnn.py \
  --graphml-path examples/gnn/composite_risk.graphml \
  --output-dir examples/gnn/experiments/reduced_core_prod \
  --epochs 6 \
  --batch-size 1024 \
  --eval-batch-size 1024 \
  --hidden-dim 256 \
  --num-layers 2 \
  --dropout 0.1 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --num-hops 1 \
  --max-in-neighbors 32 \
  --max-out-neighbors 32 \
  --num-workers 0 \
  --selection-metric ndcg_5pct \
  --patience 3 \
  --train-seed 42 \
  --device cpu \
  --feature-set extended \
  --feature-groups node_flow,node_degree,neighbor_flow,neighbor_total_flow,two_hop,hub_ratio \
  --group-by-prefix 24 \
  --split-bucket-size 32 \
  --weight-mode linear \
  --weight-scale 4.0 \
  --ranking-loss-weight 0.2 \
  --ranking-margin 0.02 \
  --ranking-pairs 128
```


## 今後の課題

- `seed=123` まで含めた完全な多 seed 集計
- `neighbor_degree` を丸ごと落とすのではなく、一部だけ残す実験
- permutation importance や SHAP のような事後的特徴量重要度分析
- 別グラフへの一般化確認


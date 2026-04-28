# Experiments Overview

このディレクトリには、`composite_score` 予測 GNN の各種実験結果を保存している。

各フォルダには通常、次のファイルが含まれる。

- `composite_score_gnn.pt`
- `composite_score_predictions.csv`
- `metrics.json`

補足:

- 新しい実験では `metrics.json` に `config` が保存されている
- 古い実験の一部には `config` が残っていない
- `config` がないものは、ディレクトリ名と過去の実験経緯から推定している


## 実験一覧

| フォルダ | 実験内容 |
| --- | --- |
| `output_prod_mps_fast` | 初期の本番向け学習 run。古い設定の基準モデル。`config` が残っていないため詳細は推定。 |
| `quick_1hop` | 初期の `1-hop` ベースライン。`config` なし。 |
| `quick_1hop_n32` | `1-hop` ベースライン。近傍数を `32` にした比較 run。 |
| `quick_1hop_n48` | `1-hop` ベースライン。近傍数を `48` にした比較 run。 |
| `quick_1hop_h512` | `1-hop` ベースライン。`hidden_dim=512` にした比較 run。 |
| `quick_1hop_h512_lr1e4` | `quick_1hop_h512` から学習率を `1e-4` に変えた run。 |
| `quick_1hop_top5` | `1-hop` 系の比較 run。名前から `top 5%` 重視の探索時 run と考えられるが、`config` なし。 |
| `search_e1` | 初期の探索 run の 1 つ。`config` がなく詳細不明。 |
| `ext_n32` | `extended` 特徴量を導入した基準 run。`/24 split`、`linear` weight、ranking loss なし。 |
| `ext_16` | `extended` 特徴量で `group_by_prefix=16` を試した比較 run。 |
| `ext_quad` | `extended` 特徴量で `weight_mode=quadratic`、`weight_scale=6.0` を試した run。 |
| `ext_rank` | `extended` 特徴量に `ranking_loss_weight=0.2` を加えた run。以前のベスト設定。 |
| `ext_rank_01` | `ext_rank` の ranking loss 重みを `0.1` に下げた比較 run。 |
| `ext_rank_03` | `ext_rank` の ranking loss 重みを `0.3` に上げた比較 run。 |
| `ext_rank_ndcg1` | `ext_rank` 系で `selection_metric=ndcg_1pct` を使った run。 |
| `ext_rank_quad` | `extended` + `quadratic` weight + ranking loss の組み合わせを試した run。 |
| `exp_t02_ext24_linear_rank02` | `extended`、`/24 split`、`linear` weight、`ranking_loss_weight=0.2` の run。実質 `ext_rank` と同系統。 |
| `focus_quad_sel1` | `extended`、`quadratic` weight、`selection_metric=ndcg_1pct` を試した run。ranking loss なし。 |
| `seed7_ext24_linear` | `extended`、`/24 split`、`linear` weight、seed を `7` にした run。ranking loss なし。 |
| `seed7_ext24_linear_rank` | `seed7_ext24_linear` に ranking loss を追加した run。`ranking_loss_weight=0.1`。 |
| `seed7_ext24_quad` | seed を `7` にして `quadratic` weight を試した run。ranking loss なし。 |
| `seed7_ext24_quad_rank` | seed `7`、`quadratic` weight、ranking loss ありの run。 |
| `feature_groups_smoke` | 特徴量切替のスモークテスト。`node_flow,node_degree,ip` のみ使用。小さい設定で動作確認。 |
| `fs_drop_balance` | `flow_balance` 特徴群を削除した特徴量削減 run。 |
| `fs_drop_ip` | `ip` 特徴群を削除した特徴量削減 run。 |
| `fs_drop_ip_balance_degree` | `ip`、`flow_balance`、`neighbor_degree` を削除した特徴量削減 run。`reduced_core` の元になった run。 |
| `reduced_core_rerun` | `reduced_core` 特徴量で再学習した評価 run。使用特徴群は `node_flow,node_degree,neighbor_flow,neighbor_total_flow,two_hop,hub_ratio`。 |
| `feature_ablation_local_loo` | `leave-one-out` 特徴量アブレーション結果。各サブフォルダに「1群だけ削除した」run を保存。 |
| `feature_ablation_c` | `only-one` 特徴量アブレーション結果。各サブフォルダに「1群だけ残した」run を保存。 |
| `feature_seed_stability` | `full`、`drop_ip`、`reduced_core` を複数 seed で比較した run。再現性確認用。 |
| `tune_top5_baseline` | `run.log` のみがある補助ディレクトリ。`top_5` 系の探索時ログ。 |


## 特徴量削減系の意味

### `reduced_core`

`extended` 特徴量から、次の 3 群を削除した構成である。

- `ip`
- `flow_balance`
- `neighbor_degree`

残す特徴群は次の通り。

- `node_flow`
- `node_degree`
- `neighbor_flow`
- `neighbor_total_flow`
- `two_hop`
- `hub_ratio`


## 参照ドキュメント

- [composite_score_gnn_experiment_report.md](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/docs/composite_score_gnn_experiment_report.md)
- [composite_score_gnn_feature_ablation_report.md](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/docs/composite_score_gnn_feature_ablation_report.md)

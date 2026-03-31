# Composite Score GNN 実験レポート

## 概要

`examples/gnn/composite_risk.graphml` を用いて、ノードごとの `composite_score` を直接予測する GNN の改善実験を行った。

本タスクの主目的は、回帰誤差を最小化することではなく、`高リスクノードを上位に正しく並べること` である。  
そのため、評価では `top_k_recall` と `NDCG` を重視した。

今回の結論は次の通りである。

- 最も効いた改善は `extended` 特徴量の追加だった
- その次に効いた改善は `pairwise ranking loss` の追加だった
- split 粒度は `/24` が良く、`/16` は validation が良く見えても test で崩れた
- 最終的なベスト設定は `extended features + /24 split + linear weights + ranking loss` だった


## 実装した改善

今回の実験では、以下をコードに追加して比較した。

- `extended` 特徴量
  - 近傍の `in_flow` / `out_flow` の合計、平均、最大、上位四分位
  - 近傍ノードの総 flow / 総 degree の統計
  - 2-hop 近傍数
  - 近傍平均との差に基づく局所ハブ特徴
- split 粒度の切替
  - `/24`
  - `/16`
- サンプル重み付け
  - `linear`
  - `quadratic`
- pairwise ranking loss
  - 回帰損失に加えて、真の高スコアノードが低スコアノードより上に来るように学習

関連コード:

- [composite_dataset.py](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/composite_dataset.py)
- [train_composite_score_gnn.py](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/train_composite_score_gnn.py)


## 比較した主要設定

### 1. 元の本番設定

ファイル:

- [output_prod_mps_fast/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/output_prod_mps_fast/metrics.json)

結果:

- `log_mae = 0.0576`
- `spearman = -0.2112`
- `top_1pct_recall = 0.0000`
- `top_5pct_recall = 0.0223`
- `ndcg_1pct = 0.0148`
- `ndcg_5pct = 0.0440`

所見:

- 高リスクノードの順位付けにほぼ失敗していた
- 実用水準ではなかった


### 2. 1-hop ベースライン

ファイル:

- [quick_1hop_n32/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/quick_1hop_n32/metrics.json)

結果:

- `log_mae = 0.0466`
- `spearman = 0.3284`
- `top_1pct_recall = 0.0000`
- `top_5pct_recall = 0.1339`
- `ndcg_1pct = 0.0200`
- `ndcg_5pct = 0.1071`

所見:

- 2-hop より `1-hop` の方が安定して良かった
- ただし最上位 1% はまだ全く拾えていなかった


### 3. `extended` 特徴量のみ追加

ファイル:

- [ext_n32/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/ext_n32/metrics.json)

結果:

- `log_mae = 0.0413`
- `spearman = 0.3956`
- `top_1pct_recall = 0.2273`
- `top_5pct_recall = 0.5446`
- `ndcg_1pct = 0.1118`
- `ndcg_5pct = 0.3065`

所見:

- 一番大きな改善要因だった
- 特徴量追加だけで、上位ノード検出性能が大幅に改善した
- `top_1pct_recall` が 0 から 0.227 まで上がった


### 4. `/16` split の比較

ファイル:

- [ext_16/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/ext_16/metrics.json)

結果:

- `log_mae = 0.0144`
- `spearman = -0.1654`
- `top_1pct_recall = 0.0000`
- `top_5pct_recall = 0.0483`
- `ndcg_1pct = 0.0104`
- `ndcg_5pct = 0.0381`

所見:

- validation では良く見えても、test では大きく崩れた
- `/16` は group が粗すぎて、一般化評価として不安定だった
- 今回のデータでは `/24` の方が妥当だった


### 5. `ranking loss` を加えた最良設定

ファイル:

- [ext_rank/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/ext_rank/metrics.json)

結果:

- `log_mae = 0.0808`
- `spearman = 0.7683`
- `top_1pct_recall = 0.4091`
- `top_5pct_recall = 0.7366`
- `ndcg_1pct = 0.2433`
- `ndcg_5pct = 0.4494`

所見:

- 今回の目的である `高リスクノードを上位に出す` という観点では最良
- `top_1pct_recall` と `top_5pct_recall` が大きく改善した
- 一方で `log_mae` は悪化しており、値そのものの回帰精度より順位最適化に寄った設定である


### 6. `selection_metric=ndcg_1pct` の比較

ファイル:

- [ext_rank_ndcg1/metrics.json](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/ext_rank_ndcg1/metrics.json)

結果:

- `log_mae = 0.0797`
- `spearman = 0.7868`
- `top_1pct_recall = 0.3409`
- `top_5pct_recall = 0.7500`
- `ndcg_1pct = 0.1503`
- `ndcg_5pct = 0.3690`

所見:

- 全体順位相関は高かった
- ただし `ndcg_1pct` と `ndcg_5pct` が `ext_rank` より悪く、今回の主目的では次点と判断した


## 主要比較表

| 設定 | log_mae | spearman | top_1pct_recall | top_5pct_recall | ndcg_1pct | ndcg_5pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 元の本番設定 | 0.0576 | -0.2112 | 0.0000 | 0.0223 | 0.0148 | 0.0440 |
| 1-hop ベースライン | 0.0466 | 0.3284 | 0.0000 | 0.1339 | 0.0200 | 0.1071 |
| `extended` のみ | 0.0413 | 0.3956 | 0.2273 | 0.5446 | 0.1118 | 0.3065 |
| `/16` split | 0.0144 | -0.1654 | 0.0000 | 0.0483 | 0.0104 | 0.0381 |
| 最良設定 `ext_rank` | 0.0808 | 0.7683 | 0.4091 | 0.7366 | 0.2433 | 0.4494 |
| `ndcg_1pct` 選択 | 0.0797 | 0.7868 | 0.3409 | 0.7500 | 0.1503 | 0.3690 |


## 最終的に最も良かった設定

最良設定は [ext_rank](/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/examples/gnn/ext_rank) である。

設定は以下の通り。

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

実行コマンド:

```bash
python examples/gnn/train_composite_score_gnn.py \
  --graphml-path examples/gnn/composite_risk.graphml \
  --output-dir examples/gnn/ext_rank \
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
  --num-workers 4 \
  --prefetch-factor 2 \
  --selection-metric ndcg_5pct \
  --patience 3 \
  --train-seed 42 \
  --device mps \
  --feature-set extended \
  --group-by-prefix 24 \
  --weight-mode linear \
  --weight-scale 4.0 \
  --ranking-loss-weight 0.2 \
  --ranking-margin 0.02 \
  --ranking-pairs 128
```


## 指標の解釈

- `top_1pct_recall`
  - 真の上位 1% ノードのうち、予測でも上位 1% に入れられた割合
- `top_5pct_recall`
  - 真の上位 5% ノードのうち、予測でも上位 5% に入れられた割合
- `ndcg_1pct`
  - 上位 1% 領域で、重要ノードをどれだけ良い順に並べられたか
- `ndcg_5pct`
  - 上位 5% 領域での順位品質
- `spearman`
  - 全ノードの順位相関
- `log_mae`
  - `log1p(composite_score)` 回帰値の絶対誤差

このタスクでは、回帰誤差よりも `top_1pct_recall`、`top_5pct_recall`、`ndcg_5pct` を重視するべきである。  
理由は、目的が `危険ノードを見つけること` だからである。


## 最終評価

今回の改善により、モデルは `高リスクノード候補を絞り込むモデル` としてはかなり有望になった。

特に最良設定では、

- 真の上位 1% の約 40.9% を回収
- 真の上位 5% の約 73.7% を回収
- 上位 5% 領域の順位品質も大きく改善

という結果になった。

一方で、まだ次の限界はある。

- 最上位 1% の取りこぼしはまだ残る
- スコア値そのものの回帰精度は最適化していない
- この結果は 1 つの GraphML データセット上のものであり、他期間・他ネットワークへの一般化は未検証

したがって、現時点では次の判断が妥当である。

- `高リスクノードの優先度付け` 用途には十分使える可能性が高い
- `本番投入前` には、seed を変えた再現性確認と別データでの外部検証が必要


## 今後の推奨

- 複数 seed で再学習し、結果の分散を確認する
- 別期間・別ネットワークのグラフで外部検証する
- `node downtime impact` により近い教師ラベルが作れるなら、`composite_score` ではなくそのラベルを直接学習する
- 上位ノードに対する説明性を補うため、予測上位ノードの近傍特徴を可視化する

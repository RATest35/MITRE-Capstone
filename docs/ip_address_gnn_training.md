# IP Address GNN Training Script Documentation

このドキュメントは `examples/ip-address/train_gnn.py` の構造と処理内容を説明する。

## 1. このスクリプトの目的

このスクリプトは、GraphML 形式の IP 通信グラフを読み込み、各ノードの `flow_loss` を予測する GNN を学習する。

入力データの前提は以下の通り。

- ノードは IP アドレス
- エッジは通信の向き
- エッジ属性 `flow` は通信量
- ノード属性 `flow_loss` は、そのノードに関連する流量損失の指標

このスクリプトでは `flow_loss` をそのまま予測するのではなく、`log1p(flow_loss)` を予測する回帰問題として扱う。
値のスケールが大きく偏っているため、対数変換した方が学習が安定しやすいからである。

## 2. コード全体の構造

スクリプトは大きく 6 つの要素で構成されている。

1. 定数定義
2. GNN モデル定義
3. GraphML から学習データを作る処理
4. train/validation/test 分割処理
5. 学習と評価処理
6. 予測結果の保存とメイン実行処理

処理の流れは次の通り。

1. GraphML を読み込む
2. ノードごとの特徴量を作る
3. PyTorch Geometric の `Data` に変換する
4. ノードを train/validation/test に分割する
5. GraphSAGE モデルを学習する
6. テストノードに対する予測精度を計算する
7. 予測結果を CSV に保存する

## 3. 先頭の定数が表しているもの

スクリプトの先頭には、実行設定をまとめた定数が置かれている。

- `GRAPHML_PATH`
  - 学習対象の GraphML ファイルのパス
- `PREDICTION_CSV_PATH`
  - 予測結果を保存する CSV の出力先
- `RANDOM_SEEDS`
  - 複数回学習するための乱数シード一覧
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
  - ノード分割比率
- `HIDDEN_CHANNELS`
  - GNN の隠れ層次元
- `NUM_EPOCHS`
  - 最大エポック数
- `LEARNING_RATE`
  - 学習率
- `WEIGHT_DECAY`
  - L2 正則化の強さ
- `PATIENCE`
  - validation 損失が改善しない状態を何回まで許容するか
- `DROPOUT`
  - 中間層で使う dropout の割合

この構成にしているため、CLI 引数を追加しなくても、コード先頭だけ見れば実験条件を把握できる。

## 4. モデル部分

### `FlowLossGNN`

このクラスはノードごとの `flow_loss` を予測する GNN 本体である。

構成は次の通り。

- `SAGEConv` 2 層
- `ReLU` 活性化
- 1 層目の後に `dropout`
- 最後に全結合層 `Linear`

役割は単純で、ノード特徴量とグラフ接続関係から各ノード 1 つの回帰値を返す。

`forward()` の流れは以下の通り。

1. 1 層目の GraphSAGE で近傍情報を集約する
2. ReLU を通す
3. dropout をかける
4. 2 層目の GraphSAGE を通す
5. もう一度 ReLU を通す
6. `Linear` で 1 次元の予測値に変換する

## 5. データ作成部分

### `build_data()`

この関数は GraphML を読み込み、PyTorch Geometric の `Data` に変換する。

### 入力

- GraphML ファイルのパス

### 出力

- `Data`
- ノード ID のリスト

ノード ID のリストを一緒に返しているのは、あとで予測結果を CSV に保存するときに、数値インデックスではなく元の IP アドレスで出力するためである。

### この関数が作るノード特徴量

各ノードに対して以下の特徴量を作っている。

- IP アドレスの 4 オクテット
- `in_degree`
- `out_degree`
- `degree`
- 入力流量合計
- 出力流量合計
- 総流量
- 平均入力流量
- 平均出力流量

ここで重要なのは、予測対象である `flow_loss` を入力特徴量に入れていないことだ。
もし `flow_loss` を特徴量に入れると、モデルは答えを見ながら学習することになり、評価が無意味になる。

### ラベル

ラベルには各ノードの `flow_loss` を使うが、そのままではなく `math.log1p()` を通して保存している。

### エッジ

GraphML の有向エッジをそのまま `edge_index` に変換している。
つまり、このモデルは元の通信方向を保ったまま学習する。

## 6. ノード分割と標準化

### `build_masks()`

この関数はノードを train、validation、test に分けるための boolean mask を作る。

処理は単純で、乱数シードを固定してノード順をシャッフルし、指定比率で先頭から切り分ける。

出力は以下の 3 つである。

- `train_mask`
- `val_mask`
- `test_mask`

このスクリプトは単一グラフを扱う transductive learning の形になっている。
つまり、グラフ全体の接続は使うが、損失計算に使うノードだけを mask で切り替える。

### `standardize_features()`

この関数はノード特徴量を標準化する。

標準化に使う平均と標準偏差は train ノードだけから計算している。
これは test ノードの情報が前処理に漏れないようにするためである。

処理内容は以下の通り。

- train ノードの平均を計算する
- train ノードの標準偏差を計算する
- 全ノード特徴量に対して `(x - mean) / std` を適用する

`std` が 0 になると除算できないため、最小値を `1e-6` に固定している。

## 7. 学習と評価

### `train_and_evaluate()`

この関数は 1 つの乱数シードに対して、学習から評価までをまとめて実行する。

### 前半の処理

- シード固定
- `base_data` のコピー作成
- mask の付与
- 特徴量の標準化
- モデル生成
- Optimizer 生成

### 学習ループ

学習ループでは以下を繰り返す。

1. train モードで順伝播する
2. train ノードだけで MSE 損失を計算する
3. 逆伝播してパラメータを更新する
4. eval モードに切り替える
5. validation ノードだけで損失を計算する
6. validation 損失が改善したらベストモデルを保存する

### Early Stopping

`PATIENCE` 回連続で validation 損失が改善しなければ学習を打ち切る。
これにより、小さなデータセットで過学習し続けるのを防ぐ。

### テスト評価

学習後は、最良だったモデル状態を読み戻して test ノードに対する予測を行う。

計算している評価指標は以下の 4 つである。

- `mae`
- `rmse`
- `log_mae`
- `log_rmse`

意味は次の通り。

- `mae`, `rmse`
  - 元の `flow_loss` スケールでの誤差
- `log_mae`, `log_rmse`
  - 対数変換後スケールでの誤差

元スケールの誤差を出すときは、`torch.expm1()` で予測値と正解値を元に戻している。

### 予測結果の保持

test ノードについて、以下の情報を辞書として保存している。

- `seed`
- `node_id`
- `actual_flow_loss`
- `predicted_flow_loss`

これが最後に CSV に書き出される。

## 8. 出力保存

### `write_predictions()`

この関数は予測結果の辞書リストを CSV に保存する。

出力される列は以下の 4 つである。

- `seed`
- `node_id`
- `actual_flow_loss`
- `predicted_flow_loss`

この CSV を使うと、どの IP アドレスで誤差が大きかったかをあとで確認できる。

## 9. 実行全体をまとめる `main()`

`main()` は全体の制御を行う。

処理の順番は以下の通り。

1. GraphML を読み込んで `Data` を作る
2. `RANDOM_SEEDS` の各シードで `train_and_evaluate()` を実行する
3. 各シードの評価指標を表示する
4. すべての予測結果を CSV に保存する
5. 平均 MAE、平均 RMSE、平均 log MAE、平均 log RMSE を表示する

つまり、1 回の実行で単発の結果ではなく、複数シード平均のベースラインを確認できる。

## 10. このスクリプトの設計上の意図

この実装は、まず動くベースラインを作ることを重視している。

そのため、あえて次のような設計にしている。

- モデルは 2 層の軽量 GraphSAGE に限定
- 複雑な特徴量エンジニアリングは避ける
- エッジ属性 `flow` は message passing に直接渡さず、ノード集計特徴量として使う
- 実験設定はコード先頭の定数に集約
- 関数数を増やしすぎず、追いやすい構造にする

## 11. 改善するとしたらどこか

今後の改善候補は次の通り。

- `flow` を edge attribute として直接使う GNN に変える
- `GCNConv` や `GATConv` と比較する
- 特徴量に subnet 単位の情報を追加する
- クロスバリデーション方法を見直す
- 予測値と実測値の散布図を追加する

ただし、現状のスクリプトはあくまで「単一グラフで `flow_loss` を予測する最小限の GNN ベースライン」である。

## 12. 実行方法

依存関係をインストールした後、以下で実行する。

```bash
python examples/ip-address/train_gnn.py
```

実行後は、各シードの評価指標が標準出力に表示され、予測結果が `examples/ip-address/gnn_predictions.csv` に保存される。

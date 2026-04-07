gnn/dataset/composite_risk.graphml のcomposite scoreを予測するモデルを作りたい。

composite score は pagerank と weighted_betweenness から計算された値です。これらは使用せずに、他の特徴量から予測するようにして。
また、importance も composite scoreと同じなので、使ってはいけません。

モデルはgnn/model.pyに定義されています。まず、モデルの特徴量inputとして何を使用すべきかを、データセットを分析して教えて。
なるべく最小限の特徴量で性能が出るようにしてほしい。

node, edge, また、選択されたnodeの周辺ノードの情報も使用するようにしてください。

まだコードを書かずに教えて。

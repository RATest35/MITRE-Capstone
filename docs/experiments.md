python3 gnn/train.py \
    --graphml-path gnn/dataset/composite_risk.graphml \
    --output-path gnn/composite_score_gnn.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 30 \
    --patience 5 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 32 \
    --max-out-neighbors 32 \
    --seed 42
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py
epoch=1 train_loss=0.016412 val_loss=0.000275 p@5=0.0240 ndcg@5=0.0795
epoch=2 train_loss=0.002112 val_loss=0.000135 p@5=0.2994 ndcg@5=0.1615
epoch=3 train_loss=0.001142 val_loss=0.000101 p@5=0.3293 ndcg@5=0.1406
epoch=4 train_loss=0.000727 val_loss=0.000048 p@5=0.3772 ndcg@5=0.1733
epoch=5 train_loss=0.000497 val_loss=0.000040 p@5=0.2455 ndcg@5=0.1236
epoch=6 train_loss=0.000352 val_loss=0.000030 p@5=0.1796 ndcg@5=0.1103
epoch=7 train_loss=0.000270 val_loss=0.000012 p@5=0.3054 ndcg@5=0.1561
epoch=8 train_loss=0.000214 val_loss=0.000011 p@5=0.1976 ndcg@5=0.1201
epoch=9 train_loss=0.000171 val_loss=0.000011 p@5=0.2275 ndcg@5=0.1303
early_stop_epoch=9
test_loss=0.000049
test_p@5=0.3719 test_ndcg@5=0.2394
saved=gnn/composite_score_gnn.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --graphml-path gnn/dataset/composite_risk.graphml \
    --output-path gnn/composite_score_gnn.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 30 \
    --patience 5 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 2 \
    --max-in-neighbors 32 \
    --max-out-neighbors 32 \
    --seed 42
epoch=1 train_loss=0.018158 val_loss=0.000319 p@5=0.0240 ndcg@5=0.1154
epoch=2 train_loss=0.002159 val_loss=0.000236 p@5=0.1856 ndcg@5=0.1108
epoch=3 train_loss=0.001137 val_loss=0.000084 p@5=0.2814 ndcg@5=0.1341
epoch=4 train_loss=0.000713 val_loss=0.000056 p@5=0.2515 ndcg@5=0.1370
epoch=5 train_loss=0.000494 val_loss=0.000044 p@5=0.1437 ndcg@5=0.1122
epoch=6 train_loss=0.000362 val_loss=0.000020 p@5=0.1437 ndcg@5=0.1234
epoch=7 train_loss=0.000274 val_loss=0.000013 p@5=0.1257 ndcg@5=0.1145
epoch=8 train_loss=0.000218 val_loss=0.000008 p@5=0.1317 ndcg@5=0.1083
early_stop_epoch=8
test_loss=0.000089
test_p@5=0.3266 test_ndcg@5=0.2310
saved=gnn/composite_score_gnn.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --graphml-path gnn/dataset/composite_risk.graphml \
    --output-path gnn/composite_score_gnn.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 30 \
    --patience 5 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16\ 
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.016413 val_loss=0.000262 p@5=0.0240 ndcg@5=0.0793
epoch=2 train_loss=0.002107 val_loss=0.000132 p@5=0.3114 ndcg@5=0.1681
epoch=3 train_loss=0.001136 val_loss=0.000103 p@5=0.3234 ndcg@5=0.1379
epoch=4 train_loss=0.000729 val_loss=0.000048 p@5=0.3772 ndcg@5=0.1744
epoch=5 train_loss=0.000500 val_loss=0.000042 p@5=0.2335 ndcg@5=0.1218
epoch=6 train_loss=0.000354 val_loss=0.000029 p@5=0.1617 ndcg@5=0.1058
epoch=7 train_loss=0.000270 val_loss=0.000012 p@5=0.2754 ndcg@5=0.1508
epoch=8 train_loss=0.000214 val_loss=0.000010 p@5=0.2156 ndcg@5=0.1278
epoch=9 train_loss=0.000172 val_loss=0.000009 p@5=0.2096 ndcg@5=0.1237
early_stop_epoch=9
test_loss=0.000049
test_p@5=0.3769 test_ndcg@5=0.2466
saved=gnn/composite_score_gnn.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --graphml-path gnn/dataset/composite_risk.graphml \
    --output-path gnn/composite_score_gnn.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 30 \
    --patience 10 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16\
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.016413 val_loss=0.000262 p@5=0.0240 ndcg@5=0.0793
epoch=2 train_loss=0.002107 val_loss=0.000132 p@5=0.3114 ndcg@5=0.1681
epoch=3 train_loss=0.001136 val_loss=0.000103 p@5=0.3234 ndcg@5=0.1379
epoch=4 train_loss=0.000729 val_loss=0.000048 p@5=0.3772 ndcg@5=0.1744
epoch=5 train_loss=0.000500 val_loss=0.000042 p@5=0.2335 ndcg@5=0.1218
epoch=6 train_loss=0.000354 val_loss=0.000029 p@5=0.1617 ndcg@5=0.1058
epoch=7 train_loss=0.000270 val_loss=0.000012 p@5=0.2754 ndcg@5=0.1508
epoch=8 train_loss=0.000214 val_loss=0.000010 p@5=0.2156 ndcg@5=0.1273
epoch=9 train_loss=0.000172 val_loss=0.000009 p@5=0.2096 ndcg@5=0.1237
epoch=10 train_loss=0.000142 val_loss=0.000007 p@5=0.1018 ndcg@5=0.0856
epoch=11 train_loss=0.000123 val_loss=0.000007 p@5=0.0838 ndcg@5=0.0793
epoch=12 train_loss=0.000109 val_loss=0.000006 p@5=0.0838 ndcg@5=0.0769
epoch=13 train_loss=0.000094 val_loss=0.000004 p@5=0.2156 ndcg@5=0.1141
epoch=14 train_loss=0.000086 val_loss=0.000003 p@5=0.1737 ndcg@5=0.0997
early_stop_epoch=14
test_loss=0.000049
test_p@5=0.3769 test_ndcg@5=0.2466
saved=gnn/composite_score_gnn.pt


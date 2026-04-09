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
    --output-path build/exp_logs/exp_a.pt \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.0 \
    --epochs 8 \
    --patience 3 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.001442 val_loss=0.000055 p@5=0.1257 ndcg@5=0.0887
epoch=2 train_loss=0.000044 val_loss=0.000024 p@5=0.0659 ndcg@5=0.0780
epoch=3 train_loss=0.000025 val_loss=0.000017 p@5=0.0359 ndcg@5=0.0719
epoch=4 train_loss=0.000017 val_loss=0.000012 p@5=0.0898 ndcg@5=0.0846
early_stop_epoch=4
test_loss=0.000067
test_p@5=0.1508 test_ndcg@5=0.1559
saved=build/exp_logs/exp_a.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_b.pt \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.2 \
    --epochs 8 \
    --patience 3 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.009305 val_loss=0.000093 p@5=0.0299 ndcg@5=0.0853
epoch=2 train_loss=0.001173 val_loss=0.000050 p@5=0.0240 ndcg@5=0.0791
epoch=3 train_loss=0.000401 val_loss=0.000014 p@5=0.1138 ndcg@5=0.0763
epoch=4 train_loss=0.000191 val_loss=0.000011 p@5=0.1257 ndcg@5=0.0884
epoch=5 train_loss=0.000114 val_loss=0.000007 p@5=0.0599 ndcg@5=0.0894
epoch=6 train_loss=0.000065 val_loss=0.000004 p@5=0.1557 ndcg@5=0.0945
epoch=7 train_loss=0.000045 val_loss=0.000002 p@5=0.2395 ndcg@5=0.1086
epoch=8 train_loss=0.000031 val_loss=0.000001 p@5=0.2515 ndcg@5=0.1087
test_loss=0.000001
test_p@5=0.0955 test_ndcg@5=0.1575
saved=build/exp_logs/exp_b.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_c.pt \
    --hidden-dim 64 \
    --num-layers 3 \
    --dropout 0.1 \
    --epochs 8 \
    --patience 3 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.007267 val_loss=0.000222 p@5=0.0539 ndcg@5=0.0787
epoch=2 train_loss=0.001495 val_loss=0.000083 p@5=0.0120 ndcg@5=0.0618
epoch=3 train_loss=0.000654 val_loss=0.000043 p@5=0.0299 ndcg@5=0.0647
epoch=4 train_loss=0.000379 val_loss=0.000026 p@5=0.0180 ndcg@5=0.0624
early_stop_epoch=4
test_loss=0.000228
test_p@5=0.0402 test_ndcg@5=0.1333
saved=build/exp_logs/exp_c.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_d.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 8 \
    --patience 3 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-hops 1 \
    --max-in-neighbors 8 \
    --max-out-neighbors 8 \
    --seed 42
epoch=1 train_loss=0.012551 val_loss=0.000199 p@5=0.1257 ndcg@5=0.1030
epoch=2 train_loss=0.001315 val_loss=0.000103 p@5=0.3952 ndcg@5=0.2746
epoch=3 train_loss=0.000719 val_loss=0.000043 p@5=0.3114 ndcg@5=0.1621
epoch=4 train_loss=0.000433 val_loss=0.000033 p@5=0.1317 ndcg@5=0.0987
epoch=5 train_loss=0.000296 val_loss=0.000023 p@5=0.2275 ndcg@5=0.1244
early_stop_epoch=5
test_loss=0.000105
test_p@5=0.3216 test_ndcg@5=0.2829
saved=build/exp_logs/exp_d.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_e.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 12 \
    --patience 4 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 8 \
    --max-out-neighbors 8 \
    --seed 42
epoch=1 train_loss=0.016417 val_loss=0.000277 p@5=0.0240 ndcg@5=0.0792
epoch=2 train_loss=0.002111 val_loss=0.000130 p@5=0.2994 ndcg@5=0.1614
epoch=3 train_loss=0.001140 val_loss=0.000103 p@5=0.3473 ndcg@5=0.2089
epoch=4 train_loss=0.000728 val_loss=0.000048 p@5=0.3772 ndcg@5=0.1752
epoch=5 train_loss=0.000500 val_loss=0.000041 p@5=0.2575 ndcg@5=0.1251
epoch=6 train_loss=0.000354 val_loss=0.000031 p@5=0.1916 ndcg@5=0.1126
epoch=7 train_loss=0.000269 val_loss=0.000012 p@5=0.2874 ndcg@5=0.1584
epoch=8 train_loss=0.000214 val_loss=0.000010 p@5=0.1976 ndcg@5=0.1209
early_stop_epoch=8
test_loss=0.000050
test_p@5=0.3668 test_ndcg@5=0.2415
saved=build/exp_logs/exp_e.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_f.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 12 \
    --patience 4 \
    --batch-size 128 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.027947 val_loss=0.000519 p@5=0.0539 ndcg@5=0.1426
epoch=2 train_loss=0.003807 val_loss=0.000282 p@5=0.1198 ndcg@5=0.1196
epoch=3 train_loss=0.002401 val_loss=0.000211 p@5=0.2635 ndcg@5=0.1847
epoch=4 train_loss=0.001626 val_loss=0.000127 p@5=0.3533 ndcg@5=0.2491
epoch=5 train_loss=0.001167 val_loss=0.000099 p@5=0.3114 ndcg@5=0.2322
epoch=6 train_loss=0.000897 val_loss=0.000059 p@5=0.1856 ndcg@5=0.1134
epoch=7 train_loss=0.000708 val_loss=0.000047 p@5=0.1976 ndcg@5=0.1174
epoch=8 train_loss=0.000564 val_loss=0.000045 p@5=0.1198 ndcg@5=0.0942
early_stop_epoch=8
test_loss=0.000127
test_p@5=0.2362 test_ndcg@5=0.2571
saved=build/exp_logs/exp_f.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_g.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --epochs 12 \
    --patience 4 \
    --batch-size 128 \
    --lr 2e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.016443 val_loss=0.000299 p@5=0.0359 ndcg@5=0.0668
epoch=2 train_loss=0.001968 val_loss=0.000168 p@5=0.4012 ndcg@5=0.2846
epoch=3 train_loss=0.001125 val_loss=0.000075 p@5=0.2754 ndcg@5=0.2085
epoch=4 train_loss=0.000703 val_loss=0.000038 p@5=0.1377 ndcg@5=0.1006
epoch=5 train_loss=0.000480 val_loss=0.000045 p@5=0.1677 ndcg@5=0.1074
epoch=6 train_loss=0.000368 val_loss=0.000042 p@5=0.1677 ndcg@5=0.1037
early_stop_epoch=6
test_loss=0.000161
test_p@5=0.2714 test_ndcg@5=0.2742
saved=build/exp_logs/exp_g.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 gnn/train.py \
    --output-path build/exp_logs/exp_h.pt \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.0 \
    --epochs 12 \
    --patience 4 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --seed 42
epoch=1 train_loss=0.011866 val_loss=0.000135 p@5=0.0539 ndcg@5=0.1342
epoch=2 train_loss=0.000123 val_loss=0.000055 p@5=0.1677 ndcg@5=0.1739
epoch=3 train_loss=0.000062 val_loss=0.000032 p@5=0.2695 ndcg@5=0.2772
epoch=4 train_loss=0.000038 val_loss=0.000022 p@5=0.2814 ndcg@5=0.2611
epoch=5 train_loss=0.000026 val_loss=0.000017 p@5=0.3234 ndcg@5=0.2716
epoch=6 train_loss=0.000020 val_loss=0.000013 p@5=0.2934 ndcg@5=0.2544
epoch=7 train_loss=0.000016 val_loss=0.000012 p@5=0.3473 ndcg@5=0.2735
epoch=8 train_loss=0.000014 val_loss=0.000009 p@5=0.3653 ndcg@5=0.2735
epoch=9 train_loss=0.000011 val_loss=0.000009 p@5=0.4012 ndcg@5=0.2840
epoch=10 train_loss=0.000010 val_loss=0.000007 p@5=0.2395 ndcg@5=0.2223
epoch=11 train_loss=0.000009 val_loss=0.000006 p@5=0.2395 ndcg@5=0.2233
epoch=12 train_loss=0.000008 val_loss=0.000006 p@5=0.2575 ndcg@5=0.2353
test_loss=0.000009
test_p@5=0.2312 test_ndcg@5=0.2287
saved=build/exp_logs/exp_h.pt
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

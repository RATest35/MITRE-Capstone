Structure: PNAConv


---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/pna_base.pt \
    --hidden-dim \
    64 \
    --num-layers \
    2 \
    --dropout \
    0.1 \
    --epochs \
    10 \
    --patience \
    4 \
    --batch-size \
    64 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    32 \
    --max-out-neighbors \
    32 \
    --seed \
    42
epoch=1 train_loss=0.002053 val_loss=0.000040 p@5=0.2036 ndcg@5=0.0903
epoch=2 train_loss=0.000419 val_loss=0.000077 p@5=0.0240 ndcg@5=0.0718
epoch=3 train_loss=0.000232 val_loss=0.000020 p@5=0.0719 ndcg@5=0.0734
epoch=4 train_loss=0.000151 val_loss=0.000030 p@5=0.0719 ndcg@5=0.0759
epoch=5 train_loss=0.000107 val_loss=0.000012 p@5=0.0180 ndcg@5=0.0733
early_stop_epoch=5
test_loss=0.000047
test_p@5=0.1307 test_ndcg@5=0.1510
saved=pnaconv-gnn/build/exp_logs/pna_base.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/probe_1.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    10 \
    --patience \
    4 \
    --batch-size \
    128 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    32 \
    --max-out-neighbors \
    32 \
    --seed \
    42
epoch=1 train_loss=0.001375 val_loss=0.000031 p@5=0.0539 ndcg@5=0.1091
epoch=2 train_loss=0.000039 val_loss=0.000016 p@5=0.0719 ndcg@5=0.1105
epoch=3 train_loss=0.000022 val_loss=0.000010 p@5=0.1198 ndcg@5=0.1495
epoch=4 train_loss=0.000016 val_loss=0.000008 p@5=0.1916 ndcg@5=0.1770
epoch=5 train_loss=0.000012 val_loss=0.000008 p@5=0.2455 ndcg@5=0.2464
epoch=6 train_loss=0.000009 val_loss=0.000007 p@5=0.2695 ndcg@5=0.2509
epoch=7 train_loss=0.000008 val_loss=0.000005 p@5=0.2575 ndcg@5=0.3647
epoch=8 train_loss=0.000007 val_loss=0.000006 p@5=0.3054 ndcg@5=0.4600
epoch=9 train_loss=0.000006 val_loss=0.000004 p@5=0.1497 ndcg@5=0.3609
epoch=10 train_loss=0.000007 val_loss=0.000004 p@5=0.1198 ndcg@5=0.3397
test_loss=0.000008
test_p@5=0.1759 test_ndcg@5=0.2863
saved=pnaconv-gnn/build/exp_logs/probe_1.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/probe_2.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.1 \
    --epochs \
    10 \
    --patience \
    4 \
    --batch-size \
    128 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    64 \
    --max-out-neighbors \
    64 \
    --seed \
    42
interrupted_before_first_epoch_output

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/focus_1.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    8 \
    --patience \
    3 \
    --batch-size \
    128 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    16 \
    --max-out-neighbors \
    16 \
    --seed \
    42
epoch=1 train_loss=0.001375 val_loss=0.000031 p@5=0.0838 ndcg@5=0.1157
epoch=2 train_loss=0.000040 val_loss=0.000016 p@5=0.0539 ndcg@5=0.1124
epoch=3 train_loss=0.000023 val_loss=0.000009 p@5=0.1317 ndcg@5=0.1549
epoch=4 train_loss=0.000016 val_loss=0.000007 p@5=0.1976 ndcg@5=0.1686
epoch=5 train_loss=0.000013 val_loss=0.000008 p@5=0.1677 ndcg@5=0.2589
epoch=6 train_loss=0.000010 val_loss=0.000006 p@5=0.3054 ndcg@5=0.3516
epoch=7 train_loss=0.000008 val_loss=0.000004 p@5=0.1976 ndcg@5=0.2614
epoch=8 train_loss=0.000007 val_loss=0.000004 p@5=0.3293 ndcg@5=0.3583
test_loss=0.000006
test_p@5=0.3065 test_ndcg@5=0.2981
saved=pnaconv-gnn/build/exp_logs/focus_1.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/focus_2.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    8 \
    --patience \
    3 \
    --batch-size \
    128 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    64 \
    --max-out-neighbors \
    64 \
    --seed \
    42
epoch=1 train_loss=0.001374 val_loss=0.000031 p@5=0.0539 ndcg@5=0.1190
epoch=2 train_loss=0.000038 val_loss=0.000016 p@5=0.0359 ndcg@5=0.1044
epoch=3 train_loss=0.000022 val_loss=0.000009 p@5=0.1138 ndcg@5=0.1488
epoch=4 train_loss=0.000015 val_loss=0.000007 p@5=0.1856 ndcg@5=0.1747
interrupted_after_epoch_4

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/refine_1.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    8 \
    --patience \
    3 \
    --batch-size \
    128 \
    --lr \
    2e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    16 \
    --max-out-neighbors \
    16 \
    --seed \
    42
epoch=1 train_loss=0.002697 val_loss=0.000013 p@5=0.0838 ndcg@5=0.0917
epoch=2 train_loss=0.000019 val_loss=0.000008 p@5=0.1377 ndcg@5=0.1461
epoch=3 train_loss=0.000013 val_loss=0.000005 p@5=0.0778 ndcg@5=0.1287
epoch=4 train_loss=0.000010 val_loss=0.000005 p@5=0.0659 ndcg@5=0.1193
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.0898 ndcg@5=0.1373
early_stop_epoch=5
test_loss=0.000010
test_p@5=0.1206 test_ndcg@5=0.2648
saved=pnaconv-gnn/build/exp_logs/refine_1.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/refine_2.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    8 \
    --patience \
    3 \
    --batch-size \
    128 \
    --lr \
    5e-5 \
    --num-hops \
    1 \
    --max-in-neighbors \
    16 \
    --max-out-neighbors \
    16 \
    --seed \
    42
epoch=1 train_loss=0.001427 val_loss=0.000075 p@5=0.0120 ndcg@5=0.0682
epoch=2 train_loss=0.000072 val_loss=0.000028 p@5=0.0539 ndcg@5=0.0836
epoch=3 train_loss=0.000035 val_loss=0.000016 p@5=0.0898 ndcg@5=0.0963
epoch=4 train_loss=0.000023 val_loss=0.000011 p@5=0.0778 ndcg@5=0.0893
interrupted_after_epoch_4

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/final_batch256.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    8 \
    --patience \
    3 \
    --batch-size \
    256 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    16 \
    --max-out-neighbors \
    16 \
    --seed \
    42
epoch=1 train_loss=0.002665 val_loss=0.000096 p@5=0.0359 ndcg@5=0.0670
epoch=2 train_loss=0.000077 val_loss=0.000032 p@5=0.1377 ndcg@5=0.0906
epoch=3 train_loss=0.000043 val_loss=0.000021 p@5=0.2754 ndcg@5=0.1500
epoch=4 train_loss=0.000031 val_loss=0.000016 p@5=0.2994 ndcg@5=0.1673
epoch=5 train_loss=0.000024 val_loss=0.000013 p@5=0.2635 ndcg@5=0.1489
epoch=6 train_loss=0.000019 val_loss=0.000011 p@5=0.2635 ndcg@5=0.1686
epoch=7 train_loss=0.000016 val_loss=0.000009 p@5=0.2216 ndcg@5=0.1656
early_stop_epoch=7
test_loss=0.000023
test_p@5=0.1357 test_ndcg@5=0.2469
saved=pnaconv-gnn/build/exp_logs/final_batch256.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/exp_logs/final_long.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    12 \
    --patience \
    4 \
    --batch-size \
    128 \
    --lr \
    1e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    16 \
    --max-out-neighbors \
    16 \
    --seed \
    42
epoch=1 train_loss=0.001375 val_loss=0.000031 p@5=0.0838 ndcg@5=0.1149
epoch=2 train_loss=0.000040 val_loss=0.000016 p@5=0.0479 ndcg@5=0.1089
epoch=3 train_loss=0.000023 val_loss=0.000009 p@5=0.1198 ndcg@5=0.1555
epoch=4 train_loss=0.000016 val_loss=0.000007 p@5=0.1916 ndcg@5=0.1732
epoch=5 train_loss=0.000013 val_loss=0.000008 p@5=0.1856 ndcg@5=0.2625
epoch=6 train_loss=0.000010 val_loss=0.000006 p@5=0.3234 ndcg@5=0.3511
epoch=7 train_loss=0.000008 val_loss=0.000004 p@5=0.2156 ndcg@5=0.2680
epoch=8 train_loss=0.000007 val_loss=0.000005 p@5=0.3054 ndcg@5=0.4337
epoch=9 train_loss=0.000006 val_loss=0.000004 p@5=0.1138 ndcg@5=0.3297
epoch=10 train_loss=0.000006 val_loss=0.000005 p@5=0.2036 ndcg@5=0.3420
early_stop_epoch=10
test_loss=0.000008
test_p@5=0.3216 test_ndcg@5=0.3163
saved=pnaconv-gnn/build/exp_logs/final_long.pt

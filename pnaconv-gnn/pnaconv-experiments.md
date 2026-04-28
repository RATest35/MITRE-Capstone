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

---

Loss Function Search: PNAConv

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_mse.pt \
    --loss \
    mse \
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
epoch=1 train_loss=0.001375 val_loss=0.000031 p@5=0.0838 ndcg@5=0.1156
epoch=2 train_loss=0.000040 val_loss=0.000016 p@5=0.0539 ndcg@5=0.1120
epoch=3 train_loss=0.000023 val_loss=0.000009 p@5=0.1257 ndcg@5=0.1559
epoch=4 train_loss=0.000016 val_loss=0.000007 p@5=0.1976 ndcg@5=0.1740
epoch=5 train_loss=0.000013 val_loss=0.000008 p@5=0.1737 ndcg@5=0.2595
epoch=6 train_loss=0.000010 val_loss=0.000006 p@5=0.3174 ndcg@5=0.3577
epoch=7 train_loss=0.000008 val_loss=0.000004 p@5=0.1856 ndcg@5=0.2287
epoch=8 train_loss=0.000007 val_loss=0.000004 p@5=0.2994 ndcg@5=0.3482
epoch=9 train_loss=0.000006 val_loss=0.000004 p@5=0.1018 ndcg@5=0.3271
epoch=10 train_loss=0.000006 val_loss=0.000005 p@5=0.2036 ndcg@5=0.2544
early_stop_epoch=10
test_loss=0.000008
test_p@5=0.3317 test_ndcg@5=0.3142
saved=pnaconv-gnn/build/loss_logs/loss_mse.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_l1.pt \
    --loss \
    l1 \
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
epoch=1 train_loss=0.018800 val_loss=0.009553 p@5=0.0240 ndcg@5=0.0592
epoch=2 train_loss=0.007138 val_loss=0.004892 p@5=0.0180 ndcg@5=0.0616
epoch=3 train_loss=0.005153 val_loss=0.006012 p@5=0.2335 ndcg@5=0.0946
epoch=4 train_loss=0.005138 val_loss=0.007374 p@5=0.1737 ndcg@5=0.0792
epoch=5 train_loss=0.004392 val_loss=0.010248 p@5=0.0060 ndcg@5=0.0564
epoch=6 train_loss=0.003932 val_loss=0.003939 p@5=0.1437 ndcg@5=0.1528
epoch=7 train_loss=0.003832 val_loss=0.002475 p@5=0.0000 ndcg@5=0.0549
early_stop_epoch=7
test_loss=0.005924
test_p@5=0.0754 test_ndcg@5=0.1373
saved=pnaconv-gnn/build/loss_logs/loss_l1.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_smooth_l1_b1.pt \
    --loss \
    smooth_l1 \
    --loss-beta \
    1.0 \
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
epoch=1 train_loss=0.000687 val_loss=0.000016 p@5=0.0778 ndcg@5=0.1129
epoch=2 train_loss=0.000020 val_loss=0.000008 p@5=0.0299 ndcg@5=0.1046
epoch=3 train_loss=0.000011 val_loss=0.000005 p@5=0.1198 ndcg@5=0.1513
epoch=4 train_loss=0.000008 val_loss=0.000004 p@5=0.1916 ndcg@5=0.1709
epoch=5 train_loss=0.000006 val_loss=0.000004 p@5=0.2216 ndcg@5=0.2741
epoch=6 train_loss=0.000005 val_loss=0.000003 p@5=0.3234 ndcg@5=0.3533
epoch=7 train_loss=0.000004 val_loss=0.000002 p@5=0.2036 ndcg@5=0.2511
epoch=8 train_loss=0.000004 val_loss=0.000002 p@5=0.3413 ndcg@5=0.3601
epoch=9 train_loss=0.000003 val_loss=0.000002 p@5=0.0958 ndcg@5=0.2395
epoch=10 train_loss=0.000003 val_loss=0.000002 p@5=0.1437 ndcg@5=0.2434
epoch=11 train_loss=0.000003 val_loss=0.000001 p@5=0.2096 ndcg@5=0.3491
epoch=12 train_loss=0.000002 val_loss=0.000001 p@5=0.1796 ndcg@5=0.3432
early_stop_epoch=12
test_loss=0.000003
test_p@5=0.3116 test_ndcg@5=0.2948
saved=pnaconv-gnn/build/loss_logs/loss_smooth_l1_b1.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_smooth_l1_b05.pt \
    --loss \
    smooth_l1 \
    --loss-beta \
    0.5 \
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
epoch=1 train_loss=0.001375 val_loss=0.000031 p@5=0.0838 ndcg@5=0.1156
epoch=2 train_loss=0.000040 val_loss=0.000016 p@5=0.0539 ndcg@5=0.1125
epoch=3 train_loss=0.000023 val_loss=0.000009 p@5=0.1317 ndcg@5=0.1568
epoch=4 train_loss=0.000016 val_loss=0.000007 p@5=0.1976 ndcg@5=0.1741
epoch=5 train_loss=0.000013 val_loss=0.000008 p@5=0.1677 ndcg@5=0.2591
epoch=6 train_loss=0.000010 val_loss=0.000006 p@5=0.3353 ndcg@5=0.3672
epoch=7 train_loss=0.000008 val_loss=0.000004 p@5=0.1796 ndcg@5=0.2608
epoch=8 train_loss=0.000007 val_loss=0.000004 p@5=0.3473 ndcg@5=0.4501
epoch=9 train_loss=0.000006 val_loss=0.000004 p@5=0.1138 ndcg@5=0.3368
epoch=10 train_loss=0.000006 val_loss=0.000004 p@5=0.2395 ndcg@5=0.2675
epoch=11 train_loss=0.000005 val_loss=0.000003 p@5=0.3413 ndcg@5=0.4658
epoch=12 train_loss=0.000005 val_loss=0.000003 p@5=0.3234 ndcg@5=0.4601
early_stop_epoch=12
test_loss=0.000006
test_p@5=0.2864 test_ndcg@5=0.2886
saved=pnaconv-gnn/build/loss_logs/loss_smooth_l1_b05.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_huber_d1.pt \
    --loss \
    huber \
    --loss-beta \
    1.0 \
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
epoch=1 train_loss=0.000687 val_loss=0.000016 p@5=0.0838 ndcg@5=0.1133
epoch=2 train_loss=0.000020 val_loss=0.000008 p@5=0.0299 ndcg@5=0.1044
epoch=3 train_loss=0.000012 val_loss=0.000005 p@5=0.1377 ndcg@5=0.1525
epoch=4 train_loss=0.000008 val_loss=0.000004 p@5=0.1737 ndcg@5=0.1686
epoch=5 train_loss=0.000007 val_loss=0.000004 p@5=0.2335 ndcg@5=0.2745
epoch=6 train_loss=0.000005 val_loss=0.000003 p@5=0.2814 ndcg@5=0.2799
epoch=7 train_loss=0.000004 val_loss=0.000002 p@5=0.1976 ndcg@5=0.2185
epoch=8 train_loss=0.000004 val_loss=0.000002 p@5=0.3593 ndcg@5=0.3697
epoch=9 train_loss=0.000003 val_loss=0.000002 p@5=0.1018 ndcg@5=0.2401
epoch=10 train_loss=0.000003 val_loss=0.000002 p@5=0.1737 ndcg@5=0.3375
epoch=11 train_loss=0.000003 val_loss=0.000001 p@5=0.2335 ndcg@5=0.3629
epoch=12 train_loss=0.000003 val_loss=0.000002 p@5=0.1976 ndcg@5=0.3509
early_stop_epoch=12
test_loss=0.000003
test_p@5=0.3266 test_ndcg@5=0.2257
saved=pnaconv-gnn/build/loss_logs/loss_huber_d1.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_huber_d05.pt \
    --loss \
    huber \
    --loss-beta \
    0.5 \
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
epoch=1 train_loss=0.000687 val_loss=0.000016 p@5=0.0778 ndcg@5=0.1133
epoch=2 train_loss=0.000020 val_loss=0.000008 p@5=0.0299 ndcg@5=0.1046
epoch=3 train_loss=0.000011 val_loss=0.000005 p@5=0.1257 ndcg@5=0.1516
epoch=4 train_loss=0.000008 val_loss=0.000004 p@5=0.1916 ndcg@5=0.1678
epoch=5 train_loss=0.000007 val_loss=0.000005 p@5=0.2216 ndcg@5=0.2753
epoch=6 train_loss=0.000005 val_loss=0.000003 p@5=0.3353 ndcg@5=0.3577
epoch=7 train_loss=0.000004 val_loss=0.000002 p@5=0.2096 ndcg@5=0.2206
epoch=8 train_loss=0.000004 val_loss=0.000002 p@5=0.3533 ndcg@5=0.3576
epoch=9 train_loss=0.000003 val_loss=0.000002 p@5=0.0958 ndcg@5=0.2399
epoch=10 train_loss=0.000003 val_loss=0.000002 p@5=0.1557 ndcg@5=0.2463
epoch=11 train_loss=0.000002 val_loss=0.000001 p@5=0.2096 ndcg@5=0.3502
epoch=12 train_loss=0.000003 val_loss=0.000002 p@5=0.1796 ndcg@5=0.3473
early_stop_epoch=12
test_loss=0.000003
test_p@5=0.3065 test_ndcg@5=0.2953
saved=pnaconv-gnn/build/loss_logs/loss_huber_d05.pt

---

(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 pnaconv-gnn/train.py \
    --output-path \
    pnaconv-gnn/build/loss_logs/loss_log_cosh.pt \
    --loss \
    log_cosh \
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
epoch=1 train_loss=0.000676 val_loss=0.000015 p@5=0.1257 ndcg@5=0.0891
epoch=2 train_loss=0.000019 val_loss=0.000007 p@5=0.1198 ndcg@5=0.1322
epoch=3 train_loss=0.000011 val_loss=0.000005 p@5=0.1257 ndcg@5=0.1616
epoch=4 train_loss=0.000008 val_loss=0.000004 p@5=0.2096 ndcg@5=0.1893
epoch=5 train_loss=0.000006 val_loss=0.000004 p@5=0.2335 ndcg@5=0.2310
epoch=6 train_loss=0.000005 val_loss=0.000003 p@5=0.1677 ndcg@5=0.2267
epoch=7 train_loss=0.000004 val_loss=0.000002 p@5=0.2695 ndcg@5=0.2811
epoch=8 train_loss=0.000004 val_loss=0.000004 p@5=0.3293 ndcg@5=0.2824
epoch=9 train_loss=0.000003 val_loss=0.000003 p@5=0.1198 ndcg@5=0.1927
epoch=10 train_loss=0.000003 val_loss=0.000002 p@5=0.1317 ndcg@5=0.2502
epoch=11 train_loss=0.000003 val_loss=0.000002 p@5=0.2575 ndcg@5=0.3585
epoch=12 train_loss=0.000003 val_loss=0.000002 p@5=0.2335 ndcg@5=0.2326
early_stop_epoch=12
test_loss=0.000004
test_p@5=0.2663 test_ndcg@5=0.3254
saved=pnaconv-gnn/build/loss_logs/loss_log_cosh.pt

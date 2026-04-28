
Structure: TransformerConv


python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/baseline.pt
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/baseline.pt
terminated_manually_due_to_runtime_screening_strategy_change=1

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/screen_a.pt \
    --hidden-dim \
    64 \
    --num-layers \
    2 \
    --dropout \
    0.1 \
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
    --epochs \
    12 \
    --patience \
    4 \
    --seed \
    42
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/screen_a.pt --hidden-dim 64 --num-layers 2 --dropout 0.1 --batch-size 64 --lr 1e-4 --num-hops 1 --max-in-neighbors 32 --max-out-neighbors 32 --epochs 12 --patience 4 --seed 42
Traceback (most recent call last):
  File "/Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/transformerconv-gnn/train.py", line 7, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
command_failed_exit_code=1

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/screen_a_rerun.pt \
    --hidden-dim \
    64 \
    --num-layers \
    2 \
    --dropout \
    0.1 \
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
    --epochs \
    12 \
    --patience \
    4 \
    --seed \
    42
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/screen_a_rerun.pt --hidden-dim 64 --num-layers 2 --dropout 0.1 --batch-size 64 --lr 1e-4 --num-hops 1 --max-in-neighbors 32 --max-out-neighbors 32 --epochs 12 --patience 4 --seed 42
epoch=1 train_loss=0.008015 val_loss=0.000078 p@5=0.0120 ndcg@5=0.0565
epoch=2 train_loss=0.002877 val_loss=0.000035 p@5=0.0240 ndcg@5=0.0603
epoch=3 train_loss=0.001633 val_loss=0.000073 p@5=0.0060 ndcg@5=0.0569
epoch=4 train_loss=0.001042 val_loss=0.000150 p@5=0.0120 ndcg@5=0.0711
epoch=5 train_loss=0.000758 val_loss=0.000136 p@5=0.0120 ndcg@5=0.0754
terminated_manually_due_to_slow_epoch_and_low_validation_precision=1

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_1.pt \
    --hidden-dim \
    64 \
    --num-layers \
    2 \
    --dropout \
    0.1 \
    --epochs \
    8 \
    --patience \
    3 \
    --batch-size \
    128 \
    --lr \
    3e-4 \
    --num-hops \
    1 \
    --max-in-neighbors \
    8 \
    --max-out-neighbors \
    8 \
    --seed \
    42
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_1.pt --hidden-dim 64 --num-layers 2 --dropout 0.1 --epochs 8 --patience 3 --batch-size 128 --lr 3e-4 --num-hops 1 --max-in-neighbors 8 --max-out-neighbors 8 --seed 42
epoch=1 train_loss=0.007117 val_loss=0.000124 p@5=0.0240 ndcg@5=0.0658
epoch=2 train_loss=0.002083 val_loss=0.000035 p@5=0.0120 ndcg@5=0.0578
epoch=3 train_loss=0.001126 val_loss=0.000121 p@5=0.0180 ndcg@5=0.0796
epoch=4 train_loss=0.000714 val_loss=0.000134 p@5=0.0240 ndcg@5=0.0867
early_stop_epoch=4
test_loss=0.000131
test_p@5=0.0302 test_ndcg@5=0.1142
saved=transformerconv-gnn/build/exp_logs/fast_probe_1.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_2.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.1 \
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
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_2.pt --hidden-dim 128 --num-layers 2 --dropout 0.1 --epochs 8 --patience 3 --batch-size 128 --lr 2e-4 --num-hops 1 --max-in-neighbors 16 --max-out-neighbors 16 --seed 42
epoch=1 train_loss=0.004449 val_loss=0.000107 p@5=0.0419 ndcg@5=0.1437
epoch=2 train_loss=0.001178 val_loss=0.000092 p@5=0.2096 ndcg@5=0.1518
epoch=3 train_loss=0.000568 val_loss=0.000036 p@5=0.0060 ndcg@5=0.0545
epoch=4 train_loss=0.000313 val_loss=0.000012 p@5=0.0958 ndcg@5=0.1087
epoch=5 train_loss=0.000196 val_loss=0.000006 p@5=0.0000 ndcg@5=0.0531
early_stop_epoch=5
test_loss=0.000110
test_p@5=0.1357 test_ndcg@5=0.2616
saved=transformerconv-gnn/build/exp_logs/fast_probe_2.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_3.pt \
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
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_3.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 8 --patience 3 --batch-size 128 --lr 2e-4 --num-hops 1 --max-in-neighbors 16 --max-out-neighbors 16 --seed 42
epoch=1 train_loss=0.000586 val_loss=0.000010 p@5=0.1796 ndcg@5=0.0879
epoch=2 train_loss=0.000011 val_loss=0.000005 p@5=0.2335 ndcg@5=0.1620
epoch=3 train_loss=0.000007 val_loss=0.000005 p@5=0.2515 ndcg@5=0.1969
epoch=4 train_loss=0.000006 val_loss=0.000003 p@5=0.2335 ndcg@5=0.1635
epoch=5 train_loss=0.000005 val_loss=0.000002 p@5=0.2814 ndcg@5=0.2159
epoch=6 train_loss=0.000005 val_loss=0.000002 p@5=0.3772 ndcg@5=0.2612
epoch=7 train_loss=0.000004 val_loss=0.000002 p@5=0.2874 ndcg@5=0.1858
epoch=8 train_loss=0.000004 val_loss=0.000002 p@5=0.4431 ndcg@5=0.3011
test_loss=0.000003
test_p@5=0.2915 test_ndcg@5=0.3089
saved=transformerconv-gnn/build/exp_logs/fast_probe_3.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_4.pt \
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
    16 \
    --max-out-neighbors \
    16 \
    --seed \
    42
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_4.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 10 --patience 4 --batch-size 128 --lr 1e-4 --num-hops 1 --max-in-neighbors 16 --max-out-neighbors 16 --seed 42
epoch=1 train_loss=0.000706 val_loss=0.000027 p@5=0.0000 ndcg@5=0.0590
epoch=2 train_loss=0.000029 val_loss=0.000010 p@5=0.2036 ndcg@5=0.0895
epoch=3 train_loss=0.000014 val_loss=0.000006 p@5=0.1856 ndcg@5=0.0925
epoch=4 train_loss=0.000010 val_loss=0.000006 p@5=0.2335 ndcg@5=0.0995
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.2216 ndcg@5=0.1031
epoch=6 train_loss=0.000007 val_loss=0.000003 p@5=0.2635 ndcg@5=0.1106
epoch=7 train_loss=0.000006 val_loss=0.000003 p@5=0.1677 ndcg@5=0.0946
epoch=8 train_loss=0.000005 val_loss=0.000002 p@5=0.3234 ndcg@5=0.1549
epoch=9 train_loss=0.000005 val_loss=0.000002 p@5=0.2934 ndcg@5=0.1935
epoch=10 train_loss=0.000004 val_loss=0.000002 p@5=0.4012 ndcg@5=0.2475
test_loss=0.000001
test_p@5=0.3166 test_ndcg@5=0.3260
saved=transformerconv-gnn/build/exp_logs/fast_probe_4.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_5.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    14 \
    --patience \
    5 \
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
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_5.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 14 --patience 5 --batch-size 128 --lr 1e-4 --num-hops 1 --max-in-neighbors 16 --max-out-neighbors 16 --seed 42
epoch=1 train_loss=0.000706 val_loss=0.000027 p@5=0.0000 ndcg@5=0.0590
epoch=2 train_loss=0.000029 val_loss=0.000010 p@5=0.2036 ndcg@5=0.0895
epoch=3 train_loss=0.000014 val_loss=0.000006 p@5=0.1856 ndcg@5=0.0925
epoch=4 train_loss=0.000010 val_loss=0.000006 p@5=0.2335 ndcg@5=0.0995
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.2515 ndcg@5=0.1078
epoch=6 train_loss=0.000007 val_loss=0.000003 p@5=0.3054 ndcg@5=0.1248
epoch=7 train_loss=0.000006 val_loss=0.000003 p@5=0.1317 ndcg@5=0.0876
epoch=8 train_loss=0.000005 val_loss=0.000002 p@5=0.3413 ndcg@5=0.1584
epoch=9 train_loss=0.000005 val_loss=0.000002 p@5=0.3174 ndcg@5=0.1888
epoch=10 train_loss=0.000004 val_loss=0.000002 p@5=0.4192 ndcg@5=0.2541
epoch=11 train_loss=0.000004 val_loss=0.000002 p@5=0.1317 ndcg@5=0.1665
epoch=12 train_loss=0.000004 val_loss=0.000002 p@5=0.2335 ndcg@5=0.1961
epoch=13 train_loss=0.000003 val_loss=0.000002 p@5=0.3293 ndcg@5=0.2637
epoch=14 train_loss=0.000003 val_loss=0.000001 p@5=0.2695 ndcg@5=0.2198
test_loss=0.000002
test_p@5=0.3065 test_ndcg@5=0.3219
saved=transformerconv-gnn/build/exp_logs/fast_probe_5.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_6.pt \
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
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_6.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 10 --patience 4 --batch-size 128 --lr 1e-4 --num-hops 1 --max-in-neighbors 32 --max-out-neighbors 32 --seed 42
epoch=1 train_loss=0.000706 val_loss=0.000026 p@5=0.0000 ndcg@5=0.0589
epoch=2 train_loss=0.000029 val_loss=0.000011 p@5=0.2335 ndcg@5=0.0934
epoch=3 train_loss=0.000014 val_loss=0.000006 p@5=0.2216 ndcg@5=0.0970
epoch=4 train_loss=0.000010 val_loss=0.000005 p@5=0.2395 ndcg@5=0.0973
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.2156 ndcg@5=0.0988
epoch=6 train_loss=0.000007 val_loss=0.000003 p@5=0.3054 ndcg@5=0.1150
epoch=7 train_loss=0.000006 val_loss=0.000003 p@5=0.1617 ndcg@5=0.0872
epoch=8 train_loss=0.000005 val_loss=0.000002 p@5=0.3114 ndcg@5=0.1606
epoch=9 train_loss=0.000005 val_loss=0.000002 p@5=0.4192 ndcg@5=0.2037
epoch=10 train_loss=0.000004 val_loss=0.000002 p@5=0.4731 ndcg@5=0.2408
test_loss=0.000001
test_p@5=0.3266 test_ndcg@5=0.3286
saved=transformerconv-gnn/build/exp_logs/fast_probe_6.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_7.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    14 \
    --patience \
    5 \
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
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_7.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 14 --patience 5 --batch-size 128 --lr 1e-4 --num-hops 1 --max-in-neighbors 32 --max-out-neighbors 32 --seed 42
epoch=1 train_loss=0.000706 val_loss=0.000026 p@5=0.0000 ndcg@5=0.0589
epoch=2 train_loss=0.000029 val_loss=0.000011 p@5=0.2335 ndcg@5=0.0934
epoch=3 train_loss=0.000014 val_loss=0.000006 p@5=0.2216 ndcg@5=0.0970
epoch=4 train_loss=0.000010 val_loss=0.000005 p@5=0.2395 ndcg@5=0.0973
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.2156 ndcg@5=0.0988
epoch=6 train_loss=0.000007 val_loss=0.000003 p@5=0.3054 ndcg@5=0.1150
epoch=7 train_loss=0.000006 val_loss=0.000003 p@5=0.1617 ndcg@5=0.0872
epoch=8 train_loss=0.000005 val_loss=0.000002 p@5=0.3114 ndcg@5=0.1606
epoch=9 train_loss=0.000005 val_loss=0.000002 p@5=0.4311 ndcg@5=0.2128
epoch=10 train_loss=0.000004 val_loss=0.000002 p@5=0.4731 ndcg@5=0.2485
epoch=11 train_loss=0.000004 val_loss=0.000002 p@5=0.2036 ndcg@5=0.1645
epoch=12 train_loss=0.000004 val_loss=0.000002 p@5=0.2754 ndcg@5=0.1754
epoch=13 train_loss=0.000004 val_loss=0.000002 p@5=0.3832 ndcg@5=0.2167
epoch=14 train_loss=0.000003 val_loss=0.000001 p@5=0.3114 ndcg@5=0.1809
test_loss=0.000001
test_p@5=0.3216 test_ndcg@5=0.3377
saved=transformerconv-gnn/build/exp_logs/fast_probe_7.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_8.pt \
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
    64 \
    --max-out-neighbors \
    64 \
    --seed \
    42
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_8.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 10 --patience 4 --batch-size 128 --lr 1e-4 --num-hops 1 --max-in-neighbors 64 --max-out-neighbors 64 --seed 42
epoch=1 train_loss=0.000706 val_loss=0.000026 p@5=0.0000 ndcg@5=0.0582
epoch=2 train_loss=0.000029 val_loss=0.000012 p@5=0.2036 ndcg@5=0.0909
epoch=3 train_loss=0.000014 val_loss=0.000006 p@5=0.1976 ndcg@5=0.0942
epoch=4 train_loss=0.000010 val_loss=0.000005 p@5=0.2096 ndcg@5=0.0950
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.2156 ndcg@5=0.1002
epoch=6 train_loss=0.000007 val_loss=0.000003 p@5=0.2814 ndcg@5=0.1472
epoch=7 train_loss=0.000006 val_loss=0.000003 p@5=0.0958 ndcg@5=0.0780
epoch=8 train_loss=0.000005 val_loss=0.000002 p@5=0.2874 ndcg@5=0.1517
epoch=9 train_loss=0.000005 val_loss=0.000002 p@5=0.3234 ndcg@5=0.1634
epoch=10 train_loss=0.000004 val_loss=0.000002 p@5=0.4192 ndcg@5=0.2234
test_loss=0.000001
test_p@5=0.3467 test_ndcg@5=0.3145
saved=transformerconv-gnn/build/exp_logs/fast_probe_8.pt

---

python3 \
    transformerconv-gnn/train.py \
    --output-path \
    transformerconv-gnn/build/exp_logs/fast_probe_9.pt \
    --hidden-dim \
    128 \
    --num-layers \
    2 \
    --dropout \
    0.0 \
    --epochs \
    14 \
    --patience \
    5 \
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
(venv) keisukemiyamoto@KeisukenoMacBook-Pro-2 MITRE-Capstone % python3 transformerconv-gnn/train.py --output-path transformerconv-gnn/build/exp_logs/fast_probe_9.pt --hidden-dim 128 --num-layers 2 --dropout 0.0 --epochs 14 --patience 5 --batch-size 128 --lr 1e-4 --num-hops 1 --max-in-neighbors 64 --max-out-neighbors 64 --seed 42
epoch=1 train_loss=0.000706 val_loss=0.000026 p@5=0.0000 ndcg@5=0.0582
epoch=2 train_loss=0.000029 val_loss=0.000012 p@5=0.2036 ndcg@5=0.0909
epoch=3 train_loss=0.000014 val_loss=0.000006 p@5=0.1976 ndcg@5=0.0942
epoch=4 train_loss=0.000010 val_loss=0.000005 p@5=0.2096 ndcg@5=0.0950
epoch=5 train_loss=0.000008 val_loss=0.000004 p@5=0.2156 ndcg@5=0.1002
epoch=6 train_loss=0.000007 val_loss=0.000003 p@5=0.2814 ndcg@5=0.1472
epoch=7 train_loss=0.000006 val_loss=0.000003 p@5=0.0958 ndcg@5=0.0779
epoch=8 train_loss=0.000005 val_loss=0.000002 p@5=0.2874 ndcg@5=0.1172
epoch=9 train_loss=0.000005 val_loss=0.000002 p@5=0.2754 ndcg@5=0.1502
epoch=10 train_loss=0.000004 val_loss=0.000002 p@5=0.3892 ndcg@5=0.1987
epoch=11 train_loss=0.000004 val_loss=0.000002 p@5=0.2156 ndcg@5=0.1555
epoch=12 train_loss=0.000004 val_loss=0.000002 p@5=0.3413 ndcg@5=0.1885
epoch=13 train_loss=0.000003 val_loss=0.000001 p@5=0.3473 ndcg@5=0.1977
epoch=14 train_loss=0.000003 val_loss=0.000001 p@5=0.2515 ndcg@5=0.1793
test_loss=0.000001
test_p@5=0.3216 test_ndcg@5=0.2940
saved=transformerconv-gnn/build/exp_logs/fast_probe_9.pt

---

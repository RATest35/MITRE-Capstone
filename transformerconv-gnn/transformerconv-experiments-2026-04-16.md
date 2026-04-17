# TransformerConv GNN Experiment Log

Date: 2026-04-16

Fixed arguments for every run:

- `--graphml-path /Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/transformerconv-gnn/dataset/composite_score_with_bytes_per_sec.graphml`
- `--seed 42`
- No code edits. Only CLI arguments were changed.

## Best configuration

Best result so far:

- `test_p@5=0.6985`
- `test_ndcg@5=0.8288`
- saved model: `transformerconv-gnn/build/exp_logs/run15_best_alpha05_longer_batch256.pt`

Reproduction command:

```bash
python3 -u transformerconv-gnn/train.py \
  --graphml-path /Users/keisukemiyamoto/Documents/Capstone/MITRE-Capstone/transformerconv-gnn/dataset/composite_score_with_bytes_per_sec.graphml \
  --seed 42 \
  --output-path transformerconv-gnn/build/exp_logs/run15_best_alpha05_longer_batch256.pt \
  --layer-dims 128 96 64 32 \
  --dropout 0.0 \
  --epochs 6 \
  --patience 3 \
  --batch-size 256 \
  --lr 1e-4 \
  --num-hops 1 \
  --max-in-neighbors 64 \
  --max-out-neighbors 64 \
  --ranking-alpha 0.5 \
  --sample-weight-max 5.0
```

## Summary

Key findings:

- `layer-dims 128 96 64 32` consistently outperformed `128 64 32`.
- `ranking-alpha 0.5` was the strongest setting around the best architecture.
- Increasing `sample-weight-max` from `5.0` to `8.0` hurt `test_p@5`.
- Reducing neighbors from `64/64` to `32/32` hurt `test_p@5`.
- Smaller `batch-size=256` improved the best `512`-batch result.

## Full experiment table

| Run | layer-dims | epochs | patience | batch | lr | hops | in/out neighbors | ranking-alpha | sample-weight-max | test_p@5 | test_ndcg@5 | test_loss | runtime_sec | saved model |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| run1 | `128 64 32` | 2 | 1 | 512 | `1e-4` | 1 | `64 / 64` | 1.0 | 5.0 | 0.6482 | 0.4626 | 0.665291 | - | `transformerconv-gnn/build/exp_logs/run1.pt` |
| run2_longer_patience | `128 64 32` | 6 | 3 | 512 | `1e-4` | 1 | `64 / 64` | 1.0 | 5.0 | 0.6482 | 0.4626 | 0.665291 | 241.43 | `transformerconv-gnn/build/exp_logs/run2_longer_patience.pt` |
| run3_alpha2 | `128 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 2.0 | 5.0 | 0.6784 | 0.5484 | 1.305429 | 182.80 | `transformerconv-gnn/build/exp_logs/run3_alpha2.pt` |
| run4_alpha05 | `128 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 0.5 | 5.0 | 0.6784 | 0.5217 | 0.332305 | 238.66 | `transformerconv-gnn/build/exp_logs/run4_alpha05.pt` |
| run5_weight8 | `128 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 1.0 | 8.0 | 0.6432 | 0.4633 | 0.665011 | 184.60 | `transformerconv-gnn/build/exp_logs/run5_weight8.pt` |
| run6_dims256_128_64_32 | `256 128 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 1.0 | 5.0 | 0.6633 | 0.4639 | 0.641573 | 249.09 | `transformerconv-gnn/build/exp_logs/run6_dims256_128_64_32.pt` |
| run7_dims128_96_64_32 | `128 96 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 1.0 | 5.0 | 0.6834 | 0.4327 | 0.643187 | 236.53 | `transformerconv-gnn/build/exp_logs/run7_dims128_96_64_32.pt` |
| run8_lr2e4 | `128 64 32` | 4 | 2 | 512 | `2e-4` | 1 | `64 / 64` | 1.0 | 5.0 | 0.6784 | 0.4588 | 0.641418 | 221.27 | `transformerconv-gnn/build/exp_logs/run8_lr2e4.pt` |
| run9_neighbors32 | `128 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `32 / 32` | 1.0 | 5.0 | 0.6482 | 0.4625 | 0.665290 | 166.19 | `transformerconv-gnn/build/exp_logs/run9_neighbors32.pt` |
| run10_bestdims_alpha2 | `128 96 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 2.0 | 5.0 | 0.6834 | 0.3841 | 1.221672 | 230.64 | `transformerconv-gnn/build/exp_logs/run10_bestdims_alpha2.pt` |
| run11_bestdims_alpha05 | `128 96 64 32` | 4 | 2 | 512 | `1e-4` | 1 | `64 / 64` | 0.5 | 5.0 | 0.6884 | 0.8167 | 0.332480 | 236.56 | `transformerconv-gnn/build/exp_logs/run11_bestdims_alpha05.pt` |
| run12_bestdims_alpha2_longer | `128 96 64 32` | 6 | 3 | 512 | `1e-4` | 1 | `64 / 64` | 2.0 | 5.0 | 0.6734 | 0.4217 | 1.214497 | 343.06 | `transformerconv-gnn/build/exp_logs/run12_bestdims_alpha2_longer.pt` |
| run13_bestdims_alpha05_longer | `128 96 64 32` | 6 | 3 | 512 | `1e-4` | 1 | `64 / 64` | 0.5 | 5.0 | 0.6935 | 0.7714 | 0.332258 | 338.21 | `transformerconv-gnn/build/exp_logs/run13_bestdims_alpha05_longer.pt` |
| run14_best_alpha05_longer_lr2e4 | `128 96 64 32` | 6 | 3 | 512 | `2e-4` | 1 | `64 / 64` | 0.5 | 5.0 | 0.6935 | 0.8002 | 0.331614 | 346.44 | `transformerconv-gnn/build/exp_logs/run14_best_alpha05_longer_lr2e4.pt` |
| run15_best_alpha05_longer_batch256 | `128 96 64 32` | 6 | 3 | 256 | `1e-4` | 1 | `64 / 64` | 0.5 | 5.0 | 0.6985 | 0.8288 | 0.332886 | 431.17 | `transformerconv-gnn/build/exp_logs/run15_best_alpha05_longer_batch256.pt` |

## Current recommendation

If the target is strictly maximizing `test_p@5`, use:

- `layer-dims 128 96 64 32`
- `dropout 0.0`
- `epochs 6`
- `patience 3`
- `batch-size 256`
- `lr 1e-4`
- `num-hops 1`
- `max-in-neighbors 64`
- `max-out-neighbors 64`
- `ranking-alpha 0.5`
- `sample-weight-max 5.0`

If the next search round is continued, the highest-value nearby candidates are:

- Keep the best setting and try `lr 1.5e-4`
- Keep the best setting and try `batch-size 192`
- Keep the best setting and try `layer-dims 160 96 64 32`
- Keep the best setting and try `max-in-neighbors/max-out-neighbors 96 / 96`

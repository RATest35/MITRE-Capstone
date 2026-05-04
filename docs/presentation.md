python3 graphsage-gnn/train.py \
    --graphml-path composite_score_with_bytes_per_sec.graphml \
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

python3 pnaconv-gnn/train.py \
    --graphml-path composite_score_with_bytes_per_sec.graphml \
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

python3 transformerconv-gnn/train.py \
    --graphml-path composite_score_with_bytes_per_sec.graphml \
    --output-path gnn/composite_score_gnn.pt \
    --layer-dims 128 96 64 32 \
    --dropout 0.0 \
    --epochs 50 \
    --patience 5 \ 
    --batch-size 128 \
    --lr 2e-4 \
    --num-hops 1 \
    --max-in-neighbors 16 \
    --max-out-neighbors 16 \
    --ranking-alpha 0.1 \
    --sample-weight-max 2.0 \
    --seed 42
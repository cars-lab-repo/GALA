
## GALA/scripts/run_experiments.sh

```bash
#!/bin/bash
# Run GALA experiments

echo "============================================"
echo "GALA Experiments"
echo "============================================"

# Extract features first
echo "Extracting power and area features..."
python scripts/extract_features.py --benchmarks ../benchmarks/ISCAS85/locked --tool estimate

# Create directories
mkdir -p checkpoints
mkdir -p runs
mkdir -p results

# Graph-level experiments
echo ""
echo "========== Graph-level GALA =========="
for lock in xor mux lut sar; do
    echo ""
    echo "Training graph-level model on ${lock} locking..."
    python train.py --config config/config_${lock}.yaml --model graph
    
    echo "Testing graph-level model on ${lock} locking..."
    python test.py --checkpoint checkpoints/best_graph_${lock}_model.pth --model graph > results/graph_${lock}_results.txt
done

# Subgraph-level experiments
echo ""
echo "========== Subgraph-level GALA =========="
for lock in xor mux lut sar; do
    echo ""
    echo "Training subgraph-level model on ${lock} locking..."
    python train.py --config config/config_${lock}.yaml --model subgraph
    
    echo "Testing subgraph-level model on ${lock} locking..."
    python test.py --checkpoint checkpoints/best_subgraph_${lock}_model.pth --model subgraph > results/subgraph_${lock}_results.txt
done

# Mixed locking
echo ""
echo "Training on mixed locking schemes (graph-level)..."
python train.py --config config/config_mixed.yaml --model graph

echo "Testing on mixed locking (graph-level)..."
python test.py --checkpoint checkpoints/best_graph_mixed_model.pth --model graph > results/graph_mixed_results.txt

echo ""
echo "============================================"
echo "GALA experiments complete! Results in results/"
echo "============================================"
```

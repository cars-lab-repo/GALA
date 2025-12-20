## LIPSTICK/scripts/run_experiments.sh

```bash
#!/bin/bash
# Run LIPSTICK experiments

echo "============================================"
echo "LIPSTICK Experiments"
echo "============================================"

# Create directories
mkdir -p checkpoints
mkdir -p runs
mkdir -p results

# Single locking schemes
for lock in xor mux lut sar; do
    echo ""
    echo "Training on ${lock} locking..."
    python train.py --config config/config_${lock}.yaml
    
    echo "Testing on ${lock} locking..."
    python test.py --checkpoint checkpoints/best_${lock}_model.pth --split test > results/${lock}_results.txt
done

# Mixed locking schemes
echo ""
echo "Training on mixed locking schemes..."
python train.py --config config/config_mixed.yaml

echo "Testing on mixed locking..."
python test.py --checkpoint checkpoints/best_mixed_model.pth --split test > results/mixed_results.txt

echo ""
echo "============================================"
echo "Experiments complete! Results in results/"
echo "============================================"
```
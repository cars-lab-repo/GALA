## GALA/README.md

```markdown
# GALA: GNN-based Oracle-Less Logic Locking Attack

GALA (GNN-based Approach for enhancing oracle-less Logic locking Attacks) is an enhanced version of LIPSTICK that incorporates **behavioral features** (power consumption and area overhead) in addition to functional features.

## Key Improvements over LIPSTICK

1. **Behavioral Features**: Integrates gate-level power (static + dynamic) and area metrics
2. **Dual Attack Modes**: 
   - Subgraph-level attack (enhanced OMLA)
   - Graph-level attack (enhanced LIPSTICK)
3. **Higher Accuracy**: 14-17% KPR improvement over LIPSTICK
4. **Better Explainability**: Reveals how power/area patterns influence key recovery

## Performance

### GALA vs OMLA (Subgraph-level)
- **GALA**: 97% KPA, 85% KPR
- **OMLA**: 89% KPA, 62% KPR

### GALA vs LIPSTICK (Graph-level)
- **GALA**: 91% average KPA
- **LIPSTICK**: 84% average KPA

## Usage

### Extract Features

```bash
# Using PrimeTime (requires Synopsys license)
python scripts/extract_features.py --benchmarks ../benchmarks/ISCAS85/locked --tool primetime

# Using estimation (no external tools)
python scripts/extract_features.py --benchmarks ../benchmarks/ISCAS85/locked --tool estimate
```

### Train Model

```bash
# Graph-level GALA
python train.py --config config/config.yaml --model graph

# Subgraph-level GALA
python train.py --config config/config.yaml --model subgraph
```

### Test Model

```bash
python test.py --checkpoint checkpoints/best_graph_model.pth --model graph
```

## Architecture

GALA extends the GIN architecture with:
- Additional power and area node features (3 extra dimensions)
- Multi-task learning heads for:
  - Key prediction (primary)
  - Lock type classification
  - Error rate prediction
  - Power prediction
  - Area prediction

## Citation

```bibtex
@article{aghamohammadi2025gala,
  title={GALA: An Explainable GNN-based Approach for Enhancing Oracle-Less Logic Locking Attacks Using Functional and Behavioral Features},
  author={Aghamohammadi, Yeganeh and Jin, Henry and Rezaei, Amin},
  year={2025}
}
```

## Files

- `models/gala_model.py`: Graph-level GALA model
- `models/gala_subgraph.py`: Subgraph-level GALA model
- `utils/power_area_extractor.py`: Power/area feature extraction
- `scripts/extract_features.py`: Feature extraction script
- `train.py`: Training script
- `test.py`: Testing script
```
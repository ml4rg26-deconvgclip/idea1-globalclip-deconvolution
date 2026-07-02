# ML4RG Idea 1: globalCLIP Deconvolution

This repository is the starting point for the ML4RG Idea 1 globalCLIP deconvolution project.

The project uses a pretrained Parnet model to predict 223 RBP-cell-line profiles from 600 nt RNA windows. A separate reconstruction head will then be trained to combine those predicted profiles into one observed globalCLIP profile for each window.

This scaffold does not implement full training yet. It defines the initial repository layout, configuration entry points, and script interfaces so the data schema, Parnet loading path, reconstruction-head training loop, and evaluation protocol can be filled in incrementally.

## Repository Layout

```text
.
├── configs/
│   └── vm.yaml
├── docs/
├── results/
├── scripts/
│   ├── evaluate_reconstruction.py
│   ├── inspect_globalclip_schema.py
│   ├── smoke_test_parnet.py
│   └── train_reconstruction_head.py
├── .gitignore
└── README.md
```

## Configuration

All machine-specific paths should live in `configs/vm.yaml`. The scripts read paths from that config instead of embedding local filesystem paths in code.

The default config uses project-relative placeholders such as `data/`, `models/`, and `results/`. Adjust those values on each VM before running real data inspection, model loading, training, or evaluation.

## Script Entry Points

```bash
python scripts/inspect_globalclip_schema.py --config configs/vm.yaml
python scripts/smoke_test_parnet.py --config configs/vm.yaml
python scripts/train_reconstruction_head.py --config configs/vm.yaml --dry-run
python scripts/evaluate_reconstruction.py --config configs/vm.yaml
```

Current script behavior is intentionally lightweight:

- `inspect_globalclip_schema.py` will inspect configured globalCLIP metadata once the schema is finalized.
- `smoke_test_parnet.py` will verify that the pretrained Parnet checkpoint can produce 223 profiles for 600 nt windows.
- `train_reconstruction_head.py` will eventually train the reconstruction head on top of frozen or cached Parnet outputs.
- `evaluate_reconstruction.py` will evaluate reconstructed globalCLIP profiles against observed targets.

## Next Steps

1. Document the globalCLIP input schema in `docs/`.
2. Add a pinned Python environment file once package versions are known.
3. Implement schema validation and minimal data loading.
4. Wire the pretrained Parnet model into `smoke_test_parnet.py`.
5. Add the reconstruction-head model and a small training smoke test.

# GeoScaleInv Template

## Directory

```text
GeoScaleInv/
├── configs/
│   ├── default.yaml
│   └── _quick.yaml
├── models/
├── utils/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Model

- Model class: `GeoScaleInv` (`models/geoscaleinv.py`)
- Metrics: `RMSE`, `MAE`, `MAPE`, `R2`
- Inputs: `PE, DEN, AC, GR, RS, CNL, M2R1`
- Target: `TOC`

## Train

```bash
python train.py --config configs/default.yaml --data_path "YOUR_PRIVATE_DATA.xlsx"
```

Quick sanity run:

```bash
python train.py --config configs/_quick.yaml --data_path "YOUR_PRIVATE_DATA.xlsx"
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_GeoScaleInv.pt --data_path "YOUR_PRIVATE_DATA.xlsx"
```

## Data requirement

Your private table must include columns:

- `PE`, `DEN`, `AC`, `GR`, `RS`, `CNL`, `M2R1`, `TOC`


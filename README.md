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

## Publication
Paper describing this work has been received in Frontiers of Computer Science（FCS） special column “Code & Data in Earth Science”.

Cited as: Sibo Qiao, Jiankang Guo, Hengxiao Li, Min Wang. GeoScaleInv: A Cross-Attention Multi-scale State Space Framework

for Geophysical Inversion. Front. Comput. Sci., 2026, DOI: 10.1007/s11704-026-60838-w
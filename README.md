# HEA Fatigue Life Prediction — README

## 🏗️ Project Structure
```
Major Project/
├── hea_fatigue/
│   ├── data/loader.py              # Excel data loader + merger
│   ├── preprocessing/preprocessor.py  # Imputation, scaling, encoding
│   ├── feature_engineering/feature_engineer.py  # VEC, ΔSmix, ΔHmix
│   ├── models/
│   │   ├── weibull_model.py        # Weibull AFT
│   │   ├── cox_model.py            # Cox Proportional Hazards
│   │   ├── rsf_model.py            # Random Survival Forest
│   │   └── trainer.py              # Unified trainer + CV
│   ├── evaluation/evaluator.py     # C-index, Brier Score, calibration
│   └── visualization/plotter.py   # 7 interactive Plotly charts
├── api/server.py                   # FastAPI REST API
├── app.py                          # Streamlit interactive UI
├── main.py                         # End-to-end pipeline runner
├── requirements.txt
└── FatigueData-CMA2022.xlsx        # Real experimental dataset
```

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models (generates outputs/)
```bash
python main.py
```

### 3. Launch Streamlit UI
```bash
streamlit run app.py
```
→ Opens at http://localhost:8501

### 4. Start REST API
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
→ Swagger docs at http://localhost:8000/docs

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model_info` | Model metadata |
| POST | `/predict` | Survival probability at given stress & cycles |
| POST | `/sn_curve` | Full probabilistic S-N curve |
| POST | `/risk_map` | 2-D failure probability heatmap |

### Example: /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stress_amplitude": 300,
    "cycles": 1000000,
    "material_type": "MPEA",
    "composition": "CrMnFeCoNi"
  }'
```

## 📊 Outputs (after running main.py)

| File | Description |
|------|-------------|
| `outputs/dashboard.html` | Full interactive multi-panel dashboard |
| `outputs/survival_curves.html` | S(N) vs N for test samples |
| `outputs/sn_curves.html` | Probabilistic S-N bands (10/50/90%) |
| `outputs/feature_importance.html` | RSF feature importance |
| `outputs/risk_heatmap.html` | Failure probability heatmap |
| `outputs/brier_score.html` | Brier score over time |
| `outputs/model_comparison.html` | C-index comparison bar chart |
| `outputs/model_comparison.csv` | Numeric results table |
| `outputs/calibration_table.csv` | Calibration binning table |
| `hea_fatigue/models/saved/*.pkl` | Trained model files |

## 🧪 Models Trained

| Model | Method | Handles Censoring | Feature Importance |
|-------|--------|-------------------|-------------------|
| Weibull AFT | Parametric | ✅ | ❌ |
| Cox PH | Semi-parametric | ✅ | ❌ |
| **Random Survival Forest** | Non-parametric | ✅ | ✅ (Gini) |

## 📐 Thermodynamic Features Computed

- **VEC**: Valence Electron Concentration = Σ(ci × VECi)
- **ΔSmix**: Entropy of mixing = −R Σ(ci ln ci)  [J/mol·K]
- **ΔHmix**: Enthalpy of mixing (Miedema pairwise model)  [kJ/mol]
- **n_elements**: Number of principal alloying elements

## 📋 Dataset
- **Source**: FatigueData-CMA2022.xlsx
- **Total S-N data points**: 1,492
- **Experimental datasets**: 272
- **Material types**: Metallic Glass (MG), Multi-Principal Element Alloy (MPEA)
- **Runouts (censored)**: automatically detected via `runout` column

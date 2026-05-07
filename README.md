# Hybrid Reinforcement Learning for Personalized Diabetes Care

**MSc in Artificial Intelligence – National College of Ireland (2025)**  
**Author:** Shayshank Rathore

---

## Project Overview

This project develops a **Hybrid Reinforcement Learning (RL) framework** to support personalised diabetes care. It combines two core innovations:

- **Reward Decomposition** — splits the reward signal into clinically interpretable components: glycaemic control (Time-in-Range), hypoglycaemia penalties (safety), and treatment cost (insulin use burden)
- **Meta-Learning** — enables rapid per-patient adaptation using a Reptile-style fine-tuning loop on top of a PPO base model

A custom `gymnasium` environment was built on the **UCI AIM'94 Diabetes dataset** (70 patients). The primary RL agent is **Proximal Policy Optimization (PPO)**, benchmarked against a simple rule-based controller.

---

## Repository Structure

```
hybrid-rl-diabetes/
├── notebooks/
│   ├── thesis-code.ipynb       # Full MSc thesis pipeline (PPO + meta-learning + explainability)
│   ├── project-code.ipynb      # Kaggle-style project notebook
│   └── msc-project.ipynb       # Clean MSc submission version
├── data/
│   └── diabetes-data.zip       # UCI Diabetes dataset (70 patients)
├── output/
│   ├── all_outputs_20250829_002210.zip   # Saved model artefacts + plots
│   └── 1Output.zip
├── docs/
│   ├── MSc_Research_Project_Report_Shayshank_Rathore.pdf
│   └── MSc_Research_Project_Config_Manual_Shayshank_Rathore.pdf
├── requirements.txt
└── LICENSE
```

---

## How to Run

### Option A — Kaggle (recommended, GPU available)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create a new notebook and upload any of the `.ipynb` files from `notebooks/`
3. Add the `diabetes-data.zip` file as a dataset (attach under the **Input** panel)
4. Enable **GPU (T4 x2)** accelerator
5. Run all cells sequentially

### Option B — Local

```bash
git clone https://github.com/shayshankr/hybrid-rl-diabetes.git
cd hybrid-rl-diabetes
pip install -r requirements.txt
jupyter notebook notebooks/msc-project.ipynb   # or thesis-code.ipynb
```

The notebooks auto-detect the dataset: they will find and extract `data/diabetes-data.zip` automatically — no manual path changes needed.

---

## Environment Design

| Component | Detail |
|-----------|--------|
| **Dataset** | UCI AIM'94 Diabetes — 70 patients, tab-separated event logs |
| **Preprocessing** | Hourly resample, glucose forward-fill (6h), feature engineering (lags, rolling stats) |
| **State** | 10 z-scored features: glucose, lags, rolling means, insulin history, hour-of-day, day-of-week |
| **Actions** | Discrete: `0` = no dose, `1` = moderate (+2 units, −12 mg/dL), `2` = high (+6 units, −24 mg/dL) |
| **Reward** | `TIR (+1) + hypo (−1/−2) + hyper (−1) + cost (−0.05×units)` |
| **Split** | 80/20 patient-level train/test, nested 80/20 for val; no data leakage |

---

## Pipeline Summary

| Stage | Notebook cells | What it does |
|-------|---------------|--------------|
| Data loading | Cell 3–4 | Parse raw patient files, map insulin/glucose codes |
| Resampling | Cell 4A | Hourly resample, forward-fill glucose, sum insulin |
| Feature engineering | Cell 5 | Lags, rolling stats, patient-level train/val/test split |
| Environment | Cell 6 | `DiabetesEnv` — `gymnasium.Env` with decomposed reward |
| PPO training | Cell 8 | 30k timesteps, `EvalCallback`, early stopping on val |
| Evaluation | Cell 9 | TIR, hypo/hyper steps, insulin use, trajectory plots |
| Rule baseline | Cell 7 | Threshold controller for comparison |
| Meta-adaptation | Cell 10 | Per-patient fine-tuning, Δ return bar plot |
| Artifact export | Cell 11 | CSV/XLSX tables, YAML config, bundled zip |

---

## Key Results

On held-out test patients:

- **PPO vs rule baseline** — higher average return and TIR, fewer hypo events, ~86% less insulin per step
- **Meta-learning adaptation** — positive Δ return for most patients after ≤2k fine-tuning steps
- **Policy behaviour** — conservative dosing (mostly no/low and moderate actions), high-dose used sparingly

---

## Requirements

```
stable-baselines3==2.1.0
gymnasium==0.29.0
torch==2.0.0
numpy==1.25.2
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
shap==0.46.0
lime==0.2.0.1
```

See `requirements.txt` for the full pinned list.

---

## References

- [UCI AIM'94 Diabetes Dataset](https://doi.org/10.24432/C5T59G)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Farama Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- Rathore, S. *MSc Project Report: Hybrid Reinforcement Learning for Personalized Diabetes Care*, NCI, 2025

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by Shayshank Rathore** — BI Developer & Data Engineer, MSc AI (NCI Dublin)  
[LinkedIn](https://www.linkedin.com/in/shayshank-rathore/) · [GitHub](https://github.com/shayshankr)

# Hybrid Reinforcement Learning for Personalized Diabetes Care

**MSc in Artificial Intelligence – National College of Ireland (2025)**  
**Author:** Shayshank Rathore  


---

## 📖 Project Overview
This project develops a **Hybrid Reinforcement Learning (RL) framework** to support **personalized diabetes care**.  
It combines two core innovations:

- **Reward Decomposition** → Splits overall reward into clinically interpretable components:  
  - Glycaemic control (Time-in-Range)  
  - Hypoglycaemia penalties (safety)  
  - Treatment cost (insulin use burden)  

- **Meta-Learning** → Enables the RL agent to adapt quickly to new patient profiles using limited data, improving **sample efficiency** and personalisation.

A **custom Gymnasium environment** was built using the UCI AIM’94 Diabetes dataset to simulate patient trajectories and evaluate treatment decisions. The primary RL agent is **Proximal Policy Optimization (PPO)**, benchmarked against a simple **rule-based controller**.

---

## 🏗️ Repository Structure
```
hybrid-rl-diabetes/
│
├── notebooks/        # Implementation notebooks
│   ├── thesis-code.ipynb
│   ├── project-code.ipynb
│   ├── msc-project.ipynb
│
├── data/             # Dataset (UCI Diabetes – 70 patients)
│   └── diabetes-data.zip
│
├── outputs/          # Model artefacts
│   ├── all_outputs_20250829.zip
│   └── 1Output.zip
│
├── docs/             # Reports and manuals
│   ├── MSc_Research_Project_Report_Shayshank_Rathore.pdf
│   └── MSc_Research_Project_Config_Man.pdf
│
├── requirements.txt  # Python dependencies
├── LICENSE           # MIT License
└── .gitignore        # Ignore checkpoints, cache, outputs
```

---

## ⚙️ How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/<username>/hybrid-rl-diabetes.git
   cd hybrid-rl-diabetes
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks**
   - Upload to [Kaggle Notebooks](https://www.kaggle.com/code)
   - Attach the dataset (`diabetes-data.zip`)
   - Enable **GPU (T4)**
   - Execute notebooks top to bottom  

4. **Outputs**
   - Models → `outputs/models/`  
   - Logs & CSVs → `outputs/logs/`  
   - Plots → `outputs/plots/`  

Full setup details are available in the **Configuration Manual** under `/docs/`.

---

## 📊 Key Results
On held-out test patients:
- **PPO vs Rule-based baseline**  
  - ✅ Eliminated **hypoglycaemic steps** (0 vs 1)  
  - ✅ Reduced **insulin use by ~86%** (1.34 vs 9.72 units/step)  
  - **Meta-learning adaptation**:  
  - Small but consistent per-task reward gains (**∆ reward ≈ +2.28**)  
  - Suggests feasibility of rapid personalisation  

Overall, PPO learned a **conservative, safe, insulin-sparing policy**. Future iterations will focus on **finer action spaces**, **reward rebalancing**, and **stochastic dynamics**.

---

## 📌 Features
- ✅ Custom **Gymnasium** environment for diabetes treatment simulation  
- ✅ **Reward decomposition** for transparent decision-making  
- ✅ **Meta-learning** (Reptile-style fine-tuning) for per-patient adaptation  
- ✅ PPO vs Rule-based baseline comparison  
- ✅ Clinical + RL metrics: TIR, insulin use, hypo/hyper events, reward curves  

---

## 📚 References
- [UCI AIM’94 Diabetes Dataset](https://doi.org/10.24432/C5T59G)  
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)  
- [Farama Gymnasium](https://github.com/Farama-Foundation/Gymnasium)  
- Rathore, S. *MSc Project Report: Hybrid Reinforcement Learning for Personalized Diabetes Care*, NCI, 2025.  

---

## 📝 License
This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.  

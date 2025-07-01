# 🔧 AI-Powered Predictive Maintenance using NASA C-MAPSS Dataset

This project provides a full-fledged AI solution for predictive maintenance using time-series sensor data from industrial engines. It includes deep learning (LSTM), ensemble models (Random Forest and XGBoost), explainable AI (SHAP), and a real-time dashboard (Streamlit).

---

## 📁 Project Structure

```
ai-predictive-maintenance-nasa/
│
├── data/                        # Sample engine sensor data
│   └── sample_engine_data.csv
│
├── models/                      # Trained models will be saved here (after training)
│   ├── lstm_model.h5            ← 🔧 Generated after LSTM training
│   ├── rf_model.pkl             ← 🔧 Generated after RF training
│   └── xgb_model.pkl            ← 🔧 Generated after XGBoost training
│
├── notebooks/
│   └── predictive_maintenance_demo.ipynb
│
├── src/                         # Source code for training, evaluation, and dashboard
│   ├── utils.py
│   ├── feature_engineering.py
│   ├── train_lstm_model.py      ← 🔁 Option A: Modular LSTM training (recommended)
│   ├── train_rf_xgb.py
│   ├── evaluate_models.py
│   └── app.py
│
├── results/                     # Output plots (generated after evaluation)
│   ├── training_loss_plot.png
│   ├── predicted_vs_true_rul.png
│   ├── shap_summary_plot.png
│   └── feature_correlation_heatmap.png
│
├── train_lstm_local.py          ← ⚙️ Option B: Standalone script for quick LSTM training
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run This Project

> 🔰 This repo does **not include pretrained model files or result images**. You must generate them locally.

---

### 1️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Train the Models

#### 🔹 OPTION A: Train LSTM using integrated project script (Recommended)

```bash
python src/train_lstm_model.py
```
✅ Saves: `models/lstm_model.h5`  
🖼️ Also generates: `results/training_loss_plot.png`

---

#### 🔹 OPTION B: (Alternative) Train LSTM using standalone script

```bash
python train_lstm_local.py
```
✅ Also generates `lstm_model.h5` but without using `utils.py`

---

#### 🔹 Train Random Forest and XGBoost

```bash
python src/train_rf_xgb.py
```

---

### 3️⃣ Evaluate and Generate Plots

```bash
python src/evaluate_models.py
```
Generates:
- `results/predicted_vs_true_rul.png`
- `results/shap_summary_plot.png`

---

### 4️⃣ Launch the Dashboard

```bash
streamlit run src/app.py
```

Upload your CSV → View model predictions + SHAP explanations.

---

### 📓 Jupyter Notebook

To explore step-by-step training and evaluation:
```bash
notebooks/predictive_maintenance_demo.ipynb
```

---

## 🧠 Dataset Summary

Simulated data format (NASA C-MAPSS style):
- `engine_id`, `cycle`, `RUL`
- Sensor & operational readings

---

## 🎯 Use Cases

- Smart Factory Maintenance
- Industrial IoT Predictive Analytics
- Aerospace / Automotive Diagnostics

---

## 📃 License

MIT License © 2025 Shreyas Prakash

---

## 🙌 Final Notes

✅ No model files are included in the repo by default.  
👉 You must generate them using the training scripts above.

Happy modeling!

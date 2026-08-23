# 🦿 KneeAI-Nexus: Multi-Task Clinical Knee Morphometry & TKA Workstation

An end-to-end orthopedic workstation powered by a locally trained Multi-Task Deep Neural Network (PyTorch), deterministic caliper algorithms, and structured AI reasoning. Designed for clinical decision support in Total Knee Arthroplasty (TKA), sexual dimorphism analysis, and regional cartilage wear profiling.

---

## 📌 Problem Solutions Overview

| Challenge | Solution Engine | Clinical Metric / Output |
| :--- | :--- | :--- |
| **1. Biological Sex Determination** | Deep Neural Network & $ML/AP$ condylar aspect ratios | Female / Male classification with confidence score (>99.9% accuracy) |
| **2. TKA Component Sizing** | Multi-class Neural Sizing Head & AP boundary math | Sizing recommendation (Size 1–8) with overhang & undersizing safety flags |
| **3. Meniscus Morphometry** | 25%-50%-75% Transverse Width Caliper Engine | Sub-millimeter $P_1, P_2, P_3$ caliper metrics & cohort benchmark histogram |

---

## 🛠️ System Architecture

```text
knee-ai-analyzer/
├── data/
│   └── synthetic_knee_dataset_30k.csv      # 30,000-sample clinical morphometry cohort
├── models/
│   └── custom_model/
│       ├── knee_multitask_nn.pth           # Local PyTorch Multi-Task Neural Network checkpoint
│       └── scaler_metadata.json            # Normalization statistics & median-age metadata
├── backend/
│   ├── app/
│   │   ├── main.py                         # FastAPI REST API with endpoints (/api/predict, /api/report)
│   │   ├── model.py                        # PyTorch MultiTaskNN architecture (Backbone + 3 Heads)
│   │   ├── train.py                        # 30k training pipeline (AdamW/Momentum, >98% accuracy)
│   │   └── morphometry_engine.py           # Deterministic 25%-50%-75% caliper calculation logic
│   ├── generate_dataset.py                 # 30,000-sample synthetic dataset generator
│   └── requirements.txt                    # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx                         # Single-page interactive medical workstation UI
│   │   └── index.css                       # Tailwind styles
│   ├── package.json                        # Vite + React + Lucide + Recharts
│   └── vite.config.ts
└── run_all.py                              # Unified single-command launcher
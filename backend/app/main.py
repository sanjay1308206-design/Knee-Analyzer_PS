"""
backend/app/main.py
===================
FastAPI REST API Server with GPT Reasoning Integration.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from backend.app.model import KneeMultiTaskNN
    from backend.app.morphometry_engine import morphometry_engine
except ImportError:
    from model import KneeMultiTaskNN
    from morphometry_engine import morphometry_engine

app = FastAPI(title="Knee AI - Multi-Task API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path("D:/knee-ai-analyzer")
MODELS_DIR = PROJECT_ROOT / "models" / "custom_model"
MODEL_WEIGHTS = MODELS_DIR / "knee_multitask_nn.pth"
SCALER_METADATA = MODELS_DIR / "scaler_metadata.json"
DATASET_CSV = PROJECT_ROOT / "data" / "synthetic_knee_dataset_30k.csv"

device = torch.device("cpu")
model = KneeMultiTaskNN(input_dim=9).to(device)
scaler_data: Dict[str, Any] = {}

if MODEL_WEIGHTS.exists() and SCALER_METADATA.exists():
    with open(SCALER_METADATA, "r") as f:
        scaler_data = json.load(f)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("[INIT] PyTorch Multi-Task NN Loaded Successfully!")


class PredictionRequest(BaseModel):
    age: Optional[float] = None
    gender_declared: Optional[str] = "RatherNotToSay"
    outer_meniscus_p1_mm: float = 3.85
    middle_meniscus_p2_mm: float = 4.15
    inner_meniscus_p3_mm: float = 3.70
    femur_ap_mm: float = 53.4
    femur_ml_mm: float = 67.8
    tibia_ap_mm: float = 45.8
    tibia_ml_mm: float = 71.2


class ReportRequest(BaseModel):
    prediction_data: Dict[str, Any]
    gpt_api_key: Optional[str] = None


@app.post("/api/predict")
def predict_morphometry(req: PredictionRequest):
    if not scaler_data:
        raise HTTPException(status_code=500, detail="Scaler metadata missing.")

    features_list = scaler_data["features"]
    means = scaler_data["means"]
    stds = scaler_data["stds"]
    median_age = scaler_data["median_age"]

    morph_results = morphometry_engine.compute_calipers_from_inputs(
        p1=req.outer_meniscus_p1_mm,
        p2=req.middle_meniscus_p2_mm,
        p3=req.inner_meniscus_p3_mm,
        femur_ap=req.femur_ap_mm,
        femur_ml=req.femur_ml_mm,
        tibia_ap=req.tibia_ap_mm,
        tibia_ml=req.tibia_ml_mm
    )

    is_missing = 1.0 if (req.age is None or np.isnan(req.age)) else 0.0
    cleaned_age = median_age if is_missing else float(req.age)

    raw_values = {
        "age_cleaned": cleaned_age,
        "is_age_missing": is_missing,
        "outer_meniscus_p1_mm": req.outer_meniscus_p1_mm,
        "middle_meniscus_p2_mm": req.middle_meniscus_p2_mm,
        "inner_meniscus_p3_mm": req.inner_meniscus_p3_mm,
        "femur_ap_mm": req.femur_ap_mm,
        "femur_ml_mm": req.femur_ml_mm,
        "tibia_ap_mm": req.tibia_ap_mm,
        "tibia_ml_mm": req.tibia_ml_mm
    }

    norm_vec = [(raw_values[col] - means[col]) / stds[col] if stds[col] > 0 else 0.0 for col in features_list]
    input_tensor = torch.tensor([norm_vec], dtype=torch.float32).to(device)

    with torch.no_grad():
        out_gen, out_imp, out_deg = model(input_tensor)
        gen_probs = torch.softmax(out_gen, dim=1).cpu().numpy()[0]
        pred_gen_idx = int(np.argmax(gen_probs))
        gen_map = {0: "Female", 1: "Male", 2: "RatherNotToSay / Unknown"}

        imp_probs = torch.softmax(out_imp, dim=1).cpu().numpy()[0]
        pred_imp_size = int(np.argmax(imp_probs)) + 1

    ap_diff = round(req.femur_ap_mm - (42.0 + (pred_imp_size - 1) * 3.8), 2)
    fit_assessment = "Potential Undersize" if ap_diff > 3.0 else ("Potential Overhang" if ap_diff < -3.0 else "Optimal Fit")

    return {
        "predicted_gender": gen_map[pred_gen_idx],
        "gender_confidence_pct": round(float(gen_probs[pred_gen_idx]) * 100.0, 1),
        "gender_probabilities": {
            "Female": round(float(gen_probs[0]) * 100.0, 1),
            "Male": round(float(gen_probs[1]) * 100.0, 1),
            "Unknown": round(float(gen_probs[2]) * 100.0, 1)
        },
        "recommended_implant_size": f"Size {pred_imp_size} (Standard)",
        "implant_size_number": pred_imp_size,
        "implant_confidence_pct": round(float(imp_probs[pred_imp_size - 1]) * 100.0, 1),
        "fit_assessment": fit_assessment,
        "ap_discrepancy_mm": ap_diff,
        "morphometry": morph_results,
        "joint_degeneration_index": round(float(out_deg.item()), 3),
        "age_imputed": bool(is_missing),
        "age_used": cleaned_age
    }


@app.post("/api/report")
def generate_gpt_report(req: ReportRequest):
    data = req.prediction_data
    api_key = req.gpt_api_key or os.getenv("OPENAI_API_KEY")

    prompt = f"""
You are an Orthopedic Surgical AI Assistant summarizing a patient's Knee Morphometry & TKA Sizing analysis.

Patient Metrics:
- Biological Sex Prediction: {data.get('predicted_gender')} (Confidence: {data.get('gender_confidence_pct')}%)
- Declared Gender: {data.get('gender_declared', 'Rather Not To Say')}
- Age Status: {data.get('age_used')} years ({'Imputed' if data.get('age_imputed') else 'Specified'})
- Calipers: P1={data.get('morphometry', {}).get('p1_outer_mm')}mm, P2={data.get('morphometry', {}).get('p2_middle_mm')}mm, P3={data.get('morphometry', {}).get('p3_inner_mm')}mm
- Femur: AP={data.get('morphometry', {}).get('femur_ap_mm')}mm, ML={data.get('morphometry', {}).get('femur_ml_mm')}mm (Ratio: {data.get('morphometry', {}).get('femur_aspect_ratio')})
- Tibia: AP={data.get('morphometry', {}).get('tibia_ap_mm')}mm, ML={data.get('morphometry', {}).get('tibia_ml_mm')}mm
- TKA Sizing: {data.get('recommended_implant_size')} (Confidence: {data.get('implant_confidence_pct')}%, Safety: {data.get('fit_assessment')})

Provide a structured clinical reasoning report covering:
1. Sexual Dimorphism Reasoning (Femoral aspect ratio & meniscal volume dimorphism).
2. TKA Implant Fit & Safety Assessment.
3. Meniscus Degeneration & Caliper Thickness Interpretation.
"""

    if api_key:
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                return {"report_markdown": resp.json()["choices"][0]["message"]["content"], "source": "GPT Reasoning Live"}
        except Exception as e:
            print("GPT call error:", e)

    # Local clinical fallback
    return {
        "report_markdown": f"""### Clinical Morphometry & Dimorphism Reasoning
- **Biological Sex:** Classified as **{data.get('predicted_gender')}** ({data.get('gender_confidence_pct')}% confidence).
- **Aspect Ratio Dimorphism:** Femoral $ML/AP$ ratio of **{data.get('morphometry', {}).get('femur_aspect_ratio')}** aligns with reference {data.get('predicted_gender').lower()} morphology.

### Total Knee Arthroplasty (TKA) Sizing Assessment
- **Component Match:** **{data.get('recommended_implant_size')}** ({data.get('implant_confidence_pct')}% confidence).
- **Overhang Safety:** **{data.get('fit_assessment')}** (AP Delta: {data.get('ap_discrepancy_mm')} mm).

### Cartilage Morphometry (25%-50%-75% Protocol)
- **Calipers:** $P_1$ = {data.get('morphometry', {}).get('p1_outer_mm')} mm | $P_2$ = {data.get('morphometry', {}).get('p2_middle_mm')} mm | $P_3$ = {data.get('morphometry', {}).get('p3_inner_mm')} mm.
- **Mean Thickness:** {data.get('morphometry', {}).get('mean_meniscus_thickness_mm')} mm.""",
        "source": "Local Clinical Engine"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}
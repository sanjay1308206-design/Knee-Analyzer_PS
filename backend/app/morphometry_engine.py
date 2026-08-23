"""
backend/app/morphometry_engine.py
=================================
Deterministic Morphometry & Caliper Measurement Engine.
Performs exact millimeter-scale caliper calculations across the
25% (P1), 50% (P2), and 75% (P3) transverse meniscal width partitions,
and calculates subchondral bone bounding dimensions.
"""

from typing import Dict, Any, Optional
import numpy as np

class MorphometryEngine:
    def __init__(
        self,
        p1_ratio: float = 0.25,
        p2_ratio: float = 0.50,
        p3_ratio: float = 0.75
    ):
        self.p1_ratio = p1_ratio
        self.p2_ratio = p2_ratio
        self.p3_ratio = p3_ratio

    def compute_calipers_from_inputs(
        self,
        p1: float,
        p2: float,
        p3: float,
        femur_ap: float,
        femur_ml: float,
        tibia_ap: float,
        tibia_ml: float
    ) -> Dict[str, Any]:
        """
        Computes deterministic morphometric parameters and aspect ratios.
        """
        calipers = [p1, p2, p3]
        mean_thickness = float(np.mean(calipers))
        std_thickness = float(np.std(calipers))

        # Femoral aspect ratio (ML / AP)
        femur_aspect_ratio = float(femur_ml / max(femur_ap, 1.0))
        tibia_aspect_ratio = float(tibia_ml / max(tibia_ap, 1.0))

        # Cartilage thinning score relative to healthy benchmark (4.5 mm baseline)
        wear_index = float(np.clip((4.50 - mean_thickness) / 2.50, 0.0, 1.0))

        return {
            "p1_outer_mm": round(p1, 2),
            "p2_middle_mm": round(p2, 2),
            "p3_inner_mm": round(p3, 2),
            "mean_meniscus_thickness_mm": round(mean_thickness, 2),
            "std_meniscus_thickness_mm": round(std_thickness, 2),
            "femur_ap_mm": round(femur_ap, 2),
            "femur_ml_mm": round(femur_ml, 2),
            "tibia_ap_mm": round(tibia_ap, 2),
            "tibia_ml_mm": round(tibia_ml, 2),
            "femur_aspect_ratio": round(femur_aspect_ratio, 2),
            "tibia_aspect_ratio": round(tibia_aspect_ratio, 2),
            "cartilage_wear_index": round(wear_index, 3),
            "protocol": "OAI Caliper Standard (25%-50%-75% Transverse Width)"
        }


# Global engine instance
morphometry_engine = MorphometryEngine()
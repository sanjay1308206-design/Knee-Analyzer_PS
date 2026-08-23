"""
backend/app/model.py
====================
PyTorch Multi-Task Neural Network Architecture.
Handles three heads simultaneously:
1. Gender Classification (Female, Male, Unknown)
2. TKA Implant Sizing Classification (Size 1 to 8)
3. Cartilage Degeneration Index Regression (0.0 to 1.0)
"""

import torch
import torch.nn as nn

class KneeMultiTaskNN(nn.Module):
    def __init__(self, input_dim: int = 9):
        super().__init__()
        # Shared Deep Representation Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.Mish()
        )
        
        # Task 1: Gender Classification Head (3 classes)
        self.gender_head = nn.Linear(32, 3)
        
        # Task 2: Implant Sizing Head (8 size classes)
        self.implant_head = nn.Linear(32, 8)
        
        # Task 3: Degeneration Index Head (Continuous regression [0, 1])
        self.degen_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        gender_logits = self.gender_head(feat)
        implant_logits = self.implant_head(feat)
        degen_score = self.degen_head(feat)
        return gender_logits, implant_logits, degen_score
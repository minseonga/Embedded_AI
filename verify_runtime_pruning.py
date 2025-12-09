
import sys
import os
import torch
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "src"))

from hand_tracking.pipeline import HandTrackingPipeline

def check_model_sparsity(model, name):
    total_params = 0
    zero_params = 0
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight.detach().cpu().numpy()
            total_params += w.size
            zero_params += np.sum(w == 0)
    
    if total_params > 0:
        print(f"{name} Sparsity: {zero_params / total_params:.2%}")
    else:
        print(f"{name} Sparsity: N/A")

print("Initializing pipeline with prune_ratio=0.3...")
pipeline = HandTrackingPipeline(prune_ratio=0.3, precision="fp32")

print("\nChecking sparsity...")
check_model_sparsity(pipeline.detector.wrapper.model, "Detector")
check_model_sparsity(pipeline.landmark.wrapper.model, "Landmark")


import onnx
import numpy as np
import os
from onnx import numpy_helper

def check_sparsity(model_path):
    print(f"\nChecking sparsity for {model_path}...")
    try:
        model = onnx.load(model_path)
        total_params = 0
        zero_params = 0
        
        for init in model.graph.initializer:
            w = numpy_helper.to_array(init)
            total_params += w.size
            zero_params += np.sum(w == 0)
            
        if total_params == 0:
            print("No initializers found (weights might be in external data or inputs).")
            return

        sparsity = zero_params / total_params
        print(f"Total params: {total_params}")
        print(f"Zero params: {zero_params}")
        print(f"Sparsity: {sparsity:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

base_dir = "assets/models/onnx_models"
check_sparsity(os.path.join(base_dir, "palm_detector.onnx"))
check_sparsity(os.path.join(base_dir, "palm_detector_pruned30.onnx"))

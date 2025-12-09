
import torch
import onnxruntime as ort
import numpy as np

class OrtWrapper:
    """Wrapper for ONNX Runtime inference with PyTorch interface."""
    
    def __init__(self, onnx_path, device, precision="fp16"):
        self.device = device
        self.precision = precision
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Configure providers
        providers = ['CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        
        # Prioritize CUDA (Jetson/Linux)
        if 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
        # Prioritize CoreML (Mac)
        elif 'CoreMLExecutionProvider' in available_providers:
            providers.insert(0, 'CoreMLExecutionProvider')
            
        self.session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
        
        # Check if requested device matches available providers
        used_providers = self.session.get_providers()
        print(f"[OrtWrapper] Active providers: {used_providers}")
        
        if device.type == 'cuda' and 'CUDAExecutionProvider' not in used_providers:
             print(f"[OrtWrapper] WARN: CUDA requested but 'CUDAExecutionProvider' not active. Available: {available_providers}")
        elif device.type == 'mps' and 'CoreMLExecutionProvider' not in used_providers:
             print(f"[OrtWrapper] WARN: MPS requested but 'CoreMLExecutionProvider' not active. Available: {available_providers}")
            
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type  # e.g. 'tensor(float)'

    def __call__(self, x: torch.Tensor):
        # Convert PyTorch tensor to numpy
        x_np = x.detach().cpu().numpy()
        
        # Check expected input type and cast if necessary
        # 'tensor(float)' corresponds to float32
        if self.input_type == 'tensor(float)' and x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32)
        elif self.input_type == 'tensor(float16)' and x_np.dtype != np.float16:
            x_np = x_np.astype(np.float16)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: x_np})
        
        # Convert back to PyTorch tensors
        # Note: outputs is a list of numpy arrays
        # We return them as a list of tensors on the correct device
        torch_outs = [torch.from_numpy(o).to(self.device) for o in outputs]
        
        # If single output, return it directly (to match onnx2pytorch behavior if needed, 
        # but onnx2pytorch usually returns list/tuple for multiple outputs)
        if len(torch_outs) == 1:
            return torch_outs[0]
        return torch_outs

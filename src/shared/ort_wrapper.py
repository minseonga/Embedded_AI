
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
        if device.type == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
            
        self.session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
    def __call__(self, x: torch.Tensor):
        # Convert PyTorch tensor to numpy
        x_np = x.detach().cpu().numpy()
        
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

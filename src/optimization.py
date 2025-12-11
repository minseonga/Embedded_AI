
import torch
import torch_pruning as tp
from ultralytics import YOLO
import torch.nn as nn

class ModelPruner:
    def __init__(self, model_path, example_inputs=None):
        self.model_wrapper = YOLO(model_path)
        self.model = self.model_wrapper.model
        self.example_inputs = example_inputs if example_inputs is not None else torch.randn(1, 3, 640, 640)
        
    def prune(self, pruning_ratio=0.3, importance_method='l1'):
        """
        Apply structured pruning to the model.
        Args:
            pruning_ratio: Target pruning ratio globally (e.g. 0.3 for 30% pruning)
            importance_method: Method to calculate importance ('l1', 'l2', etc.)
        """
        print(f"Pruning model with ratio: {pruning_ratio}")
        
        # 1. Importance criteria
        if importance_method == 'l1':
            imp = tp.importance.MagnitudeImportance(p=1)
        else:
            raise NotImplementedError(f"Importance method {importance_method} not implemented")

        # 2. Pruner initialization
        # Use a global pruning ratio
        ignored_layers = []
        
        # YOLO specific: Ignore the Detect/Pose head roughly or let TP handle it
        # Inspecting the model (yolo11n-pose), the last layer is complex.
        # Let's try to prune everything possible but be careful with the head.
        
        for m in self.model.modules():
             if isinstance(m, torch.nn.modules.linear.Linear) and m.out_features == self.model.nc:
                 ignored_layers.append(m)
        
        # NOTE: YOLO11 might need 'root_module_types' configuration or dependency graph fix
        # But standard TP mostly works. The issue might be that 'iterative_steps=1' needs 'ch_sparsity' 
        # to be fully met. 
        
        pruner = tp.pruner.MagnitudePruner(
            self.model,
            example_inputs=self.example_inputs,
            importance=imp,
            iterative_steps=1,
            ch_sparsity=pruning_ratio, # Target sparsity
            ignored_layers=ignored_layers,
            root_module_types=[torch.nn.Conv2d, torch.nn.Linear]
        )

        # 3. Pruning
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        print(f"Before Pruning:  MACs={base_macs/1e9:.4f} G, Params={base_nparams/1e6:.4f} M")

        pruner.step()

        # 4. Cleanup & Validation
        # Fix implicit dependencies (often needed for YOLO's C2f bottlenecks etc)
        # torch_pruning handles most, but sometimes we need manual check.
        # However, for recent TP versions, step() handles it well.
        
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        print(f"After Pruning:   MACs={pruned_macs/1e9:.4f} G, Params={pruned_nparams/1e6:.4f} M")
        print(f"Reduction:       MACs={1 - pruned_macs/base_macs:.2%}, Params={1 - pruned_nparams/base_nparams:.2%}")

        return self.model_wrapper

    def save(self, save_path):
        # We need to save purely the state dict or the full model
        # Ultralytics has its own save mechanism but we modified the internal nn.Module
        # Re-wrapping might be tricky because the config (yaml) doesn't match the new structure.
        # It's safest to save the torch model directly or update the YOLO wrapper.
        
        # For simplicity in this context, we save the weights compatible with torch.load
        # But for Ultralytics optimization, we usually want to save as .pt that YOLO() can load.
        
        # The modified model structure needs to be serialized.
        torch.save(self.model, save_path)
        print(f"Saved pruned model to {save_path}")

def get_model_info(model, input_size=(1,3,640,640)):
    """Return MACs and Params"""
    example_inputs = torch.randn(input_size).to(next(model.parameters()).device)
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params

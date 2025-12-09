
import sys
import os
import torch
import torch_pruning as tp
sys.path.append(os.path.join(os.getcwd(), "src"))

from hand_tracking.pipeline import HandTrackingPipeline

def test_pruning():
    print("Initializing pipeline...")
    # Load unpruned model
    pipeline = HandTrackingPipeline(prune_ratio=0.0, precision="fp32")
    model = pipeline.detector.wrapper.model
    
    print("Original model loaded.")
    
    # Example input for dependency tracing
    example_inputs = torch.randn(1, 3, 256, 256)
    
    # Initialize pruner
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000: # Example ignore
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        ch_sparsity=0.3,
        ignored_layers=ignored_layers,
    )

    print("Pruner initialized. Stepping...")
    try:
        pruner.step()
        print("Pruning step successful!")
        print(f"Model after pruning: {model}")
    except Exception as e:
        print(f"Pruning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pruning()

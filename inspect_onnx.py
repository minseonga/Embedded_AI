
import onnx

import sys
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "assets/models/palm_detector_fp16.onnx"
model = onnx.load(model_path)

print(f"Inspecting {model_path}...")

print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {inp.type.tensor_type.shape}")

print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {out.type.tensor_type.shape}")

# Check for kernel_shape in Conv/ConvTranspose/MaxPool/AvgPool
print("\nChecking nodes for missing kernel_shape...")
relevant_ops = {"Conv", "ConvTranspose", "MaxPool", "AvgPool"}
for node in model.graph.node:
    if node.op_type in relevant_ops:
        attr_names = [attr.name for attr in node.attribute]
        has_kernel_shape = "kernel_shape" in attr_names
        print(f"Node {node.name} ({node.op_type}) has kernel_shape: {has_kernel_shape}")
        if not has_kernel_shape:
            print(f"  WARNING: Node {node.name} is missing kernel_shape!")
    
    if node.op_type == "MaxPool":
        for attr in node.attribute:
            if attr.name == "storage_order":
                print(f"  Node {node.name} has storage_order: {attr.i}")

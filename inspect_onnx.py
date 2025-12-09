
import onnx

model_path = "assets/models/palm_detector_fp16.onnx"
model = onnx.load(model_path)

print(f"Inspecting {model_path}...")
relevant_ops = {"Conv", "ConvTranspose", "MaxPool", "AvgPool"}
for node in model.graph.node:
    if node.op_type in relevant_ops:
        attr_names = [attr.name for attr in node.attribute]
        has_kernel_shape = "kernel_shape" in attr_names
        print(f"Node {node.name} ({node.op_type}) has kernel_shape: {has_kernel_shape}")
        if not has_kernel_shape:
            print(f"  WARNING: Node {node.name} is missing kernel_shape!")

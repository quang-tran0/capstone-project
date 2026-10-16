import onnx
from onnx import numpy_helper

def load_initializers(path):
    graph = onnx.load(str(path), load_external_data=False).graph

    tensors = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    # Preserve ONNX names and add readable aliases; arrays are shared, not copied.
    for node in graph.node:
        if (
            node.op_type == "MatMul"
            and node.input[1] in tensors
            and node.name.startswith("/encoder/layer.")
        ):
            name = node.name.strip("/").removesuffix("/MatMul").replace("/", ".")
            tensors[f"{name}.weight"] = tensors[node.input[1]]
    return tensors

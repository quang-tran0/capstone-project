import onnx
from onnx import numpy_helper

def load_initializers(path):
    graph = onnx.load(str(path), load_external_data=False).graph

    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }

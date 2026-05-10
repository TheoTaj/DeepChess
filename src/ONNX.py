import onnxruntime as ort
import numpy as np
import torch

class ONNXModelWrapper:
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        """Allow the wrapper to be called like a function, passing the input tensor."""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()        
        outputs = self.session.run(None, {self.input_name: x})
        return torch.tensor(outputs[0])
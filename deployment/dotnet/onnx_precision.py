"""Export the stable inference body without an unsupported double Softplus op."""
from __future__ import annotations

import torch
from torch.onnx import symbolic_helper


def stable_session_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_prepacking", "1")
    return options


def _softplus_float64(g, value, beta, threshold):
    """Preserve PyTorch's linear threshold and use stable float64 primitives."""
    beta_value = symbolic_helper._get_const(beta, "f", "beta")
    threshold_value = symbolic_helper._get_const(threshold, "f", "threshold")

    def constant(number):
        return g.op("Constant", value_t=torch.tensor(float(number), dtype=torch.float64))

    scaled = g.op("Mul", value, constant(beta_value))
    # max(z,0)+log(1+exp(-abs(z))) avoids overflow in the unselected branch.
    stable = g.op("Add", g.op("Max", scaled, constant(0)),
                  g.op("Log", g.op("Add", constant(1), g.op("Exp", g.op("Neg", g.op("Abs", scaled))))))
    return g.op("Where", g.op("Greater", scaled, constant(threshold_value)),
                value, g.op("Div", stable, constant(beta_value)))


def export_stable_model(model, inputs, output, **kwargs):
    # The CPU ONNX Runtime Softplus kernel does not accept double. Keep this
    # export-only override local in lifetime; do not modify PyTorch forward.
    torch.onnx.register_custom_op_symbolic("aten::softplus", _softplus_float64, 17)
    try:
        torch.onnx.export(model, inputs, output, **kwargs)
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::softplus", 17)

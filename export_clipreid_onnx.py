"""
Export the trained CLIP-ReID (ViT-B/16) checkpoint to ONNX for use as the
DeepStream secondary-gie (see ds_include/sgie_config_clipreid.yml).

Run under the py310 conda env (has torch/timm/ftfy/regex/yacs/onnx/onnxruntime):
    /home/lakshh/miniconda3/envs/py310/bin/python export_clipreid_onnx.py

No custom export wrapper is needed: build_transformer.forward() (see
model/make_model_clipreid.py in the CLIP-ReID repo) already returns exactly
torch.cat([img_feature, img_feature_proj], dim=1) - 768 + 512 = 1280-d - on its
eval-mode path (cfg.TEST.NECK_FEAT='before' for this checkpoint), since the
classifier/text-encoder branches only run when self.training is True. Exporting
the model directly in eval() mode traces only that path.

num_class=751 is required to match the checkpoint's own classifier.weight shape
(751, 768) for load_param()'s raw, shape-strict state_dict copy to succeed - the
classifier itself never reaches the exported graph either way. camera_num/view_num
are inert here: MODEL.SIE_CAMERA/SIE_VIEW are both False for this checkpoint
(commented out in configs/person/vit_clipreid.yml), so build_transformer.__init__
never even constructs self.cv_embed.
"""

import sys

CLIP_REID_ROOT = "/home/lakshh/workspace/reid/CLIP-ReID"
if CLIP_REID_ROOT not in sys.path:
    sys.path.append(CLIP_REID_ROOT)

import numpy as np
import onnxruntime
import torch

from config import cfg
from model.make_model_clipreid import make_model

CHECKPOINT = f"{CLIP_REID_ROOT}/runs/ViT-B-16_60.pth"
CONFIG_FILE = f"{CLIP_REID_ROOT}/configs/person/vit_clipreid.yml"
ONNX_OUT = "ds_include/onnx_model/clipreid_vitb16_market1501.onnx"


def build_model() -> torch.nn.Module:
    cfg.merge_from_file(CONFIG_FILE)
    cfg.freeze()
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    model.load_param(CHECKPOINT)
    # load_clip_to_cpu() (in the CLIP-ReID repo) moves the freshly-built CLIP
    # architecture to CUDA before load_param() overwrites its weights - pull it
    # back to CPU here so export/onnxruntime verification don't need a GPU.
    model = model.cpu()
    model.eval()
    return model


def export(model: torch.nn.Module, out_path: str, dummy: torch.Tensor) -> None:
    # The legacy TorchScript-based exporter (torch.onnx.export's default) fails
    # two different ways on this model: it can't trace build_transformer's
    # `get_text`/`get_image` Python-bool branches without producing a bogus
    # aten::eq(Tensor, bool) op, and separately chokes on nn.MultiheadAttention's
    # internal reshape/transpose sequence ("transpose for tensor of unknown
    # rank") even once that's worked around via a pre-trace. The dynamo-based
    # exporter (torch.export under the hood) captures the actual executed graph
    # from concrete tensor shapes instead of symbolic tracing, and handles both
    # cleanly.
    # dynamic_axes (the legacy-exporter kwarg) is silently ignored under
    # dynamo=True - dynamic_shapes (torch.export's own format) is what actually
    # controls this path, and must be shaped like the `args` tuple passed below
    # (one entry per positional arg actually supplied, not the full forward()
    # signature).
    # Exporting with an example batch of 1 lets torch.export specialize the
    # batch dim away entirely (1 is degenerate for broadcasting/reshape
    # branches), even with dynamic_shapes given - trace with a batch >1 sample
    # instead so the dynamic dim actually sticks.
    export_dummy = torch.randn(2, *dummy.shape[1:])
    batch = torch.export.Dim("batch", min=1, max=128)
    torch.onnx.export(
        model, (export_dummy,), out_path,
        input_names=["input"], output_names=["output"],
        dynamic_shapes=({0: batch},),
        opset_version=18, dynamo=True,
    )
    print(f"Exported ONNX to {out_path}")


def verify(model: torch.nn.Module, onnx_path: str, dummy: torch.Tensor) -> None:
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]

    assert onnx_out.shape == (dummy.shape[0], 1280), f"unexpected output shape {onnx_out.shape}"
    close = np.allclose(torch_out, onnx_out, atol=1e-3)
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"output shape: {onnx_out.shape}  max_abs_diff: {max_diff:.6f}  "
          f"parity: {'PASS' if close else 'FAIL'}")
    assert close, "ONNX output diverges from PyTorch output beyond tolerance"


def main():
    model = build_model()
    dummy = torch.randn(1, 3, 256, 128)
    export(model, ONNX_OUT, dummy)
    verify(model, ONNX_OUT, dummy)


if __name__ == "__main__":
    main()

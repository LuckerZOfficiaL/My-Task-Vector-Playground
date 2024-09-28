import logging
import os
import pickle
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import ListConfig
from pytorch_lightning import Callback

pylogger = logging.getLogger(__name__)


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


def print_params_summary(model: torch.nn.Module):
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, ({sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()) * 100}%)"
    )


def print_mask_summary(model: torch.nn.Module, mask_mode):
    pct_masked = {}

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        pct_masked[name] = (
            round(1.0 - torch.mean(module.weight_mask).item(), 2) if hasattr(module, "weight_mask") else 0.0
        )

    print("Percentage of masked weights in each layer:")
    from rich import pretty

    pretty.pprint(pct_masked)

    # TODO remove this when one of the two (basically equivalent) methods, when a choice has been taken
    if mask_mode == "by_layer":
        print(
            f"% of masked weights across the entire network: {round(torch.as_tensor(data=list(pct_masked.values()), dtype=torch.float32).mean().item(), 2)}"
        )
    elif mask_mode == "by_nn":
        print(
            f"Sum percentage of masked weights across the entire network: {round(torch.as_tensor(data=list(pct_masked.values()), dtype=torch.float32).mean().item(), 2)}"
        )


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks

import torch

def clip_vector_norm_(vector: torch.Tensor, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    """
    Clip the norm of a 1D tensor (vector) to the specified max_norm.

    Args:
        vector (Tensor): the 1D tensor that will have its norm clipped.
        max_norm (float): max norm of the vector.
        norm_type (float): type of the used p-norm. Can be 'inf' for infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the norm of the vector
                                   is nan, inf, or -inf. Default: False.

    Returns:
        Total norm of the original vector before clipping.
    """
    if not isinstance(vector, torch.Tensor):
        raise ValueError("The input `vector` must be a torch.Tensor.")
    if vector.dim() != 1:
        raise ValueError("The input `vector` must be a 1D tensor.")

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Compute the norm of the vector
    total_norm = torch.linalg.vector_norm(vector, norm_type)

    # Check if the norm is non-finite (nan, inf, -inf)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The norm of order {norm_type} for the input `vector` is non-finite, so it cannot be clipped. '
            'To disable this error and scale the vector by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`.')

    # Compute the clip coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Scale the vector in place
    vector.mul_(clip_coef_clamped)

    return total_norm

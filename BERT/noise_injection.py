# noise_injection.py
# Temperature-aware "fault injection" for ReRAM CiM drift (PHANTOM-style).
#
# Usage:
#   noisy = make_noisy_model_for_temperature(model, T_tile=360.0)
#   # attach LoRA to `noisy` and train LoRA params only
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, Set
import torch
import torch.nn as nn
# ----------------------------
# Beta(T) from PHANTOM Eq.(1)
# ----------------------------
def beta_phantom_eq1(
    T: Union[float, torch.Tensor],
    T_ref: float = 300.0,
    onoff_ref: float = 20.0,
    slope_onoff_per_K: float = -0.1,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Union[float, torch.Tensor]:
    """
    Compute beta(T) = G_on(T) / G_on(T_ref) using PHANTOM Eq.(1):
        G_on(T) = (slope_onoff_per_K*(T - T_ref) + onoff_ref) * G_off
    With G_off treated as invariant, beta(T) = onoff(T)/onoff_ref.
    Default parameters implement the paper's linear ON/OFF drop:
      ON/OFF: 20 at 300K -> 10 at 400K  (i.e., 50% drop).
    clamp:
      If not None, clamps beta into [clamp[0], clamp[1]] to avoid negative beta
      for out-of-range temperatures.
    """
    # onoff(T) = slope*(T - T_ref) + onoff_ref
    onoff_T = slope_onoff_per_K * (T - T_ref) + onoff_ref
    beta = onoff_T / onoff_ref
    if clamp is not None:
        lo, hi = clamp
        if isinstance(beta, torch.Tensor):
            beta = beta.clamp(min=lo, max=hi)
        else:
            beta = max(lo, min(hi, float(beta)))
    return beta
# ----------------------------
# Injection configuration
# ----------------------------
@dataclass(frozen=True)
class InjectionConfig:
    # Which modules to perturb (default: Linear + Conv2d).
    inject_linear: bool = True
    inject_conv2d: bool = True
    # Optional relative noise (Gaussian) to emulate additional analog error.
    # If sigma_rel=0.0, you get pure multiplicative drift: w <- beta*w
    # If sigma_rel>0: w <- beta*w + N(0, (sigma_rel*(1-beta)*|w|)^2)
    sigma_rel: float = 0.0
    # If True, bias is also scaled/noised; often you keep bias digital => False.
    inject_bias: bool = False
    # Clamp beta to avoid negative/unphysical scaling outside [300,400]K
    beta_clamp: Optional[Tuple[float, float]] = (0.0, 1.0)
    # If you only want to inject a subset of module names, specify allowed prefixes.
    # Example: {"layer1.", "layer2."} or {"encoder.", "backbone."}
    allowed_name_prefixes: Optional[Set[str]] = None
# ----------------------------
# Core: create noisy model
# ----------------------------
def make_noisy_model_for_temperature(
    model: nn.Module,
    T_tile: float,
    config: InjectionConfig = InjectionConfig(),
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Returns a deep-copied model with temperature-aware weight degradation applied.
    All base parameters are frozen (requires_grad=False) so you can attach LoRA
    and train LoRA only.
    Notes:
    - This is "static" for a given T_tile: once created, weights are fixed.
    - If you want dynamic per-forward injection, don't bake it into weights;
      instead inject in the forward path.
    Parameters:
      model: pretrained model (nn.Module)
      T_tile: temperature in Kelvin (e.g., 300, 320, ..., 400)
      config: InjectionConfig
      device/dtype: optional casts for the returned model
    """
    noisy_model = copy.deepcopy(model)
    if device is not None:
        noisy_model = noisy_model.to(device)
    if dtype is not None:
        noisy_model = noisy_model.to(dtype)
    # Freeze everything; LoRA will re-enable grads on adapter params later.
    for p in noisy_model.parameters():
        p.requires_grad_(False)
    # Compute beta(T_tile) from PHANTOM Eq.(1) model.
    beta = beta_phantom_eq1(T_tile, clamp=config.beta_clamp)
    # Apply injection module-by-module.
    with torch.no_grad():
        for name, m in noisy_model.named_modules():
            if config.allowed_name_prefixes is not None:
                if not any(name.startswith(pref) for pref in config.allowed_name_prefixes):
                    continue
            if config.inject_linear and isinstance(m, nn.Linear):
                _inject_into_param(m, "weight", beta, sigma_rel=config.sigma_rel)
                if config.inject_bias and m.bias is not None:
                    _inject_into_param(m, "bias", beta, sigma_rel=config.sigma_rel)
            if config.inject_conv2d and isinstance(m, nn.Conv2d):
                _inject_into_param(m, "weight", beta, sigma_rel=config.sigma_rel)
                if config.inject_bias and m.bias is not None:
                    _inject_into_param(m, "bias", beta, sigma_rel=config.sigma_rel)
    # Helpful metadata for logging.
    noisy_model._injection_temperature_K = float(T_tile)  # type: ignore[attr-defined]
    noisy_model._injection_beta = float(beta)             # type: ignore[attr-defined]
    return noisy_model
def _inject_into_param(
    module: nn.Module,
    param_name: str,
    beta: float,
    *,
    sigma_rel: float,
) -> None:
    """
    In-place inject: param <- beta*param + optional noise.
    Preserves sign automatically for beta>0 (same as explicit pos/neg handling).
    """
    p = getattr(module, param_name, None)
    if p is None:
        return
    if not isinstance(p, torch.Tensor):
        return
    w = p.data
    # Multiplicative drift (temperature-induced attenuation)
    w_new = w * beta
    # Optional additional stochastic noise (proxy for analog non-idealities).
    if sigma_rel and sigma_rel > 0.0:
        # Scale noise with (1-beta)*|w| so it grows as drift worsens and follows magnitude.
        # You can swap this for a different noise law if your simulator suggests one.
        std = (sigma_rel * (1.0 - beta)) * w.abs()
        noise = torch.randn_like(w_new) * std
        w_new = w_new + noise
    w.copy_(w_new)

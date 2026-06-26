from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from fps_uda.io.config import normalize_config_keys


@dataclass
class FPSConfig:
    num_classes: int
    feature_dim: Optional[int] = None
    device: str = "cuda"
    optimizer: str = "sgd"
    base_lr: Optional[float] = None
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    lr_schedule: str = "linear_step"
    min_lr: float = 0.0
    iter_num: int = 36000
    alpha: float = 1.0
    beta: float = 0.1
    alpha_0: float = 0.1
    beta_0: float = 0.1
    lambda_lcr: Optional[float] = None
    randn_ratio: Optional[float] = None
    schedule_tau: float = 1000.0
    seed: int = 0
    src_sample_ratio: float = 1.0
    target_sample_ratio: float = 1.0
    normalize: str = "cross_norm"
    cross_norm_scale: float = 2.5
    cross_norm_target_weight: float = 0.5
    self_norm_scale_src: float = 1.0
    self_norm_scale_tgt: float = 1.0
    use_consistency_loss: bool = True
    use_correction: bool = False
    use_shift_constraint: bool = False
    ldelta_weight: float = 0.1
    ldelta_decay_steps: float = 500.0
    use_classes_weight: bool = True
    dynamic_parameters: bool = True
    baseline: bool = False
    use_lse: bool = True
    use_lce: bool = True
    use_lcr: bool = True
    lcr_loss: str = "mse"
    lcr_sample_weight: str = "paper"
    pseudo_margin: bool = True
    use_temperature: bool = False
    begin_temperature: float = 30.0
    use_geom_test: bool = False
    geom_alpha: float = 1.0
    geom_beta: float = 1.0
    sparse_weight_a: Union[float, str] = 5.0
    multi_class: bool = True
    eval_interval: int = 100
    progress: bool = True
    sample_entropy_type: str = "shannon"
    tsallis_q: float = 1.5
    legacy_loss_mode: bool = False
    adaptive_temp_margin_theta: float = 0.15
    adaptive_temp_margin_tau: float = 0.05
    adaptive_temp_t_min: float = 0.7
    adaptive_temp_t_max: float = 1.8
    margin_start_step: int = 100
    margin_weight_update_interval: int = 1
    margin_quantile: float = 0.15
    margin_weight_type: str = "normalized_logit_gap"
    margin_convert_mode: str = "quantile_sigmoid"
    margin_score_coef: float = 1.0
    margin_sigmoid_tau: float = 0.2
    margin_sigmoid_boundary_weight: float = 0.2
    margin_sharp_tau: float = 0.06
    output_dir: Optional[str] = None

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any], **overrides: Any) -> "FPSConfig":
        data = normalize_config_keys(dict(mapping))
        data.update({k: v for k, v in overrides.items() if v is not None})
        if "A" in data and "sparse_weight_a" not in data:
            data["sparse_weight_a"] = data.pop("A")
        if "lambda_lcr" not in data and "randn_ratio" in data:
            data["lambda_lcr"] = data["randn_ratio"]
        if "adamw_betas" in data:
            betas = tuple(float(value) for value in data["adamw_betas"])
            if len(betas) != 2:
                raise ValueError("adamw_betas must contain exactly two values.")
            data["adamw_betas"] = betas
        if data.get("use_shift_constraint"):
            data["use_correction"] = True
        known = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def validate_device(self) -> torch.device:
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
        return device

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import logging
import warnings

from .custom_optim import AVAI_OPTIMS, RAdam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print = logger.info


def build_optimizer(
    model,
    optimizer_name: str,
    lr: float,
    weight_decay=0.0,
    momentum=0.0,
    sgd_dampening=0.0,
    sgd_nesterov=False,
    rmsprop_alpha=0.99,
    adam_beta1=0.9,
    adam_beta2=0.999,
    staged_lr=None,
    lookahead=False,
):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """

    if optimizer_name not in AVAI_OPTIMS:
        raise ValueError(f"Unsupported optim: {optimizer_name}. Must be one of {AVAI_OPTIMS}")

    param_groups = model
    if optimizer_name == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise ValueError(f"optim: {optimizer_name} not in {AVAI_OPTIMS}")

    return optimizer


def build_staged_lr_param_groups(model, lr, new_layers, base_lr_mult, new_lr_mult):
    if isinstance(new_layers, list) and len(new_layers) == 0:
        warnings.warn("new_layers is empty, therefore staged lr uses only the base group")
    if isinstance(new_layers, str):
        new_layers = [new_layers]

    base_params = []
    new_params = []
    for name, params in model.named_parameters():
        if any(name.startswith(prefix) for prefix in new_layers):
            new_params.append(params)
        else:
            base_params.append(params)

    return [
        {
            "params": base_params,
            "lr": lr * base_lr_mult,
            "init_lr": lr * base_lr_mult,
            "name": f"{model.__class__.__name__}_base",
        },
        {
            "params": new_params,
            "lr": lr * new_lr_mult,
            "init_lr": lr * new_lr_mult,
            "name": f"{model.__class__.__name__}_new",
        },
    ]

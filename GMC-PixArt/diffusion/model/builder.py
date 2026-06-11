"""Minimal model registry (inference-only, no mmcv)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from diffusion.model.utils import set_grad_checkpoint


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Type] = {}

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Type] = None,
    ):
        def _register(cls: Type) -> Type:
            key = name or cls.__name__
            if key in self._module_dict and not force:
                raise KeyError(f'{key} is already registered in {self.name}')
            self._module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg: Any, default_args: Optional[dict] = None) -> Any:
        if isinstance(cfg, str):
            cfg = dict(type=cfg)
        args = dict(cfg)
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self._module_dict[obj_type]
        else:
            obj_cls = obj_type
        if default_args:
            args.update(default_args)
        return obj_cls(**args)


MODELS = Registry('models')


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, use_fp32_attention=use_fp32_attention, gc_step=gc_step)
    return model

import os
import time
from typing import Any, Dict, Optional, Type

from torch.utils.data import DataLoader

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Type] = {}

    def register_module(self, name: Optional[str] = None, force: bool = False, module: Optional[Type] = None):
        def _register(cls: Type) -> Type:
            key = name or cls.__name__
            if key in self._module_dict and not force:
                raise KeyError(f'{key} is already registered in {self.name}')
            self._module_dict[key] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register


def build_from_cfg(cfg: dict, registry: Registry, default_args: Optional[dict] = None) -> Any:
    args = dict(cfg)
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry._module_dict[obj_type]
    else:
        obj_cls = obj_type
    if default_args:
        args.update(default_args)
    return obj_cls(**args)


DATASETS = Registry('datasets')

DATA_ROOT = '/cache/data'


def set_data_root(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)


def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get('type')
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    transform = cfg.pop('transform', 'default_train')
    transform = get_transform(transform, resolution)
    dataset = build_from_cfg(cfg, DATASETS, default_args=dict(transform=transform, resolution=resolution, **kwargs))
    logger.info(f"Dataset {dataset_type} constructed. time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}")
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    return (
        DataLoader(
            dataset,
            batch_sampler=kwargs['batch_sampler'],
            num_workers=num_workers,
            pin_memory=True,
        )
        if 'batch_sampler' in kwargs
        else DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    )

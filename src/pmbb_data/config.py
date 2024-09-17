#!/usr/bin/python3
"""
PMBB dataset configuration implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania.
"""
from __future__ import annotations
import torch
import yaml
from pathlib import Path
from typing import (
    Callable, Final, NamedTuple, Optional, Sequence, Tuple, Type, Union
)


class PMBBVisionConfig(NamedTuple):
    img_shape: Optional[Tuple[int]] = None


class PMBBLanguageConfig(NamedTuple):
    tokenizer: Union[str, Callable[[str, torch.Tensor]]]


class PMBBConfig:
    def __init__(
        self,
        modalities: Optional[Sequence[str]] = None,
        body_parts_examined: Optional[Sequence[str]] = None,
        vision: Optional[PMBBVisionConfig] = None,
        language: Optional[PMBBLanguageConfig] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            modalities: a list of imaging modalities to filter by.
            body_parts_examiend: a list of body parts examined to filter by.
            vision: an optional Vision configuration object. Must be specified
                if vision data is to be used.
            language: an optional Language configuration object. Must be
                specified if language data is to be used.
            seed: optional random seed.
        """
        self.modalities: Final = modalities
        self.body_parts_examined: Final = body_parts_examined
        self.vision: Final = vision
        self.language: Final = language
        self.seed: Final = seed
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_yaml(cls: Type[PMBBConfig], fn: Union[Path, str]) -> PMBBConfig:
        """
        Loads a PMBB configuration object from a YAML file.
        Input:
            cls: the PMBB class.
            fn: the YAML file to load from.
        Returns:
            The loaded PMBB configuration object.
        """
        with open(fn, "r") as f:
            config = yaml.safe_load(f)
        vision = config.pop("vision", None)
        if vision is not None:
            vision = PMBBVisionConfig(**vision)
        language = config.pop("language", None)
        if language is not None:
            language = PMBBLanguageConfig(**language)
        modalities = config.pop("modalities", None)
        body_parts_examined = config.pop("body_parts_examined", None)
        seed = config.pop("seed", None)
        return PMBBConfig(
            modalities=modalities,
            body_parts_examined=body_parts_examined,
            vision=vision,
            language=language,
            seed=seed,
            **config
        )

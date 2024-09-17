#!/usr/bin/python3
"""
DICOM Series base class implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import cv2
import json
import nibabel as nib
import numpy as np
import os
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .base import PMBBObject


class Series(PMBBObject):
    metadata_ext: str = ".json"

    series_ext: str = ".nii.gz"

    def __init__(
        self,
        fn: Union[Path, str],
        img_shape: Optional[Tuple[int]] = None,
        return_tensors: str = "pt",
        **kwargs
    ):
        """
        Args:
            fn: the filepath of the study folder.
            img_shape: an optional HW image shape of the series.
            return_tensors: the type of tensor to return. One of ['pt', 'np'].
        """
        kwargs["img_shape"] = img_shape
        kwargs["return_tensors"] = return_tensors
        super().__init__(fn, **kwargs)

    @staticmethod
    def get_metadata_from_filepath(
        filepath: Union[Path, str]
    ) -> Dict[str, str]:
        """
        Returns the metadata from the expected filepath.
        Input:
            filepath: the filepath of the series in the PMBB subjects dataset.
        Returns:
            The metadata extracted from the filepath.
        """
        *_, pmbb_id, study_id, series_name = filepath.strip().split("/")
        return {
            "pmbb_id": pmbb_id,
            "study_id": study_id,
            "series_name": series_name
        }

    @property
    def study_id(self) -> str:
        """
        Returns the study ID of the parent study when the series was acquired.
        Input:
            None.
        Returns:
            The study ID of the parent study when the series was acquired.
        """
        return self._metadata["study_id"]

    @property
    def name(self) -> str:
        """
        Returns the name of the series.
        Input:
            None.
        Returns:
            The name of the series.
        """
        return self._metadata["series_name"]

    @property
    def nifti_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Returns the imaging study metadata associated with the series.
        Input:
            None.
        Returns:
            The imaging study metadata associated with the series.
        """
        fn = filter(
            lambda x: x.endswith(self.metadata_ext), os.listdir(self._fn)
        )
        try:
            with open(
                os.path.join(self._fn, next(fn)),
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                metadata = "\n".join(f.readlines())
        except StopIteration:
            return None
        return json.loads(metadata.replace("+", ""))

    @property
    def nifti(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Returns the imaging study data associated with the series.
        Input:
            None.
        Returns:
            The imaging study data associated with the series.
        """
        fn = filter(
            lambda x: x.endswith(self.series_ext), os.listdir(self._fn)
        )
        try:
            data = nib.load(os.path.join(self._fn, next(fn)))
        except StopIteration:
            return None
        data = data.get_fdata().squeeze()
        # TODO: Is this the correct way of dealing with 4D data?
        if data.ndim == 4:
            data = data.reshape(*data.shape[:2], -1)
        if self.img_shape is not None:
            data = cv2.resize(
                data,
                dsize=self.img_shape,
                interpolation=cv2.INTER_CUBIC
            )
        if self.return_tensors == "pt":
            return torch.from_numpy(data)
        elif self.return_tensors == "np":
            return data
        else:
            raise NotImplementedError(
                f"Unrecognized tensor type {self.return_tensors}"
            )

    @staticmethod
    def collate_fn(series: Series) -> Union[np.ndarray[torch.Tensor]]:
        data = []
        for s in series:
            img = s.nifti
            if img is None:
                continue
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            data.append(img)
        return torch.cat(data, dim=-1)

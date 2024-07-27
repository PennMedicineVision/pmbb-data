#!/usr/bin/env python3
"""
Defines the base object implementations in the PMBB.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import nibabel as nib
import os
import torch
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, NamedTuple, Optional, Sequence, Union


class Study:
    nifti_suffixes: Sequence[str] = [".nii", ".nii.gz"]

    def __init__(
        self,
        pmbb_id: str,
        nifti: Union[Path, str],
        report: Optional[Union[Path, str]] = None,
    ):
        """
        Args:
            pmbb_id: the PMBB ID associated with the study patient.
            nifti: the path to the NIFTI file with the vision data.
            report: the optional path to the radiology report data.
        """
        self._pmbb_id = pmbb_id
        self._nifti = nifti
        self._report = report

    @property
    def pmbb_id(self) -> str:
        """
        Returns the PMBB ID associated with the study patient.
        Input:
            None.
        Returns:
            The PMBB ID associated with the study patient.
        """
        return self._pmbb_id

    @property
    def nifti_dir(self) -> Union[Path, str]:
        """
        Returns the path to the NIFTI file with the vision data.
        Input:
            None.
        Returns:
            The path to the NIFTI file with the vision data.
        """
        return self._nifti

    @property
    def report_dir(self) -> Optional[Union[Path, str]]:
        """
        Returns the optional path to radiology report data.
        Input:
            None.
        Returns:
            The optional path to radiology report data.
        """
        return self._report

    @property
    def nifti(self) -> torch.Tensor:
        """
        Returns the vision data of the study.
        Input:
            None.
        Returns:
            The vision data of the study.
        """
        fn = next(
            filter(
                lambda fn: "".join(Path(fn.lower()).suffixes) in (
                    self.nifti_suffixes
                ),
                os.listdir(self.nifti_dir)
            )
        )
        return torch.from_numpy(
            nib.load(os.path.join(self.nifti_dir, fn)).get_fdata()
        )

    @property
    def nifti_metadata(self) -> Dict[str, Any]:
        """
        Returns the vision metadata of the study.
        Input:
            None.
        Returns:
            The vision metadata of the study.
        """
        fn = next(
            filter(
                lambda fn: Path(fn.lower()).suffix == ".json",
                os.listdir(self.nifti_dir)
            )
        )
        with open(os.path.join(self.nifti_dir, fn), "r") as f:
            return json.load(f)

    @property
    def report(self) -> Dict[str, Any]:
        """
        Returns the radiology report data of the study (if available).
        Input:
            None.
        Returns:
            The radiology report data of the study (if available).
        """
        if self.report_dir is None:
            return {}
        fn = next(
            filter(
                lambda fn: Path(fn.lower()).suffix == ".json",
                os.listdir(self.report_dir)
            )
        )
        with open(os.path.join(self.report_dir, fn), "r") as f:
            return json.load(f)


class BatchedStudy(NamedTuple):
    pmbb_id: Sequence[str]
    nifti: torch.Tensor
    nifti_metadata: Sequence[Dict[str, Any]]
    report: Sequence[Dict[str, Any]]


class Patient(Dataset):
    def __init__(self, pmbb_id: str, datadir: Union[Path, str]):
        """
        Args:
            pmbb_id: the PMBB ID associated with the patient.
            datadir: the data directory associated with the patient.
        """
        self._pmbb_id = pmbb_id
        self._datadir = datadir
        self._studies = []

    @property
    def pmbb_id(self) -> str:
        """
        Returns the PMBB ID associated with the patient.
        Input:
            None.
        Returns:
            The PMBB ID associated with the patient.
        """
        return self._pmbb_id

    @property
    def datadir(self) -> Union[Path, str]:
        """
        Returns the data directory associated with the patient.
        Input:
            None.
        Returns:
            The data directory associated with the patient.
        """
        return self._datadir

    def add_study(self, study: Union[Study, Sequence[Study]]) -> None:
        """
        Adds a study or list of studies to associate with the patient.
        Input:
            study: a study or list of studies to associated with the patient.
        Returns:
            None.
        """
        if isinstance(study, Study):
            study = [study]
        existing_nifti = [st.nifti_dir for st in self._studies]
        study = list(
            filter(lambda st: st.nifti_dir not in existing_nifti, study)
        )
        for st in study:
            self._studies.append(st)

    @property
    def studies(self) -> Sequence[Study]:
        """
        Returns a list of the diagnostic studies associated with the patient.
        Input:
            None.
        Returns:
            A list of the diagnostic studies associated with the patient.
        """
        return deepcopy(self._studies)

    def __len__(self) -> int:
        """
        Returns the number of diagnostic studies associated with the patient.
        Input:
            None.
        Returns:
            The number of diagnostic studies associated with the patient.
        """
        return len(self._studies)

    def __getitem__(self, idx: int) -> Study:
        """
        Returns a diagnostic study associated with the patient.
        Input:
            idx: the index of the study to retrieve.
        Returns:
            The specified diagnostic study associated with the patient.
        """
        return self._studies[idx]

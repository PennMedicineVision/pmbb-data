#!/usr/bin/env python3
"""
Defines the base object implementations in the PMBB.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import json
import nibabel as nib
import numpy as np
import os
import pickle
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
        if self.report_fn is None:
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
        existing_nifti = [st.nifti for st in self._studies]
        study = list(filter(lambda st: st.nifti not in existing_nifti, study))
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


class PMBB(Dataset):
    report_dir: str = "Diagnostic-Report"

    def __init__(
        self,
        pmbb_vision_dir: Union[Path, str],
        patients: Optional[Sequence[Patient]] = None
    ):
        """
        Args:
            pmbb_vision_dir: the parent director of the PMBB dataset.
            patients: an optional list of cached patients to exclusively load.
        """
        self.pmbb_vision_dir = pmbb_vision_dir
        if patients is not None:
            self.patients = patients
        else:
            self._load_patients()
            self._associate_studies_to_patients()
        self.cum_studies = np.cumsum([len(pt) for pt in self.patients])

    @classmethod
    def load_from_cache(cls, cache_fn: Union[Path, str]) -> PMBB:
        """
        Loads a PMBB dataset instance from a cache file.
        Input:
            cache_fn: the file path of the cache file.
        Returns:
            The PMBB dataset instance loaded from the input cache file.
        """
        with open(cache_fn, "rb") as f:
            patients, pmbb_vision_dir = pickle.load(f)
        return cls(pmbb_vision_dir, patients=patients)

    def write_to_cache(self, cache_fn: Union[Path, str]) -> None:
        """
        Writes the PMBB dataset to a cache file.
        Input:
            cache_fn: the cache file path to write to.
        Returns:
            None.
        """
        cache_dir = os.path.dirname(cache_fn)
        if len(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_fn, "wb") as f:
            pickle.dump((self.patients, self.pmbb_vision_dir), f)
        return

    def _load_patients(self) -> None:
        """
        Loads the patients from the parent PMBB data directory.
        Input:
            None.
        Returns:
            None.
        """
        subject_dirs = [
            os.path.join(self.pmbb_vision_dir, pref)
            for pref in os.listdir(self.pmbb_vision_dir)
        ]
        full_pmbbid_dirs = [
            [os.path.join(pref, suff) for suff in os.listdir(pref)]
            for pref in subject_dirs
        ]
        full_pmbbid_dirs = [
            [os.path.join(d, full) for full in os.listdir(d)]
            for d in sum(full_pmbbid_dirs, [])
        ]
        full_pmbbid_dirs = sum(full_pmbbid_dirs, [])

        # Create all of the patient objects.
        patients = map(
            lambda datadir: Patient(os.path.basename(datadir), datadir),
            full_pmbbid_dirs
        )
        self.patients = sorted(list(patients), key=lambda pt: pt.pmbb_id)
        return

    def __len__(self) -> int:
        """
        Returns the total number of unique studies in the dataset.
        Input:
            None.
        Returns:
            The total number of unique studies in the dataset.
        """
        return sum([len(pt) for pt in self.patients])

    def __getitem__(self, idx: int) -> Study:
        """
        Returns a diagnostic study from the dataset.
        Input:
            idx: the index of the study to retrieve.
        Returns:
            The specified diagnostic study from the dataset.
        """
        pt_idx = np.searchsorted(self.cum_studies, idx, side="right")
        st_idx = idx - self.cum_studies[pt_idx]
        return self.patients[pt_idx][st_idx]

    @property
    def num_patients(self) -> int:
        """
        Returns the total number of unique patients in the dataset.
        Input:
            None.
        Returns:
            The total number of unique patients in the dataset.
        """
        return len(self.patients)

    @property
    def num_studies(self) -> int:
        """
        Returns the total number of unique studies in the dataset.
        Input:
            None.
        Returns:
            The total number of unique studies in the dataset.
        """
        return len(self)

    def _associate_studies_to_patients(self) -> None:
        """
        Links the studies in the PMBB dataset to the parent patients.
        Input:
            None.
        Returns:
            None.
        """
        excluded_dirs = ["slurm"]
        for p in self.patients:
            studies = [
                [os.path.join(_id, study) for study in os.listdir(_id)]
                for _id in [
                    os.path.join(p.datadir, x) for x in os.listdir(p.datadir)
                ]
            ]
            studies = filter(
                lambda st: not any([
                    st.lower().endswith(ex) for ex in excluded_dirs
                ]),
                sum(studies, [])
            )
            studies = list(studies)

            reports = filter(
                lambda st: self.report_dir.lower() in st.lower(), studies
            )
            reports = list(reports)

            for st in studies:
                rep = filter(
                    lambda rep: os.path.dirname(st) == os.path.dirname(rep),
                    reports
                )
                rep = list(rep)
                p.add_study(Study(p.pmbb_id, st, rep[0] if len(rep) else None))

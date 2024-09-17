from __future__ import annotations
import numpy as np
import os
import pickle
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Sequence, Union

from .base import Patient, Study, BatchedStudy


class PMBB(Dataset):
    report_dir: str = "Diagnostic-Report"

    def __init__(
        self,
        pmbb_vision_dir: Union[Path, str],
        patients: Optional[Sequence[Patient]] = None
    ):
        """
        Args:
            pmbb_vision_dir: the parent directory of the PMBB dataset.
            patients: an optional list of cached patients to exclusively load.
        """
        self.pmbb_vision_dir = pmbb_vision_dir
        if patients is not None:
            self.patients = patients
        else:
            self._load_patients()
            self._associate_studies_to_patients()
        self.cum_studies = np.cumsum([len(pt) for pt in self.patients])

    def filter_by_modality(
        self,
        modality: Union[str, Sequence[str]],
        inplace: bool = False
    ) -> PMBB:
        """
        Filters the dataset to include only studies of a particular imaging
        modality.
        Input:
            modality: the imaging modality(s) to include in the dataset.
            inplace: whether to do the operation in place. Default False.
        Returns:
            The filtered dataset.
        """
        return self._filter_by_metadata_key(
            "Modality", modality, inplace=inplace
        )

    def filter_by_body_part_examined(
        self,
        body_part_examined: Union[str, Sequence[str]],
        inplace: bool = False
    ) -> PMBB:
        """
        Filters the dataset to include only studies of a particular imaging
        area.
        Input:
            body_part_examined: the imaging body part(s) examined to include
                in the dataset. By default, all body parts are included.
            inplace: whether to do the operation in place. Default False.
        Returns:
            The filtered dataset.
        """
        return self._filter_by_metadata_key(
            "BodyPartExamined", body_part_examined, inplace=inplace
        )

    def _filter_by_metadata_key(
        self, key: str, val: Union[str, Sequence[str]], inplace: bool = False
    ) -> PMBB:
        """
        Filters the dataset to include only studies of a particular value for
        a metadata key.
        Input:
            key: the vision metadata key to filter by.
            val: the allowed vision metadata value(s) to keep.
            inplace: whether to do the operation in place. Default False.
        Returns:
            The filtered dataset.
        """
        val = list(map(str.lower, [val] if isinstance(val, str) else val))

        if len(val) == 0:
            if inplace:
                return self
            return deepcopy(self)

        patients = [deepcopy(pt) for pt in self.patients]
        for pt in patients:
            pt._studies = list(
                filter(
                    lambda st: st.nifti_metadata.get(key, "").lower() in val,
                    pt._studies
                )
            )
        patients = list(filter(lambda pt: len(pt.studies) > 0, patients))

        if inplace:
            self.patients = patients
            return self
        return PMBB(pmbb_vision_dir=self.pmbb_vision_dir, patients=patients)

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
        full_pmbbid_dirs = list(
            filter(
                lambda _dir: any([
                    any([
                        f.lower().endswith(suffix.lower())
                        for suffix in Study.nifti_suffixes
                    ])
                    for _, _, fn in os.walk(_dir) for f in fn
                ]),
                full_pmbbid_dirs
            )
        )

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
        report_dir_key = "Diagnostic-Report"
        for p in self.patients:
            studies = [
                [os.path.join(_id, study) for study in os.listdir(_id)]
                for _id in [
                    os.path.join(p.datadir, x) for x in os.listdir(p.datadir)
                ]
            ]
            studies = filter(
                lambda st: not any([
                    st.lower().endswith(ex)
                    for ex in (excluded_dirs + [report_dir_key])
                ]),
                sum(studies, [])
            )
            studies = filter(
                lambda st: any([
                    any([
                        fn.lower().endswith(suffix.lower())
                        for suffix in Study.nifti_suffixes
                    ])
                    for fn in os.listdir(st)
                ]),
                list(studies)
            )
            studies = list(studies)

            report = None
            report_dir = os.path.join(
                os.path.dirname(studies[0]), report_dir_key
            )
            if os.path.isdir(report_dir):
                report = report_dir

            for st in studies:
                p.add_study(Study(p.pmbb_id, st, report))

    @staticmethod
    def collate_fn(studies: Sequence[Study]) -> BatchedStudy:
        """
        Collates a list of studies into a single BatchedStudy object.
        Input:
            studies: a list of studies to collate.
        Returns:
            A batched study containing the collated studies.
        """
        pmbb_id = [st.pmbb_id for st in studies]
        nifti = torch.vstack([st.nifti for st in studies])
        nifti_metadata = [st.nifti_metadata for st in studies]
        report = [st.report for st in studies]
        return BatchedStudy(
            pmbb_id=pmbb_id,
            nifti=nifti,
            nifti_metadata=nifti_metadata,
            report=report
        )

    @staticmethod
    def get_modality_options() -> Sequence[str]:
        """
        Returns the imaging modalities available in the dataset.
        Input:
            None.
        Returns:
            The imaging modalities present in the PMBB dataset.
        """
        return ["CR", "CT", "MR", "US"]

    @staticmethod
    def get_body_part_examined_options() -> Sequence[str]:
        """
        Returns the imaging modalities available in the dataset.
        Input:
            None.
        Returns:
            The imaging modalities present in the PMBB dataset.
        """
        return [
            "ABDOMEN",
            "ABDOMEN_CT_ABD_UNEN_PE",
            "ABDOMEN_PELVIS",
            "ABDPEL",
            "ABDPL",
            "ABD_PEL",
            "ABD_PELVIS",
            "AORTA",
            "CAP",
            "CAP_W",
            "CERVIX",
            "CHEAT",
            "CHEST",
            "CHEST_ABD",
            "CHEST_ABDOMEN",
            "CHEST_ABD_PELV",
            "CHEST_TO_PELVIS",
            "CHEST_WITHOUT",
            "CHST",
            "CH_AB_PEL",
            "COLON",
            "CSPINE",
            "CST_ABD_PEL",
            "CTA_ABD_AORTA_RU",
            "CTA_CHEST",
            "CTA_RUNOFF",
            "CT_ABDOMEN_PELVIS_W",
            "C_A_P",
            "EXTREMITY",
            "GASTRO",
            "GU",
            "HEAD",
            "HEAD_NECK",
            "HEART",
            "KIDNEY_URETER_BL",
            "LIVER_GALLBLADDE",
            "LSPINE",
            "LUMBAR_SPINE",
            "NECK",
            "NECK_CHEST_ABD",
            "OTHER",
            "PELVIS",
            "PE_CHEST",
            "RUNOFF",
            "SHOULDER",
            "SPINE",
            "THORACIC_AORTA",
            "TORSO",
            "TSPINE",
            "UNKNOWN",
            "VESSEL"
        ]

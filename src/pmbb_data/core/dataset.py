#!/usr/bin/python3
"""
PMBB dataset base class implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import os
import numpy as np
import pickle
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Type, Union

from .series import Series
from .report import Report
from .study import Study
from .patient import Patient
from ..config import PMBBConfig


class PMBBIndex(NamedTuple):
    patient_idx: int
    study_idx: int
    series_idx: int


class PMBBDataset(Dataset):
    cache_keys: Sequence[str] = [
        "patients", "studies", "series", "reports", "scans"
    ]

    def __init__(
        self,
        cache_dir: str = os.environ.get(
            "PMBB_CACHEDIR", os.path.join(os.environ["HOME"], ".cache/pmbb")
        ),
        patients: Optional[Sequence[Patient]] = None,
        strict: bool = True,
        seed: int = 0,
        **kwargs
    ):
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)
        self._vision_config = None
        self._language_config = None
        self._tokenizer = None
        for key, val in kwargs.items():
            setattr(self, key, val)
        if patients is not None:
            self._patients = patients
            return
        assert os.path.isdir(cache_dir) and all([
            os.path.isfile(os.path.join(cache_dir, fn + ".txt"))
            for fn in self.cache_keys
        ]), "PMBB cache not found, did you run `bash setup.sh`?"

        # Load the series and reports first.
        with open(os.path.join(cache_dir, "series.txt"), "r") as f:
            series = [Series(s, **kwargs) for s in list(f.readlines())]
        with open(os.path.join(cache_dir, "reports.txt"), "r") as f:
            reports = [Report(r, **kwargs) for r in list(f.readlines())]

        # Load the studies next.
        with open(os.path.join(cache_dir, "studies.txt"), "r") as f:
            studies = [
                Study(s, strict=strict, **kwargs) for s in list(f.readlines())
            ]

        # Associate series and reports with their parent studies.
        for s in tqdm(series, desc="Building Series"):
            for matched_study in filter(
                lambda x: s.study_id == x.study_id and s.pmbb_id == x.pmbb_id,
                studies
            ):
                matched_study.add_series(s)
        for r in tqdm(reports, desc="Building Reports"):
            for matched_study in filter(
                lambda x: r.study_id == x.study_id and r.pmbb_id == x.pmbb_id,
                studies
            ):
                matched_study.add_report(r)

        # Load the patients last.
        with open(os.path.join(cache_dir, "patients.txt"), "r") as f:
            patients = [
                Patient(p, strict=strict, **kwargs)
                for p in list(f.readlines())
            ]

        # Associate studies with their parent patient.
        for st in tqdm(studies, desc="Building Studies"):
            for matched_patient in filter(
               lambda x: st.pmbb_id == x.pmbb_id, patients
            ):
                matched_patient.add_study(st)

        self._patients = patients

        idxs = []
        for pt_idx in range(self.num_patients):
            for st_idx in range(self._patients[pt_idx].num_studies):
                for s_idx in range(
                    self._patients[pt_idx]._studies[st_idx].num_series
                ):
                    idxs.append(PMBBIndex(pt_idx, st_idx, s_idx))
        self._rng.shuffle(idxs)
        self._idxs = idxs

    def to_pickle(self, savepath: Union[Path, str]) -> None:
        with open(savepath, "wb") as f:
            pickle.dump(
                {
                    "patients": self._patients,
                    "indices": self._idxs,
                    "seed": self._seed,
                    "vision_config": self._vision_config,
                    "language_config": self._language_config,
                },
                f
            )

    @classmethod
    def from_pickle(
        cls: Type[PMBBDataset], savepath: Union[Path, str], **kwargs
    ) -> PMBBDataset:
        with open(savepath, "rb") as f:
            f.seek(0)
            data = pickle.load(f)
        ds = cls(patients=data["patients"], seed=data["seed"], **kwargs)
        ds._idxs = data["indices"]
        ds._vision_config = data["vision_config"]
        ds._language_config = data["language_config"]
        return ds

    def filter_by_modality(
        self,
        modality: Union[str, Sequence[str]],
        inplace: bool = True
    ) -> PMBBDataset:
        """
        Filters the dataset to include only studies of a particular imaging
        modality.
        Input:
            modality: the imaging modality(s) to include in the dataset.
            inplace: whether to do the operation in place. Default True.
        Returns:
            The filtered dataset.
        """
        return self._filter_by_metadata_key(
            "Modality", modality, inplace=inplace
        )

    def filter_by_body_part_examined(
        self,
        body_part_examined: Union[str, Sequence[str]],
        inplace: bool = True
    ) -> PMBBDataset:
        """
        Filters the dataset to include only studies of a particular imaging
        area.
        Input:
            body_part_examined: the imaging body part(s) examined to include
                in the dataset. By default, all body parts are included.
            inplace: whether to do the operation in place. Default True.
        Returns:
            The filtered dataset.
        """
        if isinstance(body_part_examined, str):
            body_part_examined = [body_part_examined]
        assert all([
            bpe in self.standardized_body_part_examined_values().keys()
            for bpe in body_part_examined
        ])
        body_part_examined = sum(
            [
                self.standardized_body_part_examined_values()[bpe]
                for bpe in body_part_examined
            ],
            []
        )
        return self._filter_by_metadata_key(
            "BodyPartExamined", body_part_examined, inplace=inplace
        )

    def _filter_by_metadata_key(
        self, key: str, val: Union[str, Sequence[str]], inplace: bool = True
    ) -> PMBBDataset:
        """
        Filters the dataset to include only studies of a particular value for
        a metadata key.
        Input:
            key: the vision metadata key to filter by.
            val: the allowed vision metadata value(s) to keep.
            inplace: whether to do the operation in place. Default True.
        Returns:
            The filtered dataset.
        """
        if not inplace:
            ds = deepcopy(self)
        else:
            ds = self
        val = list(map(str.lower, [val] if isinstance(val, str) else val))

        if len(val) == 0:
            return ds

        for pt in tqdm(ds.patients, desc=f"Filtering by {key} Metadata"):
            for st in pt.studies:
                st._series = list(
                    filter(
                        lambda s: s.nifti_metadata.get(key, "").lower() in val,
                        st._series
                    )
                )
            pt._studies = list(
                filter(lambda st: st.num_series > 0, pt._studies)
            )
        ds._patients = list(filter(lambda pt: pt.num_studies > 0, ds.patients))

        return self

    def __len__(self) -> int:
        """
        Returns the total number of unique series in the dataset.
        Input:
            None.
        Returns:
            The total number of unique series in the dataset.
        """
        return self.num_series

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Series, Report], Series, Report]:
        """
        Returns a diagnostic study from the dataset.
        Input:
            idx: the index of the study to retrieve.
        Returns:
            The specified diagnostic study from the dataset.
        """
        assert self._language_config is not None or (
            self._vision_config is not None
        )
        index = self._idxs[idx]
        study = self._patients[index.patient_idx].studies[index.study_idx]
        img = study.series[index.series_idx]
        if self._language_config is None:
            return img
        rep = study.report
        if self._vision_config is None:
            return rep
        return img, rep

    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        """
        Returns the language tokenizer if the dataset contains a
        language component.
        Input:
            None.
        Returns:
            The language tokenizer.
        """
        return self._tokenizer

    @property
    def patients(self) -> Sequence[Patient]:
        """
        Returns a list of the patients in the dataset.
        Input:
            None.
        Returns:
            A list of the patients in the dataset.
        """
        return self._patients

    @property
    def num_patients(self) -> int:
        """
        Returns the total number of unique patients in the dataset.
        Input:
            None.
        Returns:
            The total number of unique patients in the dataset.
        """
        return len(self._patients)

    @property
    def num_studies(self) -> int:
        """
        Returns the total number of unique studies in the dataset.
        Input:
            None.
        Returns:
            The total number of unique studies in the dataset.
        """
        return sum([pt.num_studies for pt in self._patients])

    @property
    def num_series(self) -> int:
        """
        Returns the total number of unique series in the dataset.
        Input:
            None.
        Returns:
            The total number of unique series in the dataset.
        """
        return sum([
            st.num_series for pt in self._patients for st in pt.studies
        ])

    def load_config(self, config: PMBBConfig) -> PMBBDataset:
        """
        Loads a configuration for the PMBB Dataset.
        Input:
            config: a PMBB configuration object.
        Returns:
            The original PMBB dataset with the loaded configuration object.
        """
        self._vision_config = config.vision
        self._language_config = config.language
        assert self._vision_config is not None or (
            self._language_config is not None
        )

        if config.body_parts_examined is not None:
            self.filter_by_body_part_examined(
                config.body_parts_examined, inplace=True
            )

        if config.modalities is not None:
            self.filter_by_modality(config.modalities, inplace=True)

        if self._language_config is not None:
            for pt in tqdm(self.patients, desc="Configuring Datatypes"):
                pt._studies = list(
                    filter(lambda st: st.report is not None, pt._studies)
                )
                if self._vision_config is not None:
                    pt._studies = list(
                        filter(lambda st: st.num_series > 0, pt._studies)
                    )
                    for st in pt._studies:
                        for s in st._series:
                            s.img_shape = self._vision_config.img_shape
            self._patients = list(
                filter(lambda pt: pt.num_studies > 0, self._patients)
            )
            self._tokenizer = self._language_config.tokenizer
            if isinstance(self._tokenizer, str):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer
                )

        self._seed = config.seed
        self._rng = np.random.default_rng(seed=self._seed)

        return self

    @staticmethod
    def standardized_body_part_examined_values() -> Dict[str, Sequence[str]]:
        return {
            "ABDOMEN": [
                "ABDOMEN",
                "ABDOMEN_CT_ABD_UNEN_PE",
                "ABDOMEN_PELVIS",
                "ABDPEL",
                "ABDPL",
                "ABD_PEL",
                "ABD_PELVIS",
                "CHEST_ABD",
                "CHEST_ABDOMEN",
                "CHEST_ABD_PELV",
                "CHEST_TO_PELVIS",
                "CH_AB_PEL",
                "CST_ABD_PEL",
                "CAP",
                "CAP_W",
                "C_A_P",
                "NECK_CHEST_ABD",
                "CT_ABDOMEN_PELVIS_W",
                "CTA_ABD_AORTA_RU",
                "LIVER_GALLBLADDE",
                "TORSO",
                "GASTRO",
                "COLON"
            ],
            "CHEST": [
                "CHEST_WITHOUT",
                "CHEAT",
                "CHST",
                "CHEST",
                "CHEST_ABD",
                "CHEST_ABDOMEN",
                "CHEST_ABD_PELV",
                "CHEST_TO_PELVIS",
                "CH_AB_PEL",
                "CST_ABD_PEL",
                "CAP",
                "CAP_W",
                "C_A_P",
                "CTA_CHEST",
                "NECK_CHEST_ABD",
                "HEART",
                "PE_CHEST",
                "TORSO"
            ],
            "AORTA": [
                "AORTA",
                "CTA_ABD_AORTA_RU",
                "THORACIC_AORTA"
            ],
            "PELVIS": [
                "CERVIX",
                "CHEST_ABD_PELV",
                "CHEST_TO_PELVIS",
                "CH_AB_PEL",
                "CST_ABD_PEL",
                "CAP",
                "CAP_W",
                "C_A_P",
                "GU",
                "CT_ABDOMEN_PELVIS_W",
                "PELVIS",
                "KIDNEY_URETER_BL"
            ],
            "EXTREMITY": [
                "EXTREMITY",
                "SHOULDER",
                "CTA_RUNOFF",
                "VESSEL"
            ],
            "HEADANDNECK": [
                "CSPINE",
                "NECK",
                "NECK_CHEST_ABD",
                "HEAD",
                "HEAD_NECK"
            ],
            "OTHER": [
                "OTHER",
                "UNKNOWN",
                "RUNOFF"
            ],
            "SPINE": [
                "SPINE",
                "CSPINE",
                "LSPINE",
                "LUMBAR_SPINE",
                "TSPINE"
            ]
        }

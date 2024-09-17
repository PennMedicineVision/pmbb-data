#!/usr/bin/python3
"""
PMBB patient base class implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import warnings
from pathlib import Path
from typing import Dict, Sequence, Union

from .base import PMBBObject
from .study import Study


class Patient(PMBBObject):
    def __init__(self, fn: Union[Path, str], strict: bool = True, **kwargs):
        """
        Args:
            fn: the filepath of the patient folder.
            strict: whether to enforce checks on the studies.
        """
        kwargs["strict"] = strict
        super().__init__(fn, **kwargs)
        self._studies = []

    def add_study(self, study: Union[Study, Sequence[Study]]) -> None:
        """
        Associates a study or multiple studies with the study.
        Input:
            study: the Study or list of Study objects to associate with the
                patient.
        Returns:
            None.
        """
        study = [study] if isinstance(study, Study) else study
        for s in study:
            if s.pmbb_id != self.pmbb_id:
                err_msg = "Patient and study metadata do not match!"
                if self.strict:
                    raise ValueError(err_msg)
                else:
                    warnings.warn(err_msg)
            self._studies.append(s)
        return

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
        *_, pmbb_id = filepath.strip().split("/")
        return {"pmbb_id": pmbb_id}

    @property
    def studies(self) -> Sequence[Study]:
        """
        Returns a list of the Study object(s) associated with the patient.
        Input:
            None.
        Returns:
            A list of the Study object(s) associated with the patient.
        """
        return self._studies

    @property
    def num_studies(self) -> int:
        """
        Returns the number of Study object(s) associated with the patient.
        Input:
            None.
        Returns:
            The number of Study object(s) associated with the patient.
        """
        return len(self._studies)

    @property
    def num_series(self) -> int:
        """
        Returns the number of Series object(s) associated with the patient.
        Input:
            None.
        Returns:
            The number of Series object(s) associated with the patient.
        """
        return sum([st.num_series for st in self._studies])

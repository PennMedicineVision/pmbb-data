#!/usr/bin/python3
"""
Diagnostic text report base class implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .base import PMBBObject


class Report(PMBBObject):
    report_ext: str = ".txt"

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
        *_, pmbb_id, study_id, _ = filepath.strip().split("/")
        return {"pmbb_id": pmbb_id, "study_id": study_id}

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
    def text(self) -> Optional[str]:
        """
        Returns the imaging study metadata associated with the series.
        Input:
            None.
        Returns:
            The imaging study metadata associated with the series.
        """
        with open(self._fn, "r") as f:
            return "".join(f.readlines())

#!/usr/bin/python3
"""
DICOM Studies base class implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from .base import PMBBObject
from .series import Series
from .report import Report


class Study(PMBBObject):
    def __init__(self, fn: Union[Path, str], strict: bool = True, **kwargs):
        """
        Args:
            fn: the filepath of the study folder.
            strict: whether to enforce checks on the series and reports.
        """
        kwargs["strict"] = strict
        super().__init__(fn, **kwargs)
        self._series = []
        self._report = None

    def add_series(self, series: Union[Series, Sequence[Series]]) -> None:
        """
        Associates a series or multiple series objects with the study.
        Input:
            series: the Series or list of Series to associate with the study.
        Returns:
            None.
        """
        series = [series] if isinstance(series, Series) else series
        for s in series:
            if s.pmbb_id != self.pmbb_id or s.study_id != self.study_id:
                err_msg = "Study and series metadata do not match!"
                if self.strict:
                    raise ValueError(err_msg)
                else:
                    warnings.warn(err_msg)
            self._series.append(s)
        return

    def add_report(self, report: Report) -> None:
        """
        Associates a report object with the study.
        Input:
            report: the Report to associate with the study.
        Returns:
            None.
        """
        if report.pmbb_id != self.pmbb_id or report.study_id != self.study_id:
            err_msg = "Study and report metadata do not match!"
            if self.strict:
                raise ValueError(err_msg)
            else:
                warnings.warn(err_msg)
        if self._report is not None:
            err_msg = "A report is already associated with this study!"
            if self.strict:
                raise ValueError(err_msg)
            else:
                warnings.warn(err_msg)
        self._report = report

    @property
    def report(self) -> Optional[Report]:
        """
        Returns the report associated with the study.
        Input:
            None.
        Returns:
            The report (if available) associated with the study.
        """
        return self._report

    @property
    def series(self) -> Sequence[Series]:
        """
        Returns a list of the series object(s) associated with the study.
        Input:
            None.
        Returns:
            A list of the series object(s) associated with the study.
        """
        return self._series

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
        *_, pmbb_id, study_id = filepath.strip().split("/")
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
    def num_series(self) -> int:
        """
        Returns the number of Series object(s) associated with the patient.
        Input:
            None.
        Returns:
            The number of Series object(s) associated with the patient.
        """
        return len(self._series)

#!/usr/bin/python3
"""
PMBB object base class implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import abc
import os
from pathlib import Path
from typing import Dict, Final, Union


class PMBBObject(abc.ABC):
    def __init__(self, fn: Union[Path, str], **kwargs):
        """
        Args:
            fn: the filepath of the series folder.
        """
        self._fn: Final = fn.strip()
        self._metadata: Final = self.get_metadata_from_filepath(self._fn)
        for key, val in kwargs.items():
            setattr(self, key, val)

    @staticmethod
    @abc.abstractmethod
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
        raise NotImplementedError

    @property
    def pmbb_id(self) -> str:
        """
        Returns the PMBB ID of the patient who received the study.
        Input:
            None.
        Returns:
            The PMBB ID of the patient who received the study.
        """
        return self._metadata["pmbb_id"]

    @property
    def abspath(self) -> Union[Path, str]:
        """
        Returns the absolute path of the series.
        Input:
            None.
        Returns:
            The absolute path of the series.
        """
        return os.path.abspath(self._fn)

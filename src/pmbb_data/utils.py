#!/usr/bin/env python3
"""
Utility functions for manipulating PMBB data.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
from typing import Generator, Optional, Tuple, Union

from .pmbb import PMBB


def train_test_split(
    dataset: PMBB,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[Union[int, Generator]] = None,
    shuffle: bool = True
) -> Tuple[PMBB]:
    """
    Splits a PMBB dataset into random train and test subsets.
    Input:
        dataset: the dataset to split.
        test_size: the proportion (if float) or absolute number (if int) of
            samples to serve as the test set. Default 0.25 if both `test_size`
            and `train_size` are not specified.
        train_size: the proportion (if float) or absolute number (if int) of
            samples to serve as the training set.
        random_state: random state or seed.
        shuffle: whether to shuffle the dataset before splitting.
    Returns:
        train: the training dataset.
        test: the test dataset.
    """
    if not isinstance(random_state, Generator):
        random_state = np.random.default_rng(seed=random_state)

    if test_size is None and train_size is None:
        test_size = round(0.25 * len(dataset))
        train_size = len(dataset) - test_size
    elif test_size is None:
        assert isinstance(train_size, float) and 0.0 < train_size < 1.0 or (
            isinstance(train_size, int) and 0 < train_size < len(dataset)
        )
        if isinstance(train_size, float):
            train_size = round(train_size * len(dataset))
        test_size = len(dataset) - train_size
    elif train_size is None:
        assert isinstance(test_size, float) and 0.0 < test_size < 1.0 or (
            isinstance(test_size, int) and 0 < test_size < len(dataset)
        )
        if isinstance(test_size, float):
            test_size = round(test_size * len(dataset))
        train_size = len(dataset) - test_size
    else:
        assert any([
            all([isinstance(n, t) for n in [train_size, test_size]])
            for t in [float, int]
        ])
        assert train_size > 0 and test_size > 0
        if isinstance(train_size, float):
            train_size = round(train_size * len(dataset))
            test_size = round(test_size * len(dataset))
        assert train_size + test_size == len(dataset)

    pt_idxs = np.arange(dataset.num_patients)
    num_studies_per_patient = np.array([len(pt) for pt in dataset.patients])
    if shuffle:
        random_state.shuffle(pt_idxs)
        num_studies_per_patient = num_studies_per_patient[pt_idxs]
        for pt in dataset.patients:
            random_state.shuffle(pt._studies)
    train_eidx = np.searchsorted(
        np.cumsum(num_studies_per_patient), train_size, side="right"
    )

    train = PMBB(
        pmbb_vision_dir=dataset.pmbb_vision_dir,
        patients=[dataset.patients[idx] for idx in pt_idxs[:train_eidx]]
    )
    test = PMBB(
        pmbb_vision_dir=dataset.pmbb_vision_dir,
        patients=[dataset.patients[idx] for idx in pt_idxs[train_eidx:]]
    )
    return train, test

#!/usr/bin/env python3
"""
Utility functions for manipulating PMBB data.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
from typing import Generator, Optional, Tuple, Union

from .core import PMBBDataset, PMBBIndex


def train_test_split(
    dataset: PMBBDataset,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[Union[int, Generator]] = None,
    shuffle: bool = True
) -> Tuple[PMBBDataset]:
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
    if shuffle:
        random_state.shuffle(pt_idxs)
        for pt in dataset.patients:
            random_state.shuffle(pt._studies)

    train = PMBBDataset(
        patients=[dataset.patients[idx] for idx in pt_idxs[:train_size]]
    )
    test = PMBBDataset(
        patients=[dataset.patients[idx] for idx in pt_idxs[train_size:]]
    )

    # Reconstruct the trainin indices.
    train_idxs = []
    for pt_idx in range(train.num_patients):
        for st_idx in range(train._patients[pt_idx].num_studies):
            for s_idx in range(
                train._patients[pt_idx]._studies[st_idx].num_series
            ):
                train_idxs.append(PMBBIndex(pt_idx, st_idx, s_idx))
    random_state.shuffle(train_idxs)
    train._idxs = train_idxs
    train_seed = dataset._seed
    if dataset._seed is not None:
        train_seed += sum([ord(c) for c in "train"])
    train._rng = np.random.default_rng(seed=train_seed)
    train._vision_config = dataset._vision_config
    train._language_config = dataset._language_config
    train._tokenizer = dataset._tokenizer

    # Reconstruct the trainin indices.
    test_idxs = []
    for pt_idx in range(test.num_patients):
        for st_idx in range(test._patients[pt_idx].num_studies):
            for s_idx in range(
                test._patients[pt_idx]._studies[st_idx].num_series
            ):
                test_idxs.append(PMBBIndex(pt_idx, st_idx, s_idx))
    random_state.shuffle(test_idxs)
    test._idxs = test_idxs
    test_seed = dataset._seed
    if dataset._seed is not None:
        test_seed += sum([ord(c) for c in "test"])
    test._rng = np.random.default_rng(seed=test_seed)
    test._vision_config = dataset._vision_config
    test._language_config = dataset._language_config
    test._tokenizer = dataset._tokenizer

    return train, test

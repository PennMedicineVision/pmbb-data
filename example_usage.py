#!/usr/bin/python3
"""
Example PMBB dataset usage.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania.
"""
import click
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

from pmbb_data.core import PMBBDataset, Series
from pmbb_data.config import PMBBConfig
from pmbb_data.utils import train_test_split


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="The dataset configuration file."
)
@click.option(
    "--cache",
    type=str,
    default="pmbb_dataset.pkl",
    show_default=True,
    help="The path to cache the dataset to and from."
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    show_default=True,
    help="Batch size."
)
def main(
    config: Union[Path, str],
    cache: Optional[Union[Path, str]] = None,
    batch_size: int = 4,
):
    """Example usage of the PMBB Data repository."""
    config = PMBBConfig.from_yaml(config)

    if os.path.isfile(cache):
        ds = PMBBDataset.from_pickle(cache)
    else:
        ds = PMBBDataset()
        ds.load_config(config)
        ds.to_pickle(cache)

    train, test = train_test_split(ds, test_size=0.2)

    def sample_collate_fn(data):
        series, reports = zip(*data)
        return Series.collate_fn(series), [r.text for r in reports]

    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=sample_collate_fn
    )

    model = nn.Identity()  # Dummy model.

    # Dummy usage.
    for img, txt in tqdm(train_dataloader):
        _ = model(img)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze the PMBB dataset.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import os
from pmbb_data import PMBB
from pmbb_data.utils import train_test_split
from pathlib import Path
from typing import Optional, Union


@click.command()
@click.option(
    "--from",
    "cache_fn",
    type=str,
    default=".pmbb_cache.pkl",
    show_default=True,
    help="Optional cache file for the PMBB dataset."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed."
)
def main(
    cache_fn: Optional[Union[Path, str]] = ".pmbb_cache.pkl",
    seed: int = 42
):
    """Sample PMBB dataset script."""
    if cache_fn is not None and os.path.isfile(cache_fn):
        ds = PMBB.load_from_cache(cache_fn)
    else:
        ds = PMBB(os.environ["PMBB_DATADIR"])
        if cache_fn is not None:
            ds.write_to_cache(cache_fn)

    train, test = train_test_split(ds, random_state=seed)

    click.secho("Number of Patients:", bold=True)
    click.echo(f"  Train: {train.num_patients}")
    click.echo(f"  Test: {test.num_patients}")

    click.secho("Number of Studies:", bold=True)
    click.echo(f"  Train: {len(train)}")
    click.echo(f"  Test: {len(test)}")


if __name__ == "__main__":
    main()
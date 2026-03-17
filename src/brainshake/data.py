"""
Data module
---
-> Used to create a pytorch dataset
-> For each individual EGG signal we're computing features ( delegate to feature module )
-> This module handles data loading
"""

import numpy as np
import argparse
import logging
import pandas as pd
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

log = logging.getLogger(__name__)


class EGGDataset(Dataset):
    def __init__(self, data_dir="data/Epilepsy"):

        self.samples: list[Tuple[Path, int]] = []
        self.patient_ids: list[str] = []

        for i in range(1, 25):
            data_filename = f"chb{str(i).zfill(2)}_seizure_EEGwindow_1.npz"
            try:
                data = np.load(f"{data_dir}/{data_filename}", allow_pickle=True)
            except Exception as e:
                log.error(f"Could not load {data_filename}: {e}")

            metadata_filename = f"chb{str(i).zfill(2)}_seizure_metadata_1.parquet"
            try:
                metadata = pd.read_parquet(f"data/Epilepsy/{metadata_filename}")
            except Exception as e:
                log.error(f"Could not load {metadata_filename}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Number of v's increases verbosity",
    )

    args = parser.parse_args()
    verbosity = args.verbose

    loglevel = logging.WARNING

    if verbosity >= 2:
        loglevel = logging.DEBUG
    elif verbosity == 1:
        loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format="%(levelname)s: %(message)s")

    load_data()


def load_data():
    log.debug("Starting data load...")

    log.debug("Starting EEG data load...")
    data = []
    for i in range(1, 25):
        filename = f"chb{str(i).zfill(2)}_seizure_EEGwindow_1.npz"
        try:
            data.append(np.load(f"data/Epilepsy/{filename}", allow_pickle=True))
        except Exception as e:
            log.error(f"Could not load {filename}: {e}")
    log.debug("...Loaded all 25 NPZ EEG files")

    log.debug("Starting EEG metadata load...")
    metadata = []
    for i in range(1, 25):
        filename = f"chb{str(i).zfill(2)}_seizure_metadata_1.parquet"
        try:
            metadata.append(pd.read_parquet(f"data/Epilepsy/{filename}"))
        except Exception as e:
            log.error(f"Could not load {filename}: {e}")
    log.debug("...Loaded all 25 parquet metadata files")

    ### EGG DATA
    egg_win_0 = data[0]["EEG_win"]
    # print(len(egg_win_0))  # 45701 --> Number of windows for patient 0
    # print(len(egg_win_0[0]))  # 21 Electrodes ?
    # print(len(egg_win_0[0][0]))  # 128 --> type : Floats : samples
    # print(len(egg_win_0[0][0][0])) # error
    ### First data block has shape (45701,21,128)

    ###
    #           class  filename_interval  global_interval      filename
    # 0          0                  1                1         chb24_01.edf
    # 1          0                  1                1         chb24_01.edf
    # ...
    # 45699      1                  1               16         chb24_21.edf
    # 45700      1                  1               16         chb24_21.edf
    #

    plt.plot(egg_win_0[0][0])
    plt.savefig("graphs/first_data_visualization.png")


if __name__ == "__main__":
    main()

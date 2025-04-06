"""extract_iram_data.py
Extract data from IRAM cubes
Trey V. Wenger - December 2024
"""

import os
import sys
import glob

import pickle
import numpy as np


def main(datadir, source):
    transitions = ["CN-12", "CN-32", "13CN-12", "13CN-32"]
    labels = ["12CN-1/2", "12CN-3/2", "13CN-1/2", "13CN-3/2"]

    data = {}
    for transition, label in zip(transitions, labels):
        with open(
            os.path.join(datadir, source, f"{source}-{transition}_data.pkl"), "rb"
        ) as f:
            data[label] = pickle.load(f)

    # split up spectra and save
    outdir = os.path.join(datadir, source, "spectra")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for file in glob.glob(os.path.join(outdir, "*.pkl")):
        os.remove(file)

    # skip spectra with nan
    mask = np.any(
        [np.any(np.isnan(data[label]["cube"]), axis=0) for label in labels], axis=0
    )
    print(f"{source} pixels masked: {mask.sum()}")

    for idx, coord in enumerate(zip(*np.where(~mask))):
        datum = {"coord": coord}
        for label in labels:
            datum[f"frequency_{label}"] = data[label]["frequency"]
            datum[f"spectrum_{label}"] = data[label]["cube"][:, *coord]
            datum[f"rms_{label}"] = data[label]["rms"][*coord]
        with open(os.path.join(outdir, f"{idx}.pkl"), "wb") as f:
            pickle.dump(datum, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python split_iram_data.py <datadir> <source>")
    else:
        main(sys.argv[1], sys.argv[2])

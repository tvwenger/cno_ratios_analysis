"""fit_iram.py
Fit bayes_hfs models to IRAM CN data.
Trey V. Wenger - May 2026
"""

import sys
import pickle
import glob

import numpy as np

import pymc as pm
import bayes_spec
import bayes_hfs

import cloudpickle as cpickle

from astropy.io import fits
import astropy.constants as c

from bayes_spec import SpecData
from bayes_hfs import supplement_molecule_data, HFSRatioModel


def main(source, n_clouds, project, prior_velocity, data_ranges):
    print(f"Starting job on {source} with n_clouds={n_clouds}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"bayes_hfs version: {bayes_hfs.__version__}")
    print(f"project: {project}")
    print(f"prior_velocity: {prior_velocity}")
    print(f"data_ranges: {data_ranges}")

    result = {
        "source": source,
        "project": project,
        "n_clouds": n_clouds,
        "prior_velocity": prior_velocity,
        "data_ranges": data_ranges,
    }

    # load CN mol_data
    with open("mol_data_12CN.pkl", "rb") as f:
        all_mol_data_12CN = pickle.load(f)
    with open("mol_metadata_12CN.pkl", "rb") as f:
        all_mol_metadata_12CN = pickle.load(f)

    # Keep only Kl = 0 transitions
    all_mol_data_12CN = all_mol_data_12CN[all_mol_data_12CN["Kl"] == 0]

    # Add GLO
    all_mol_data_12CN["GLO"] = 2 * all_mol_data_12CN["F1l"]

    # load 13CN mol_data
    with open("mol_data_13CN.pkl", "rb") as f:
        all_mol_data_13CN = pickle.load(f)
    with open("mol_metadata_13CN.pkl", "rb") as f:
        all_mol_metadata_13CN = pickle.load(f)

    # Add GLO
    all_mol_data_13CN["GLO"] = 2 * all_mol_data_13CN["F1l"] + 1

    mol_data_12CN = supplement_molecule_data(all_mol_data_12CN, all_mol_metadata_12CN)
    mol_data_13CN = supplement_molecule_data(all_mol_data_13CN, all_mol_metadata_13CN)

    # load data
    data_keys = ["CN12", "CN32", "13CN12", "13CN32"]
    labels = [
        r"CN $T_B$ (K)",
        r"CN $T_B$ (K)",
        r"$^{13}$CN $T_B$ (K)",
        r"$^{13}$CN $T_B$ (K)",
    ]
    data = {}
    for data_key, label, data_range in zip(data_keys, labels, data_ranges):
        with fits.open(f"{project}/{source}-{data_key}.fits") as hdulist:
            hdu = hdulist[0]
            freq_axis = (
                (np.arange(hdu.header["NAXIS1"]) + 1 - hdu.header["CRPIX1"])
                * hdu.header["CDELT1"]
                + hdu.header["CRVAL1"]
                + hdu.header["RESTFREQ"]
                * (1.0 - hdu.header["VELO-LSR"] / c.c.to("m/s").value)
            ) / 1.0e6  # MHz
            data[data_key] = SpecData(
                freq_axis[data_range[0] : data_range[1]],
                hdu.data[0, 0, 0, data_range[0] : data_range[1]],
                1.4826 * np.median(np.abs(hdu.data - np.median(hdu.data))),
                xlabel=r"LSRK Frequency (MHz)",
                ylabel=label,
            )

    # association each dataset with the related species
    mol_keys = {
        "12CN": ["CN12", "CN32"],
        "13CN": ["13CN12", "13CN32"],
    }

    try:
        # Initialize optimizer
        model = HFSRatioModel(
            mol_data_12CN,
            mol_data_13CN,
            mol_keys,
            data,
            bg_temp=2.7,
            Beff=0.782,
            Feff=1.0,
            n_clouds=n_clouds,
            baseline_degree=0,
            seed=1234,
            verbose=True,
        )

        # Define each model
        model.add_priors(
            prior_log10_Ntot1=[13.5, 0.5],
            prior_ratio=0.1,
            prior_fwhm2=1.0,
            prior_velocity=prior_velocity,
            prior_log10_Tex_CTEX=[0.75, 0.25],
            assume_CTEX1=False,
            assume_CTEX2=True,
            prior_log10_CTEX_variance=[-4.0, 1.0],
            clip_weights=1.0e-9,
            clip_tau=-10.0,
            prior_fwhm_L=None,
            prior_baseline_coeffs=None,
        )
        model.add_likelihood()

        # sample
        solve_kwargs = {
            "init_params": "random_from_data",
            "n_init": 10,
            "max_iter": 1_000,
            "kl_div_threshold": 0.1,
        }
        model.sample(
            init="advi+adapt_diag",
            tune=1000,
            draws=1000,
            chains=8,
            cores=8,
            n_init=200_000,
            init_kwargs={
                "rel_tolerance": 0.01,
                "abs_tolerance": 0.01,
                "learning_rate": 0.001,
                "start": {"velocity_norm": np.linspace(0.1, 0.9, n_clouds)},
            },
            nuts_kwargs={"target_accept": 0.9},
        )
        model.solve(**solve_kwargs)
        result["model"] = model

    except Exception as ex:
        result["exception"] = ex

    return result


if __name__ == "__main__":
    source = sys.argv[1]
    n_clouds = int(sys.argv[2])

    prior_velocity = None
    data_ranges = None
    with open("bayes_hfs_priors.txt", "r") as f:
        for line in f:
            parts = line.split()
            if parts[0] == source:
                prior_velocity = np.array(parts[1:3], dtype=float)
                if prior_velocity[1] < prior_velocity[0]:
                    prior_velocity = prior_velocity[::-1]
                data_ranges = np.array(parts[3:], dtype=int).reshape((4, 2))
    if prior_velocity is None:
        raise ValueError(f"{source} not found in bayes_hfs_priors.txt")

    datafiles = glob.glob(f"*/{source}*.csv")
    project = "107-23"
    for file in datafiles:
        if "042-25" in file:
            project = "042-25"

    output = main(source, n_clouds, project, prior_velocity, data_ranges)
    if "exception" in output:
        print("EXCEPTION:", output["exception"])

    # save results
    outfile = f"{source}_n{n_clouds}_results.pkl"
    with open(outfile, "wb") as f:
        cpickle.dump(output, f)

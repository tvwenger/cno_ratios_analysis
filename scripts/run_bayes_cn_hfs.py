"""run_bayes_cn_hfs.py
Optimize bayes_cn_hfs model on some data.
Trey V. Wenger - April 2025
"""

import sys
import pickle

import numpy as np
import pandas as pd

import pymc as pm
import bayes_spec
import bayes_cn_hfs

from bayes_spec import SpecData, Optimize
from bayes_cn_hfs import CNModel, CNRatioModel

flux_density_dict = {  # in Jy
    "G005.885": 7.44e0,
    "G010.623": 4.74e0,
    "G301.138": 3.16e0,
    "G305.384": 9.31e-3,
    "G309.176": 4.80e-2,
    "G310.901": 2.17e-1,
    "G311.563": 1.79e-2,
    "G312.598": 1.88e-1,
    "G317.891": 3.94e-2,
    "G318.774": 8.59e-2,
    "G320.232": 3.66e-1,
    "G320.331": 4.70e-2,
    "G320.778": 3.88e-2,
    "G323.449": 8.08e-2,
    "G326.473": 3.55e-1,
    "G328.825": 2.12e-2,
    "G329.398": 1.91e-1,
    "G329.460": 2.33e-2,
    "G333.052": 3.31e-2,
    "G333.164": 1.20e-1,
    "G334.341": 8.43e-2,
    "G334.659": 1.97e-2,
    "G334.976": 1.71e-2,
    "G336.358": 2.29e-1,
    "G336.766": 3.14e-2,
    "G337.404": 4.00e-1,
    "G337.705": 4.19e-2,
    "G337.922": -1.98e-2,
    "G338.851": 8.43e-2,
    "G339.106": 1.23e-1,
    "G339.486": 1.02e-1,
    "G339.845": 1.95e-2,
    "G340.247": 1.87e-2,
    "G342.360": 3.50e-2,
    "G344.424": 2.08e-3,
}


def main(source):
    print(f"Starting job on {source}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"bayes_cn_hfs version: {bayes_cn_hfs.__version__}")
    result = {
        "source": source,
        "exception": "",
        "opt_results": {},
        "results": {},
    }

    # load mol_data
    with open("mol_data_12CN.pkl", "rb") as f:
        mol_data_12CN = pickle.load(f)
    with open("mol_data_13CN.pkl", "rb") as f:
        mol_data_13CN = pickle.load(f)

    # load data
    data_12CN_1 = np.genfromtxt(f"{source}_feather_CN.tsv")
    data_12CN_2 = np.genfromtxt(f"{source}_feather_cont1.tsv")
    data_13CN = np.genfromtxt(f"{source}_feather_13CN.tsv")
    data_12CN_1 = data_12CN_1[~np.isnan(data_12CN_1).any(axis=1)]
    data_12CN_2 = data_12CN_2[~np.isnan(data_12CN_2).any(axis=1)]
    data_13CN = data_13CN[~np.isnan(data_13CN).any(axis=1)]

    # estimate noise
    noise_12CN_1 = 1.4826 * np.median(
        np.abs(data_12CN_1[:, 1] - np.median(data_12CN_1[:, 1]))
    )
    noise_12CN_2 = 1.4826 * np.median(
        np.abs(data_12CN_2[:, 1] - np.median(data_12CN_2[:, 1]))
    )
    noise_13CN = 1.4826 * np.median(
        np.abs(data_13CN[:, 1] - np.median(data_13CN[:, 1]))
    )

    # save data
    obs_12CN_1 = SpecData(
        data_12CN_1[:, 0],
        data_12CN_1[:, 1],
        noise_12CN_1,
        xlabel=r"LSRK Frequency (MHz)",
        ylabel=r"$T_{B,\,\rm CN}$ (K)",
    )
    obs_12CN_2 = SpecData(
        data_12CN_2[:, 0],
        data_12CN_2[:, 1],
        noise_12CN_2,
        xlabel=r"LSRK Frequency (MHz)",
        ylabel=r"$T_{B,\,\rm CN}$ (K)",
    )
    obs_13CN = SpecData(
        data_13CN[:, 0],
        data_13CN[:, 1],
        noise_13CN,
        xlabel=r"LSRK Frequency (MHz)",
        ylabel=r"$T_{B,\,^{13}\rm CN}$ (K)",
    )
    data_12CN = {"12CN_1": obs_12CN_1, "12CN_2": obs_12CN_2}
    data = {"12CN_1": obs_12CN_1, "12CN_2": obs_12CN_2, "13CN": obs_13CN}

    # Estimate background temperature
    with open(f"{source}_feather_CN.tsv", "r") as f:
        lines = [line.rstrip() for line in f]
    strings = lines[4].split("[")[3].split(",")[:2]
    theta_1 = float(strings[0].strip("'").strip('"'))
    theta_2 = float(strings[1].strip("'").strip("]").strip(" ")[:-1])
    if theta_1 > theta_2:
        theta_major = theta_1
        theta_minor = theta_2
    else:
        theta_major = theta_2
        theta_minor = theta_1

    # Convert from arcsec to radians
    factor = 4.84814e-6
    theta_major = theta_major * factor
    theta_minor = theta_minor * factor

    v = 113510e6  # Hz (precise number probably doesn't matter, only order of magnitude matters)
    k = 1.380649e-23  # J/K

    c = 3e8  # m/s
    wavelength = c / v
    Omega = np.pi * theta_major * theta_minor / (4 * np.log(2))
    hii_temp = (
        wavelength**2 * flux_density_dict[source] * 1e-26 / (2 * k * Omega)
    )  # factor of 1e-26 to convert from Jy to W*s*m^-2
    print(f"HII region BG temperature: {hii_temp:.1f}")

    # Get LSR velocity
    df = pd.read_csv("targets.csv")
    df["Source"] = df["Name"].str[:8]
    vlsr = df.loc[df["Source"] == source, " vel(km/s)"].values[0]

    try:
        # Initialize optimizer
        opt = Optimize(
            CNModel,  # model definition
            data_12CN,  # data dictionary
            max_n_clouds=8,  # maximum number of clouds
            baseline_degree=0,  # polynomial baseline degree
            bg_temp=hii_temp + 2.7,  # CMB + HII region
            Beff=1.0,  # beam efficiency
            Feff=1.0,  # forward efficiency
            mol_data=mol_data_12CN,  # molecular data
            seed=1234,  # random seed
            verbose=True,  # verbosity
        )

        # Define each model
        opt.add_priors(
            prior_log10_N=[13.5, 1.0],  # cm-2
            prior_log10_Tkin=None,  # ignored
            prior_velocity=[vlsr, 5.0],  # km s-1
            prior_fwhm_nonthermal=1.0,  # km s-1
            prior_fwhm_L=None,  # assume Gaussian line profile
            prior_rms=None,  # do not infer spectral rms
            prior_baseline_coeffs=None,  # use default baseline priors
            assume_LTE=False,  # do not assume LTE
            prior_log10_Tex=[0.5, 0.1],  # K
            assume_CTEX=False,  # do not assume CTEX
            prior_LTE_precision=100.0,  # width of LTE precision prior
            fix_log10_Tkin=1.5,  # kinetic temperature is fixed (K)
            ordered=False,  # do not assume optically-thin
        )
        opt.add_likelihood()

        # optimize
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 0.02,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 1000,
            "draws": 1000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(
            bic_threshold=10.0,
            sample_kwargs=sample_kwargs,
            fit_kwargs=fit_kwargs,
            approx=False,
        )

        # save BICs and results for each model
        opt_results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            opt_results[n_gauss] = {"bic": np.inf, "solutions": {}}
            for solution in model.solutions:
                # get BIC
                bic = model.bic(solution=solution)

                # get summary
                summary = pm.summary(model.trace[f"solution_{solution}"])

                # check convergence
                converged = summary["r_hat"].max() < 1.05

                if converged and bic < opt_results[n_gauss]["bic"]:
                    opt_results[n_gauss]["bic"] = bic

                # save posterior samples for un-normalized params (except baseline)
                data_vars = list(model.trace[f"solution_{solution}"].data_vars)
                data_vars = [
                    data_var
                    for data_var in data_vars
                    if ("baseline" in data_var) or not ("norm" in data_var)
                ]

                # only save posterior samples if converged
                opt_results[n_gauss]["solutions"][solution] = {
                    "bic": bic,
                    "summary": summary,
                    "converged": converged,
                    "trace": (
                        model.trace[f"solution_{solution}"][data_vars].sel(
                            draw=slice(None, None, 10)
                        )
                        if converged
                        else None
                    ),
                }
        result["opt_results"] = opt_results

        # Initialize model
        model = CNRatioModel(
            data,  # data dictionary
            n_clouds=opt.best_model.n_clouds,
            baseline_degree=0,  # polynomial baseline degree
            bg_temp=hii_temp + 2.7,  # CMB + HII region
            Beff=1.0,  # beam efficiency
            Feff=1.0,  # forward efficiency
            mol_data_12CN=mol_data_12CN,  # molecular data
            mol_data_13CN=mol_data_13CN,  # molecular data
            seed=1234,  # random seed
            verbose=True,  # verbosity
        )

        # Add priors
        model.add_priors(
            prior_log10_N_12CN=[13.5, 1.0],  # cm-2
            prior_ratio_12C_13C=[75.0, 25.0],
            prior_log10_Tkin=None,  # ignored
            prior_velocity=[vlsr, 5.0],  # km s-1
            prior_fwhm_nonthermal=1.0,  # km s-1
            prior_fwhm_L=None,  # assume Gaussian line profile
            prior_rms=None,  # do not infer spectral rms
            prior_baseline_coeffs=None,  # use default baseline priors
            assume_LTE=False,  # do not assume LTE
            prior_log10_Tex=[0.5, 0.1],  # K
            assume_CTEX_12CN=False,  # do not assume CTEX
            prior_LTE_precision=100.0,  # width of LTE precision prior
            assume_CTEX_13CN=True,  # assume CTEX for 13CN
            fix_log10_Tkin=1.5,  # kinetic temperature is fixed (K)
            ordered=False,  # do not assume optically-thin
        )

        # Add likelihood
        model.add_likelihood()

        # fit
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.12,
            "learning_rate": 0.02,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 1000,
            "draws": 1000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        model.sample(init="advi+adapt_diag", **sample_kwargs)
        model.solve(kl_div_threshold=0.1)

        results = {"bic": np.inf, "solutions": {}}
        for solution in model.solutions:
            # get BIC
            bic = model.bic(solution=solution)

            # get summary
            summary = pm.summary(model.trace[f"solution_{solution}"])

            # check convergence
            converged = summary["r_hat"].max() < 1.05

            if converged and bic < results["bic"]:
                results["bic"] = bic

            # save posterior samples for un-normalized params (except baseline)
            data_vars = list(model.trace[f"solution_{solution}"].data_vars)
            data_vars = [
                data_var
                for data_var in data_vars
                if ("baseline" in data_var) or not ("norm" in data_var)
            ]

            # only save posterior samples if converged
            results["solutions"][solution] = {
                "bic": bic,
                "summary": summary,
                "converged": converged,
                "trace": (
                    model.trace[f"solution_{solution}"][data_vars].sel(
                        draw=slice(None, None, 10)
                    )
                    if converged
                    else None
                ),
            }
        result["results"] = results

        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    source = sys.argv[1]
    outfile = f"{source}_results.pkl"

    output = main(source)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    with open(outfile, "wb") as f:
        pickle.dump(output, f)

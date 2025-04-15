"""analyze_bayes_cn_hfs.py
Check bayes_cn_hfs model results.
Trey V. Wenger - April 2025
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az
import pymc as pm
import bayes_spec
import bayes_cn_hfs

from bayes_spec import SpecData
from bayes_cn_hfs import CNModel, CNRatioModel

from bayes_spec import plots

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


def main(source, datadir="alma_spectra/", resultsdir="results/"):
    print(f"Starting job on {source}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"bayes_cn_hfs version: {bayes_cn_hfs.__version__}")

    # load results
    resultfname = os.path.join(resultsdir, f"{source}_results.pkl")
    if not os.path.exists(resultfname):
        raise FileNotFoundError(f"{resultfname} not found")
    with open(resultfname, "rb") as f:
        result = pickle.load(f)

    # create output directory
    outdir = os.path.join(resultsdir, source)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # load mol_data
    with open(os.path.join(datadir, "mol_data_12CN.pkl"), "rb") as f:
        mol_data_12CN = pickle.load(f)
    with open(os.path.join(datadir, "mol_data_13CN.pkl"), "rb") as f:
        mol_data_13CN = pickle.load(f)

    # load data
    data_12CN_1 = np.genfromtxt(os.path.join(datadir, f"{source}_feather_CN.tsv"))
    data_12CN_2 = np.genfromtxt(os.path.join(datadir, f"{source}_feather_cont1.tsv"))
    data_13CN = np.genfromtxt(os.path.join(datadir, f"{source}_feather_13CN.tsv"))
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
    noise_13CN = 1.4826 * np.median(np.abs(data_13CN[:, 1] - np.median(data_13CN[:, 1])))

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
    data_12CN = {"12CN": obs_12CN_1}
    data = {"12CN_1": obs_12CN_1, "12CN_2": obs_12CN_2, "13CN": obs_13CN}

    # Estimate background temperature
    with open(os.path.join(datadir, f"{source}_feather_CN.tsv"), "r") as f:
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
    df = pd.read_csv(os.path.join(datadir, "targets.csv"))
    df["Source"] = df["Name"].str[:8]
    vlsr = df.loc[df["Source"] == source, " vel(km/s)"].values[0]
    print(f"HII region VLSR: {vlsr:.1f}")

    # Plot CNModel BICs
    n_clouds = list(result["opt_results"].keys())[1:] # Skips n=0 clouds
    bics = [result["opt_results"][n_cloud]["bic"] for n_cloud in n_clouds]
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(n_clouds, bics, "ko")
    ax.set_xlabel("Number of Clouds")
    ax.set_ylabel("BIC")
    fig.savefig(os.path.join(outdir, f"{source}_bics.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Plot the 12CN and 13CN data
    fig, axes = plt.subplots(3, layout="constrained")
    axes[0].plot(obs_12CN_1.spectral, obs_12CN_1.brightness, "k-")
    axes[0].axhline(noise_12CN_1, color="b", label="rms")
    axes[0].set_xlabel("LSRK Frequency (MHz)")
    axes[0].set_ylabel(r"$T_B$ (K)")
    axes[1].plot(obs_12CN_2.spectral, obs_12CN_2.brightness, "k-")
    axes[1].axhline(noise_12CN_2, color="b", label="rms")
    axes[1].set_xlabel("LSRK Frequency (MHz)")
    axes[1].set_ylabel(r"$T_B$ (K)")
    axes[2].plot(obs_13CN.spectral, obs_13CN.brightness, "k-")
    axes[2].axhline(noise_13CN, color="b", label="rms")
    axes[2].set_xlabel("LSRK Frequency (MHz)")
    axes[2].set_ylabel(r"$T_B$ (K)")
    axes[2].legend(loc="upper right")
    fig.savefig(os.path.join(outdir, f"{source}_12_13CN_data.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Plot CNModel predictive samples
    for n_cloud in n_clouds:
        if not np.isfinite(bics[n_cloud-1]):
            continue

        model = CNModel(
            data_12CN,  # data dictionary
            n_clouds=n_cloud,  # number of clouds
            baseline_degree=0,  # polynomial baseline degree
            bg_temp=hii_temp + 2.7,  # CMB + HII region
            Beff=1.0,  # beam efficiency
            Feff=1.0,  # forward efficiency
            mol_data=mol_data_12CN,  # molecular data
            seed=1234,  # random seed
            verbose=True,  # verbosity
        )
        model.add_priors(
            prior_log10_N=[13.5, 1.0],  # cm-2
            prior_log10_Tkin=None,  # ignored
            prior_velocity=[vlsr, 10.0],  # km s-1
            prior_fwhm_nonthermal=1.0,  # km s-1
            prior_fwhm_L=None,  # assume Gaussian line profile
            prior_rms=None,  # do not infer spectral rms
            prior_baseline_coeffs=None,  # use default baseline priors
            assume_LTE=False,  # do not assume LTE
            prior_log10_Tex=[0.5, 0.25],  # K
            assume_CTEX=False,  # do not assume CTEX
            prior_LTE_precision=100.0,  # width of LTE precision prior
            fix_log10_Tkin=1.5,  # kinetic temperature is fixed (K)
            ordered=False,  # do not assume optically-thin
        )
        model.add_likelihood()

        # plot the prior predictive
        prior = model.sample_prior_predictive(
            samples=100,  # prior predictive samples
        )
        axes = plots.plot_predictive(model.data, prior.prior_predictive)
        fig = axes.ravel()[0].figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"{source}_CN_{n_cloud}_prior_predictive.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

        # loop over solutions
        for solution in result["opt_results"][n_cloud]["solutions"].keys():
            if not result["opt_results"][n_cloud]["solutions"][solution]["converged"]:
                continue

            # pack posterior samples
            model.trace = az.convert_to_inference_data(
                result["opt_results"][n_cloud]["solutions"][solution]["trace"]
            )

            # plot traces
            axes = plots.plot_traces(
                model.trace,
                ["velocity", "fwhm", "log10_N", "tau", "weights"],
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"{source}_CN_{n_cloud}_trace.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # plot posterior pairs
            var_names = [
                param
                for param in model.cloud_deterministics
                if not set(model.model.named_vars_to_dims[param]).intersection(
                    set(["transition", "state"])
                )
            ]
            axes = plots.plot_pair(
                model.trace,  # samples
                var_names,  # var_names to plot
                labeller=model.labeller,  # label manager
                kind="scatter",  # plot type
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"{source}_CN_{n_cloud}_posterior_pair.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # plot posterior predictive
            # N.B. this does not work because we don't save the normalized posterior samples
            """
            posterior = model.sample_posterior_predictive(
                thin=1,  # keep one in {thin} posterior samples
            )
            axes = plots.plot_predictive(model.data, posterior.posterior_predictive)
            """
            thin = 10
            posterior_predictive = {"12CN": []}
            trace = az.extract(model.trace.posterior)
            for sample in trace.sample[::thin]:
                trace_sample = trace.sel(sample=sample)
                sim_params_12CN = {
                    key: trace_sample[key].data
                    for key in [
                        "log10_N",
                        "fwhm_nonthermal",
                        "velocity",
                        "log10_Tex_ul",
                        "weights",
                        "baseline_12CN_norm",
                    ]
                }
                for key in posterior_predictive.keys():
                    posterior_predictive[key].append(
                        model.model[key].eval(sim_params_12CN, on_unused_input="ignore")
                        + np.random.normal(loc=0.0, scale=model.data[key].noise)
                    )
            fig, ax = plt.subplots(1, layout="constrained")
            ax.plot(obs_12CN_1.spectral, obs_12CN_1.brightness, "k-")
            for predictive in posterior_predictive["12CN"]:
                ax.plot(obs_12CN_1.spectral, predictive, "r-", alpha=0.1)
            ax.set_xlabel("LSRK Frequency (MHz)")
            ax.set_ylabel(r"$T_B$ (K)")
            fig.savefig(
                os.path.join(outdir, f"{source}_CN_{n_cloud}_posterior_predictive.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # plot posterior predictive residuals
            fig, ax = plt.subplots(1, layout="constrained")
            ax.plot(obs_12CN_1.spectral, np.zeros_like(obs_12CN_1.brightness), "k-")
            for predictive in posterior_predictive["12CN"]:
                ax.plot(
                    obs_12CN_1.spectral,
                    obs_12CN_1.brightness - predictive,
                    "r-",
                    alpha=0.1,
                )
            ax.set_xlabel("LSRK Frequency (MHz)")
            ax.set_ylabel(r"$T_B$ (K)")
            fig.savefig(
                os.path.join(
                    outdir, f"{source}_CN_{n_cloud}_posterior_predictive_residuals.pdf"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
    
    # Plot 'best' n_clouds - 2, 'best' n_clouds - 1, 'best' n_clouds
    for n_clouds_offset in [2,1,0]:
        n_clouds_input = n_clouds[np.argmin(bics)] - n_clouds_offset
        if n_clouds_input < 1:
            continue

        model = CNRatioModel(
            data,  # data dictionary
            n_clouds=n_clouds_input,
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
            prior_ratio_12C_13C=[50.0, 50.0],
            prior_log10_Tkin=None,  # ignored
            prior_velocity=[vlsr, 10.0],  # km s-1
            prior_fwhm_nonthermal=1.0,  # km s-1
            prior_fwhm_L=None,  # assume Gaussian line profile
            prior_rms=None,  # do not infer spectral rms
            prior_baseline_coeffs=None,  # use default baseline priors
            assume_LTE=False,  # do not assume LTE
            prior_log10_Tex=[0.5, 0.25],  # K
            assume_CTEX_12CN=False,  # do not assume CTEX
            prior_LTE_precision=100.0,  # width of LTE precision prior
            assume_CTEX_13CN=True,  # assume CTEX for 13CN
            fix_log10_Tkin=1.5,  # kinetic temperature is fixed (K)
            ordered=False,  # do not assume optically-thin
        )

        # Add likelihood
        model.add_likelihood()

        # plot the prior predictive
        prior = model.sample_prior_predictive(
            samples=100,  # prior predictive samples
        )
        axes = plots.plot_predictive(model.data, prior.prior_predictive)
        fig = axes.ravel()[0].figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"{source}_12_13CN_{n_clouds_input}_prior_predictive.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

        for solution in result[f"results_{n_clouds_input}"]["solutions"].keys():
            if not result[f"results_{n_clouds_input}"]["solutions"][solution]["converged"]:
                continue

            # pack posterior samples
            model.trace = az.convert_to_inference_data(
                result[f"results_{n_clouds_input}"]["solutions"][solution]["trace"]
            )

            # plot traces
            axes = plots.plot_traces(
                model.trace,
                ["ratio_12C_13C", "velocity", "fwhm_12CN", "fwhm_13CN", "log10_N_12CN", "N_13CN", "tau_total_12CN", "tau_total_13CN"],
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"{source}_12_13CN_{n_clouds_input}_trace.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # plot posterior pairs
            axes = plots.plot_pair(
                model.trace,  # samples
                ["ratio_12C_13C", "velocity", "fwhm_12CN", "fwhm_13CN", "log10_N_12CN", "N_13CN",], # var_names  # var_names to plot
                labeller=model.labeller,  # label manager
                kind="scatter",  # plot type
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"{source}_12_13CN_{n_clouds_input}_posterior_pair.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # plot posterior predictive
            # N.B. this does not work because we don't save the normalized posterior samples
            '''
            posterior = model.sample_posterior_predictive(
                thin=1,  # keep one in {thin} posterior samples
            )
            '''
            thin = 10
            posterior_predictive = {"12CN_1": [], "12CN_2": [], "13CN": []}
            trace = az.extract(model.trace.posterior)
            for sample in trace.sample[::thin]:
                trace_sample = trace.sel(sample=sample)
                sim_params_12CN_1 = {
                    key: trace_sample[key].data
                    for key in [
                        "log10_N_12CN",
                        "fwhm_nonthermal",
                        "velocity",
                        "log10_Tex_ul",
                        "weights_12CN",
                        "baseline_12CN_1_norm",
                    ]
                }
                sim_params_12CN_2 = {
                    key: trace_sample[key].data
                    for key in [
                        "log10_N_12CN",
                        "fwhm_nonthermal",
                        "velocity",
                        "log10_Tex_ul",
                        "weights_12CN",
                        "baseline_12CN_2_norm", 
                    ]
                }
                sim_params_13CN = {
                    key: trace_sample[key].data
                    for key in [
                        "N_13CN",
                        "fwhm_nonthermal",
                        "velocity",
                        "log10_Tex_ul",
                        "baseline_13CN_norm",
                    ]
                }
                for key in posterior_predictive.keys():
                    if key == "12CN_1":
                        posterior_predictive[key].append(
                            model.model[key].eval(sim_params_12CN_1, on_unused_input="ignore")
                            + np.random.normal(loc=0.0, scale=model.data[key].noise)
                        )
                    if key == "12CN_2":
                        posterior_predictive[key].append(
                            model.model[key].eval(sim_params_12CN_2, on_unused_input="ignore")
                            + np.random.normal(loc=0.0, scale=model.data[key].noise)
                        )
                    if key == "13CN":
                        posterior_predictive[key].append(
                            model.model[key].eval(sim_params_13CN, on_unused_input="ignore")
                            + np.random.normal(loc=0.0, scale=model.data[key].noise)
                        )
            fig, axes = plt.subplots(3, layout="constrained")
            axes[0].plot(obs_12CN_1.spectral, obs_12CN_1.brightness, "k-")
            for predictive in posterior_predictive["12CN_1"]:
                axes[0].plot(obs_12CN_1.spectral, predictive, "r-", alpha=0.1)
            axes[0].set_xlabel("LSRK Frequency (MHz)")
            axes[0].set_ylabel(r"$T_B$ (K)")
            axes[1].plot(obs_12CN_2.spectral, obs_12CN_2.brightness, "k-")
            for predictive in posterior_predictive["12CN_2"]:
                axes[1].plot(obs_12CN_2.spectral, predictive, "r-", alpha=0.1)
            axes[1].set_xlabel("LSRK Frequency (MHz)")
            axes[1].set_ylabel(r"$T_B$ (K)")
            axes[2].plot(obs_13CN.spectral, obs_13CN.brightness, "k-")
            for predictive in posterior_predictive["13CN"]:
                axes[2].plot(obs_13CN.spectral, predictive, "r-", alpha=0.1)
            axes[2].set_xlabel("LSRK Frequency (MHz)")
            axes[2].set_ylabel(r"$T_B$ (K)")
            fig.savefig(
                os.path.join(outdir, f"{source}_12_13CN_{n_clouds_input}_posterior_predictive.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # plot posterior predictive residuals
            fig, axes = plt.subplots(3, layout="constrained")
            axes[0].plot(obs_12CN_1.spectral, np.zeros_like(obs_12CN_1.brightness), "k-")
            for predictive in posterior_predictive["12CN_1"]:
                axes[0].plot(
                    obs_12CN_1.spectral,
                    obs_12CN_1.brightness - predictive,
                    "r-",
                    alpha=0.1,
                )
            axes[0].set_xlabel("LSRK Frequency (MHz)")
            axes[0].set_ylabel(r"$T_B$ (K)")
            axes[1].plot(obs_12CN_2.spectral, np.zeros_like(obs_12CN_2.brightness), "k-")
            for predictive in posterior_predictive["12CN_2"]:
                axes[1].plot(
                    obs_12CN_2.spectral,
                    obs_12CN_2.brightness - predictive,
                    "r-",
                    alpha=0.1,
                )
            axes[1].set_xlabel("LSRK Frequency (MHz)")
            axes[1].set_ylabel(r"$T_B$ (K)")
            axes[2].plot(obs_13CN.spectral, np.zeros_like(obs_13CN.brightness), "k-")
            for predictive in posterior_predictive["13CN"]:
                axes[2].plot(
                    obs_13CN.spectral,
                    obs_13CN.brightness - predictive,
                    "r-",
                    alpha=0.1,
                )
            axes[2].set_xlabel("LSRK Frequency (MHz)")
            axes[2].set_ylabel(r"$T_B$ (K)")
            fig.savefig(
                os.path.join(
                    outdir, f"{source}_12_13CN_{n_clouds_input}_posterior_predictive_residuals.pdf"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            summary_12CN_12CN_df = result[f"results_{n_clouds_input}"]["solutions"][0]["summary"]
            summary_12CN_12CN_df.to_csv(os.path.join(outdir, f"{source}_12_13CN_{n_clouds_input}_summary_stats.csv"))
    
if __name__ == "__main__":
    source = sys.argv[1]
    datadir = sys.argv[2]
    resultsdir = sys.argv[3]
    main(source, datadir=datadir, resultsdir=resultsdir)

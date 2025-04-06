"""extract_iram_data.py
Extract data from IRAM cubes
Trey V. Wenger - December 2024
"""

import os
import sys

import pickle
import numpy as np

from spectral_cube import SpectralCube
import astropy.units as u


def main(datadir, source):
    transitions = ["CN-12", "CN-32", "13CN-12", "13CN-32"]
    keep_chans = [
        slice(200, -200),
        slice(250, -250),
        slice(250, -250),
        slice(250, -250),
    ]

    # storage for WCS
    wcs = None

    common_beam = None
    for transition, keep_chan in zip(transitions, keep_chans):
        print(f"Processing: {source} {transition} data")
        fname = os.path.join(datadir, source, f"{source}-{transition}.fits")
        cube = SpectralCube.read(fname)
        cube = cube.with_spectral_unit(
            u.MHz,
            velocity_convention="radio",
            rest_value=cube.header["RESTFREQ"] * u.Hz,
        )
        cube.allow_huge_operations = True

        # IRAM cubes are 683 channels covering +/- 200 MHz = +/- 550 km/s
        # we can trim them a bit
        cube = cube[keep_chan]

        # smoothing to common beam
        if common_beam is None:
            # worst beam will be at 1612. The channels are basically the same
            common_beam = cube.beam
        else:
            cube = cube.convolve_to(common_beam)

        # continuum estimate
        cont = cube.median(axis=0)

        # save WCS
        if wcs is None:
            wcs = cont.wcs

        # estimate rms
        med = np.nanmedian(cube, axis=0)
        rms = 1.4826 * np.nanmedian(np.abs(cube._data - med), axis=0)

        # save
        data = {
            "cube": cube._data,
            "rms": rms,
            "frequency": cube.spectral_axis.to("MHz").value,
        }

        with open(
            os.path.join(datadir, source, f"{source}-{transition}_data.pkl"), "wb"
        ) as f:
            pickle.dump(data, f)

        # save rms
        cont.data = rms
        cont.write(
            os.path.join(datadir, source, f"{source}-{transition}_rms.fits"),
            format="fits",
            overwrite=True,
        )

    # save WCS
    with open(os.path.join(datadir, source, f"{source}_wcs.pkl"), "wb") as f:
        pickle.dump(wcs, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python extract_iram_data.py <datadir> <source>")
    else:
        main(sys.argv[1], sys.argv[2])

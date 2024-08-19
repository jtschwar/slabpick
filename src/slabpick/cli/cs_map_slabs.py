import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import slabpick.coordinates as coordinates
import slabpick.csedit as csedit
import slabpick.dataio as dataio
import slabpick.minislab as minislab


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Map particles from slab-picking back to their 3d coordinates.",
    )
    parser.add_argument(
        "--cs_file",
        type=str,
        required=True,
        help="Cryosparc extraction job, e.g. topaz_picked_particles.cs",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        required=True,
        help="Bookkeeping file mapping particles to gallery tiles",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing tomograms used for slab generation",
    )
    parser.add_argument(
        "--extract_shape",
        type=int,
        nargs=3,
        required=True,
        help="Subvolume extraction shape in (x,y,z) in Angstrom",
    )
    parser.add_argument(
        "--voxel_spacing",
        type=float,
        required=True,
        help="Tomogram voxel spacing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Distance threshold in Angstrom for removing duplicates",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output starfile",
    )
    parser.add_argument(
        "--apix",
        type=float,
        required=False,
        default=1,
        help="Tilt-series pixel size, inverse applied if writing starfile",
    )

    return parser.parse_args()


def main():
    config = parse_args()

    extract_shape = (np.array(config.extract_shape) / config.voxel_spacing).astype(int)
    extract_shape = minislab.render_even(extract_shape)

    d_coords = csedit.curate_slab_map(
        np.load(config.cs_file), pd.read_csv(config.map_file), config.voxel_spacing,
    )

    assert extract_shape[0] == extract_shape[1]
    mask = coordinates.generate_window(extract_shape[0])

    for run_name in d_coords:
        volume = dataio.load_mrc(os.path.join(config.in_dir, f"{run_name}.mrc"))
        coords = np.round(d_coords[run_name] / config.voxel_spacing).astype(int)
        rcoords = coordinates.refine_z(coords, volume, extract_shape, mask)
        rcoords, nr = coordinates.merge_replicates(
            rcoords.astype(np.float32) * config.voxel_spacing,
            config.threshold,
        )
        d_coords[run_name] = rcoords

    dataio.make_starfile(
        d_coords,
        config.out_file,
        coords_scale=config.apix,
    )


if __name__ == "__main__":
    main()

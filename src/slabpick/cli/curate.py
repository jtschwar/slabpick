from argparse import ArgumentParser

import numpy as np
import pandas as pd

from slabpick.csedit import curate_particles_map, curate_particles_map_iterative
from slabpick.dataio import (
    CoPickWrangler,
    combine_star_files,
    make_starfile,
    read_starfile,
)


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Generate starfile based on cryosparc-curated picks.",
    )
    parser.add_argument(
        "--copick_json", type=str, required=False, help="Copick json file",
    )
    parser.add_argument(
        "--in_star",
        type=str,
        required=False,
        help="Starfile containing coordinates associated with map_file",
    )
    parser.add_argument(
        "--in_star_multiple",
        type=str,
        required=False,
        nargs="+",
        help="List of starfiles containing coordinates associated with map_file",
    )
    parser.add_argument(
        "--col_name",
        type=str,
        required=False,
        default="rlnTomoName",
        help="Tomogram column name in starfile(s)",
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
        "--particle_name",
        type=str,
        required=False,
        help="Copick particle name, required if using copick for coordinates",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        required=False,
        help="Copick session ID, required if using copick for coordinates",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=False,
        help="Copick user ID, required if using copick for coordinates",
    )
    parser.add_argument("--out_file", type=str, required=True, help="Output starfile")
    parser.add_argument(
        "--apix",
        type=float,
        required=True,
        help="Tilt-series pixel size (usually unbinned)",
    )
    parser.add_argument(
        "--rejected_set",
        action="store_true",
        help="Extract coordinates of the rejected particles in the star file",
    )

    return parser.parse_args()


def main(config):
    # extract all particle coordinates
    if config.in_star:
        d_coords = read_starfile(
            config.in_star, coords_scale=config.apix, col_name=config.col_name,
        )
    elif config.in_star_multiple:
        d_coords = combine_star_files(
            config.in_star_multiple, coords_scale=config.apix, col_name=config.col_name,
        )
    elif config.copick_json:
        cp_interface = CoPickWrangler(config.copick_json)
        d_coords = cp_interface.get_all_coords(
            config.particle_name, config.session_id, config.user_id,
        )
    else:
        raise ValueError("Either a copick config or a starfile must be provided.")
    ini_particle_count = np.sum(
        np.array([d_coords[tomo].shape[0] for tomo in d_coords]),
    )

    # map retained particles in cryosparc to gallery tiles
    cs_extract = np.load(config.cs_file)
    particles_map = pd.read_csv(config.map_file)
    if len(particles_map) > 1e6:
        curated_map = curate_particles_map_iterative(
            cs_extract, particles_map, rejected_set=config.rejected_set,
        )
    else:
        curated_map = curate_particles_map(
            cs_extract, particles_map, rejected_set=config.rejected_set,
        )

    # curate particles
    d_coords_sel = {}
    tomo_list = np.unique(curated_map.tomogram.values)
    for _i, tomo in enumerate(tomo_list):
        tomo_indices = np.where(curated_map.tomogram.values == tomo)[0]
        particle_indices = curated_map.iloc[tomo_indices].particle.values
        d_coords_sel[tomo] = d_coords[tomo][particle_indices]
        final_particle_count = np.sum(
            np.array([d_coords_sel[tomo].shape[0] for tomo in d_coords_sel]),
        )
    print(
        f"Cryosparc reduced particle set size from {ini_particle_count} to {final_particle_count}",
    )

    # generate Relion 4-compatible starfile
    make_starfile(d_coords_sel, config.out_file, coords_scale=1.0 / config.apix)


if __name__ == "__main__":
    config = parse_args()
    main(config)

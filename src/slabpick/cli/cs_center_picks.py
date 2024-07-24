import os
import shutil
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from slabpick.settings import ProcessingConfigCsCenterPicks


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Modify a picked_particles.cs file to center picks on tile centers.",
    )
    parser.add_argument(
        "--cs_file",
        type=str,
        required=True,
        help="Cryosparc picked_particles.cs file",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        required=True,
        help="Bookkeeping file mapping particles to gallery tiles",
    )
    parser.add_argument(
        "--gallery_shape",
        type=int,
        nargs=2,
        required=False,
        default=[16, 15],
        help="Number of gallery particles in (row,col) format",
    )

    return parser.parse_args()


def generate_config(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)

    reconfig = {}
    reconfig["software"] = {"name": "slabpick", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in ["cs_file", "map_file"]}
    reconfig["output"] = {k: d_config[k] for k in ["cs_file"]}
    reconfig["parameters"] = {k: d_config[k] for k in ["gallery_shape"]}

    reconfig = ProcessingConfigCsCenterPicks(**reconfig)

    out_dir = os.path.dirname(os.path.abspath(os.path.join(config.map_file, "../")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cs_center_picks.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))


def main():
    config = parse_args()
    generate_config(config)

    particles_map = pd.read_csv(config.map_file)
    cs_picks = np.load(config.cs_file)
    gallery_shape = tuple(config.gallery_shape)[
        ::-1
    ]  # order is reversed relative to gallery.py

    # find indices to retain based on correct number of particles per micrograph
    mgraph_id = cs_picks["location/micrograph_path"]
    retain_idx = []
    gnums = np.unique(particles_map.gallery.values)
    for i, gn in enumerate(gnums):
        idx = np.where(
            [f"particles_{gn:03}.mrc" in fn.decode("utf-8") for fn in mgraph_id],
        )[0]
        assert len(idx) > np.prod(gallery_shape)
        if i < gnums[-1]:
            retain_idx.extend(list(idx[: np.prod(gallery_shape)]))
        else:
            num = len(particles_map) % np.prod(gallery_shape)
            if num == 0:  # fix for case of exact multiple of gallery shape
                num = np.prod(gallery_shape)
            retain_idx.extend(list(idx[:num]))
    retain_idx = np.array(retain_idx)
    cs_picks = cs_picks[retain_idx]
    assert len(cs_picks) == len(particles_map)

    # now replace the blob picker locations with centered positions, discarding extras
    fxpos = np.arange(gallery_shape[0]) / gallery_shape[0] + 0.5 / gallery_shape[0]
    fypos = np.arange(gallery_shape[1]) / gallery_shape[1] + 0.5 / gallery_shape[1]
    xpos, ypos = np.meshgrid(fxpos, fypos)
    xpos, ypos = xpos.flatten(), ypos.flatten()
    new_xpos = np.tile(xpos, len(gnums) - 1)
    new_ypos = np.tile(ypos, len(gnums) - 1)
    num = len(particles_map) % np.prod(gallery_shape)
    if num == 0:
        num = np.prod(gallery_shape)
    new_xpos = np.concatenate((new_xpos, xpos[:num]))
    new_ypos = np.concatenate((new_ypos, ypos[:num]))

    cs_picks["location/center_x_frac"] = new_xpos
    cs_picks["location/center_y_frac"] = new_ypos

    # overwrite the original picks file but save a copy as a backup
    shutil.copy2(config.cs_file, config.cs_file.split(".cs")[0] + "_original.cs")
    with open(config.cs_file, "wb") as f:
        np.save(f, cs_picks)


if __name__ == "__main__":
    main()

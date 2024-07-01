from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from dataio import read_starfile
from dataio import make_starfile
from coordinates import consolidate_coordinates_sets
from visualize import visualize_merge_pairs

def parse_args():
    """ Parser for command line arguments.
    """
    parser = ArgumentParser(description="Generate starfile based on cryosparc-curated picks.")
    parser.add_argument("--in_star1", type=str, required=True,
                        help="Starfile containing first set of coordinates")
    parser.add_argument("--in_star2", type=str, required=True,
                        help="Starfile containing second set of coordinates")
    parser.add_argument("--out_star", type=str, required=True,
                        help="Output starfile for merged coordinates")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Distance threshold for considering coordinates replicates")
    parser.add_argument("--coords_scale", type=float, required=False, default=1.0,
                        help="Factor to scale coordinates in starfiles to Angstrom")
    parser.add_argument("--weights", type=float, required=False, nargs=2, default=[0.5,0.5],
                        help="Relative weights to apply to in_star1 and in_star2 when merging")
    parser.add_argument("--copick_json", type=str, required=False,
                        help="Copick configuration file in copick format if visualizing replicates")
    parser.add_argument("--extract_shape", type=float, required=False, nargs=3,
                        help="Subvolume extraction shape for generating minislabs")
    parser.add_argument("--out_visuals", type=str,  required=False,
                        help="Output directory for visualizing pairs of coordinates")
    
    return parser.parse_args()

def main(config):

    coords1 = read_starfile(config.in_star1, coords_scale=config.coords_scale)
    coords2 = read_starfile(config.in_star2, coords_scale=config.coords_scale)

    d_coords_merge, d_clusters = consolidate_coordinates_sets(coords1,
                                                              coords2,
                                                              config.threshold,
                                                              config.weights)
    
    make_starfile(d_coords_merge,
                  config.out_star,
                  coords_scale=1.0/config.coords_scale)

    if config.copick_json is not None and config.out_visuals is not None:
        os.makedirs(config.out_visuals, exist_ok=True)
        for run_name in d_clusters.keys():
            if len(d_clusters[run_name])!=0:
                visualize_merge_pairs(run_name, 
                                      coords1, 
                                      coords2, 
                                      d_clusters, 
                                      config.copick_json, 
                                      config.extract_shape,
                                      output=os.path.join(config.out_visuals, f"{run_name}.png"))
    
if __name__ == "__main__":
    config = parse_args()
    main(config)


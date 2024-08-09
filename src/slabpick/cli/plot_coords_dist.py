import os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from slabpick.dataio import combine_star_files


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Analyze distribution of PyTom picks.",
    )
    parser.add_argument(
        "--in_star",
        type=str,
        nargs="+",
        required=True,
        help="Star files to process",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        required=True,
        help="Output path for plot file",
    )
    parser.add_argument(
        "--coords_scale",
        required=False,
        type=float,
        default=10,
        help="Multiplicative scale factor to convert coordinates to Angstrom",
    )
    parser.add_argument(
        "--col_name",
        required=False,
        type=str,
        default="rlnMicrographName",
        help="Column name for tomogram name",
    )

    return parser.parse_args()


def main():

    config = parse_args()

    d_coords = combine_star_files(
        config.in_star, 
        col_name = config.col_name, 
        coords_scale = config.coords_scale,
    )
    
    counts = np.array([d_coords[run_name].shape[0] for run_name in d_coords.keys()])
    xpos = np.concatenate([d_coords[run_name][:,0] for run_name in d_coords.keys()])
    ypos = np.concatenate([d_coords[run_name][:,1] for run_name in d_coords.keys()])
    zpos = np.concatenate([d_coords[run_name][:,2] for run_name in d_coords.keys()])
    
    f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(16,2.5))

    ax1.hist(counts, bins=50, color='black')
    ax2.hist(xpos, bins=50, color='black')
    ax3.hist(ypos, bins=50, color='black')
    ax4.hist(zpos, bins=50, color='black')

    ax1.set_ylabel("Number of tomograms", fontsize=12)
    for ax in [ax2,ax3,ax4]:
        ax.set_ylabel("Number of particles", fontsize=12)
    ax1.set_xlabel("Particles per tomogram", fontsize=12)
    ax2.set_xlabel("X position (Å)", fontsize=12)
    ax3.set_xlabel("Y position (Å)", fontsize=12)
    ax4.set_xlabel("Z position (Å)", fontsize=12)

    f.subplots_adjust(wspace=0.4)
    plt.show()

    f.savefig(config.out_plot, dpi=300, bbox_inches='tight')

    
if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np

from slabpick.dataio import CoPickWrangler
from slabpick.minislab import Minislab


def visualize_merge_pairs(
    run_name: str,
    d_coords1: dict,
    d_coords2: dict,
    d_clusters: dict,
    copick_json: str,
    extract_shape: tuple,
    tomo_type: str = "denoised",
    voxel_spacing: float = 5.0,
    output: str = None,
    col1_title: str = None,
    col2_title: str = None,
):
    """
    Visualize the particle pairs whose coordinates were merged.

    Parameters
    ----------
    run_name: run name
    d_coords1: dict mapping run names to coordinates
    d_coords2: dict mapping run names to coordinates
    d_clusters: dict mapping run names to indices of paired coordinates
    copick_json: copick config file
    extract_shape: tuple for subvolume extraction shape in Angstrom
    tomo_type: type of tomogram to extract subvolumes from
    voxel_spacing: voxel spacing in Angstrom per pixel
    """
    # get volume and convert extraction shape to voxels
    cp_interface = CoPickWrangler(copick_json)
    volume = cp_interface.get_run_tomogram(run_name, voxel_spacing, tomo_type)
    extract_shape = (np.array(extract_shape) / voxel_spacing).astype(int)

    # get and interleave coordinates for merged pairs
    if len(d_clusters[run_name]) == 0:
        raise ValueError("No paired coordinates for this run.")
    c1 = d_coords1[run_name][d_clusters[run_name][:, 0]]
    c2 = d_coords2[run_name][d_clusters[run_name][:, 1]]
    coords = np.vstack((c1, c2))
    reindex = (
        np.vstack((np.arange(len(coords) / 2), np.arange(len(coords) / 2, len(coords))))
        .reshape((-1,), order="F")
        .astype(int)
    )
    coords = coords[reindex]

    # generate minislabs
    mslab = Minislab(extract_shape)
    mslab.generate_slabs(volume, coords / voxel_spacing, run_name)

    # visualize, with first and second columns from d_coords1 and d_coords2, respectively
    f, axs = plt.subplots(int(len(mslab.minislabs) / 2), 2, figsize=(3 * 2, 3 * int(len(mslab.minislabs) / 2)))
    for i in range(len(mslab.minislabs.keys())):
        if int(len(mslab.minislabs) / 2) == 1:
            axs[i].imshow(mslab.minislabs[i], cmap="Greys_r")
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            if col1_title is not None:
                axs[0].set_title(col1_title)
            if col2_title is not None:
                axs[1].set_title(col1_title)
        else:
            axs[i // 2, i % 2].imshow(mslab.minislabs[i], cmap="Greys_r")
            axs[i // 2, i % 2].set_xticks([])
            axs[i // 2, i % 2].set_yticks([])
            if col1_title is not None and i // 2 == 0:
                axs[i // 2, 0].set_title(col1_title)
            if col2_title is not None and i // 2 == 0:
                axs[i // 2, 1].set_title(col2_title)

    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.1)

    if output is not None:
        f.savefig(output, bbox_inches="tight", dpi=300)
        plt.close()

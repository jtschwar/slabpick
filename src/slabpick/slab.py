import glob
import os

import numpy as np
import pandas as pd

from slabpick.dataio import get_voxel_size, load_mrc, save_mrc


def determine_bounds(zthick, zslide, apix, volz):
    """
    Determine the pixel indices along the tomogram's z-depth
    that correspond to the starts of slab bounds. Center the
    indices if there are unused tomogram slices.

    Parameters
    ----------
    zthick: slab thickness in Angstrom
    zslide: sliding window depth in Angstrom
    apix: pixel size in Angstrom
    volz: number of pixels along volume's z axis

    Returns
    -------
    indices: pixel indices corresponding to slab boundaries in z
    """
    if zslide == 0:
        zslide = zthick
    zthick = int(np.around(zthick / apix))
    zslide = int(np.around(zslide / apix))

    indices = np.arange(0, volz, zslide)
    indices = np.around(indices).astype(int)
    zmax = indices[np.argmax(np.where(indices + zthick <= volz))] + zthick
    indices += int((volz - zmax) / 2)
    indices = indices[indices + zthick <= volz]

    return indices


class Slab:
    def __init__(self, zthick: float, zslide: float):
        """
        Set up class to generate slabs from volumes.

        Parameters
        ----------
        zthick: slab thickness in Angstrom
        zslide: sliding window depth in Angstrom
        """
        self.zthick = zthick
        self.zslide = zslide
        self.apix = None
        self.vol_names = []
        self.slab_names = []
        self.scount = 0

    def get_vol_list(
        self,
        in_dir: str,
        include_tags: list = None,
        exclude_tags: list = None,
        filenames: list = None,
    ) -> list:
        """
        Get list of volumes to generate slabs from.

        Parameters
        ----------
        in_dir: input directory of volumes
        include_tag: list of substrings for volume inclusion
        exclude_tag: list of substrings for volume exclusion
        filenames: file prefixes (suffix should be _Vol.mrc) to use
        """
        if exclude_tags is None:
            exclude_tags = ["EVN", "ODD"]
        if include_tags is None:
            include_tags = ["Vol"]

        if filenames is None:
            fnames = glob.glob(os.path.join(in_dir, "*.mrc"))
        else:
            fnames = [os.path.join(in_dir, f"{fn}_Vol.mrc") for fn in filenames]

        for tag in include_tags:
            fnames = [fn for fn in fnames if tag in fn]
        for tag in exclude_tags:
            fnames = [fn for fn in fnames if tag not in fn]
        return fnames

    def slice_volume(
        self,
        volume: np.ndarray,
        bounds: np.ndarray,
        thickness: int,
        out_dir: str,
    ):
        """
        Generate slabs from a single volume.

        Parameters
        ----------
        volume: tomogram
        bounds: starting indices of slices in voxels
        thickness: slice thickness in voxels
        out_dir: output directory
        """
        for _i, istart in enumerate(bounds):
            slab = volume[istart : istart + thickness]
            slab = np.sum(slab, axis=0)
            fn_out = os.path.join(out_dir, f"slab_{self.scount:03d}.mrc")
            save_mrc(slab, fn_out, apix=self.apix)
            self.slab_names.append(fn_out)
            self.scount += 1

    def generate_slabs(
        self,
        in_dir: str,
        out_dir: str,
        include_tags: list = None,
        exclude_tags: list = None,
        filenames: list = None,
    ):
        """
        Generate slabs for each volume in the input directory.
        Volumes are assumed to share a pixel size and z-depth.

        Parameters
        ----------
        out_dir: output directory for slabs
        include_tag: list of substrings for volume inclusion
        exclude_tag: list of substrings for volume exclusion
        filenames: file prefixes (suffix should be _Vol.mrc) to use
        """
        os.makedirs(out_dir, exist_ok=True)
        fnames = self.get_vol_list(in_dir, include_tags, exclude_tags, filenames)
        if len(fnames) == 0:
            raise ValueError("No volumes found to process")

        self.apix = get_voxel_size(fnames[0])
        volz = load_mrc(fnames[0]).shape[0]
        bounds = determine_bounds(self.zthick, self.zslide, self.apix, volz)
        zdepth = int(np.around(self.zthick / self.apix))

        for _i, fn in enumerate(fnames):
            volume = load_mrc(fn)
            self.slice_volume(volume, bounds, zdepth, out_dir)
            self.vol_names.extend(len(bounds) * [fn])

        df = pd.DataFrame(
            {
                "tomogram": self.vol_names,
                "slab": self.slab_names,
                "zstart": np.tile(bounds, len(fnames)),
                "zdepth": len(self.vol_names) * [zdepth],
            },
        )
        df.to_csv(os.path.join(out_dir, "slab_map.csv"))

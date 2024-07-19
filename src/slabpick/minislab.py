import glob
import os
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from slabpick.dataio import (
    CoPickWrangler,
    combine_star_files,
    load_mrc,
    make_stack_starfile,
    read_starfile,
    save_mrc,
)
from slabpick.stacker import invert_contrast, normalize_stack

warnings.simplefilter(action="ignore", category=FutureWarning)


class Minislab:

    """
    Generate "galleries" or mosaics of tiled per-particle subvolume projections,
    or pile them up into a particle stack.
    """

    def __init__(self, extract_shape: tuple[int, int, int]):
        """
        Initialize object.

        Parameters
        ----------
        extract_shape: tuple, subvolume crop shape in Angstrom along (X,Y,Z)
        """
        self.minislabs = {}
        self.num_particles = 0
        self.tomo_names = []
        self.pick_indices = []
        self.shape = extract_shape
        self.row_idx, self.col_idx, self.gallery_idx = [], [], []
        self.vol_means, self.vol_stds = [], []

    def generate_filler(self, fill_shape: tuple[int, int], sigma=1.1) -> np.ndarray:
        """
        Generate filler by sampling from a random distribution based on
        accumulated volume statistics, scaled by the number of z-pixels
        since this will fill the projected slab. A Gaussian filter can
        optionally be applied to the resulting filler region.

        Parameters
        ----------
        fill_shape: tuple, 2d shape to fill
        sigma: float, standard deviation for Gaussian kernel filter in pixels

        Returns
        -------
        np.array of filler region
        """
        filler = self.shape[2] * np.random.normal(
            loc=np.mean(np.array(self.vol_means)),
            scale=np.mean(np.array(self.vol_stds)),
            size=fill_shape,
        )
        return gaussian_filter(filler, sigma=sigma)

    def generate_slabs(self, vol: np.ndarray, coords: np.ndarray, vol_name: str):
        """
        Generate per-particle minislabs, i.e. a 2d projection along z
        centered around each particle.

        Parameters
        ----------
        vol: numpy or zarr array, tomogram
        coords: np.ndarray (n_coords, 3), xyz coordinates in pixels
        vol_name: str, name of volume
        """
        self.vol_means.append(np.mean(vol))
        self.vol_stds.append(np.std(vol))

        for i, c in enumerate(coords):
            # crop particle-centered subvolume and project along z, ensuring even dimensions once projected
            c = c.astype(int)
            xstart, xend = c[2] - int(self.shape[2] / 2), c[2] + int(self.shape[2] / 2)
            ystart, yend = c[1] - int(self.shape[1] / 2), c[1] + int(self.shape[1] / 2)
            zstart, zend = c[0] - int(self.shape[0] / 2), c[0] + int(self.shape[0] / 2)
            if xstart < 0:
                xstart = 0
            if ystart < 0:
                ystart = 0
            if zstart < 0:
                zstart = 0
            subvol = np.array(vol[xstart:xend, ystart:yend, zstart:zend])

            if np.any(np.array(subvol.shape) == 0) or np.any(c < 0):
                print(
                    f"Skipping entry with coordinates {coords[i]} due to out of bounds error",
                )
                continue
            projection = np.sum(subvol, axis=0)

            # fill any missing rows/columns if particle is along tomogram x/y edge
            if projection.shape[0] != self.shape[1]:
                edge = projection.shape[0]
                filler = self.generate_filler(
                    (self.shape[1] - edge, projection.shape[1]),
                )
                if ystart == 0:
                    projection = np.vstack((filler, projection))
                    if self.shape[1] - edge > 4:
                        projection[
                            self.shape[1] - edge - 1 : self.shape[1] - edge + 1
                        ] = gaussian_filter(
                            projection[
                                self.shape[1] - edge - 2 : self.shape[1] - edge + 2
                            ],
                            sigma=1.1,
                        )[
                            1:-1
                        ]
                else:
                    projection = np.vstack((projection, filler))
                    if self.shape[1] - edge > 4:
                        projection[edge - 1 : edge + 1] = gaussian_filter(
                            projection[edge - 2 : edge + 2], sigma=1.1,
                        )[1:-1]

            if projection.shape[1] != self.shape[0]:
                edge = projection.shape[1]
                filler = self.generate_filler(
                    (projection.shape[0], self.shape[0] - edge),
                )
                if zstart == 0:
                    projection = np.hstack((filler, projection))
                    if self.shape[0] - edge > 4:
                        projection[
                            :, self.shape[0] - edge - 1 : self.shape[0] - edge + 1,
                        ] = gaussian_filter(
                            projection[
                                :, self.shape[0] - edge - 2 : self.shape[0] - edge + 2,
                            ],
                            sigma=1.1,
                        )[
                            :, 1:-1,
                        ]
                else:
                    projection = np.hstack((projection, filler))
                    if self.shape[0] - edge > 4:
                        projection[:, edge - 1 : edge + 1] = gaussian_filter(
                            projection[:, edge - 2 : edge + 2],
                            sigma=1.1,
                        )[:, 1:-1]

            self.minislabs[self.num_particles] = projection
            self.tomo_names.append(vol_name)
            self.pick_indices.append(i)
            self.num_particles += 1

    def make_one_gallery(
        self, gshape: tuple[int, int], key_list: list,
    ) -> tuple[np.ndarray, list, list]:
        """
        Generate a single gallery from select minislabs.

        Parameters
        ----------
        gshape: number of particles per gallery edge
        key_list: list of minislab keys to tile

        Returns
        -------
        gallery: montage of particle minislabs
        row_idx: list of particles' row indices
        col_idx: list of particles' column indices
        """
        if len(key_list) > np.prod(gshape):
            raise IndexError("Number of minislabs exceeds number of gallery tiles.")

        pshape = self.minislabs[key_list[0]].shape
        gallery = np.zeros((gshape[0] * pshape[0], gshape[1] * pshape[1])).astype(
            np.float32,
        )
        row_idx, col_idx = [], []

        counter = 0
        for i in range(gshape[0]):
            for j in range(gshape[1]):
                if counter > len(key_list) - 1:
                    filler = self.generate_filler((pshape[0], pshape[1]))
                    gallery[
                        i * pshape[0] : i * pshape[0] + pshape[0],
                        j * pshape[1] : j * pshape[1] + pshape[1],
                    ] = filler
                else:
                    gallery[
                        i * pshape[0] : i * pshape[0] + pshape[0],
                        j * pshape[1] : j * pshape[1] + pshape[1],
                    ] = self.minislabs[key_list[counter]]
                    row_idx.append(i)
                    col_idx.append(j)
                counter += 1

        return gallery, row_idx, col_idx

    def make_galleries(
        self,
        gshape: tuple[int, int],
        apix: float,
        outdir: str,
        one_per_vol: bool = False,
    ):
        """
        Generate galleries by tiling the minislabs and save the resulting projected
        particle mosaics in mrc format and a corresponding bookkeeping file as csv.

        Parameters
        ----------
        gshape: tuple, number of particles along (rows, cols)
        apix: float, pixel size in Angstrom
        outdir: str, output directory for gallery mrcs and bookkeeping csv
        one_per_vol: bool, if True generate one gallery per tomogram
        """
        if len(self.minislabs) == 0:
            raise Exception("No slabs have been generated.")
        os.makedirs(outdir, exist_ok=True)

        if not one_per_vol:
            n_mgraphs = len(self.minislabs) // np.prod(gshape) + 1
        else:
            n_mgraphs = len(set(self.tomo_names))
            unique_names = pd.unique(np.array(self.tomo_names))

        for nm in range(n_mgraphs):
            if one_per_vol:
                key_list = list(
                    np.where(np.array(self.tomo_names) == unique_names[nm])[0],
                )
                filename = unique_names[nm]
            else:
                end_key = np.min(
                    [np.prod(gshape) * (nm + 1), max(list(self.minislabs.keys())) + 1],
                )
                key_list = list(np.arange(nm * np.prod(gshape), end_key).astype(int))
                filename = "particles"
            gallery, row_idx, col_idx = self.make_one_gallery(gshape, key_list)
            save_mrc(
                gallery, os.path.join(outdir, f"{filename}_{nm:03d}.mrc"), apix=apix,
            )
            self.row_idx.extend(row_idx)
            self.col_idx.extend(col_idx)
            self.gallery_idx.extend(len(row_idx) * [nm])

        # save bookkeeping file in csv format
        df = pd.DataFrame(
            {
                "tomogram": self.tomo_names,
                "particle": self.pick_indices,
                "gallery": self.gallery_idx,
                "row": self.row_idx,
                "col": self.col_idx,
            },
        )
        df.to_csv(os.path.join(outdir, "particle_map.csv"), index=False)

    def make_stacks(
        self,
        apix: float,
        outdir: str,
        normalize: bool = True,
        invert: bool = True,
        radius: float = 0.9,
    ):
        """
        Generate particle stacks from the minislabs, optionally normalizing
        and inverting the contrast.

        Parameters
        ----------
        apix: float, pixel size in Angstrom
        outdir: str, output directory
        normalize: bool, normalize stacks
        invert: bool, invert contrast
        radius: float, fractional tile radius for normalization
        """
        if len(self.minislabs) == 0:
            raise Exception("No slabs have been generated.")
        os.makedirs(outdir, exist_ok=True)

        pshape = self.minislabs[0].shape
        stack = np.zeros((len(self.minislabs), pshape[0], pshape[1])).astype(np.float32)
        for i in range(len(self.minislabs)):
            stack[i] = self.minislabs[i]

        if normalize:
            stack = normalize_stack(stack, radius=radius)
        if invert:
            stack = invert_contrast(stack)

        save_mrc(stack, os.path.join(outdir, "particles.mrcs"), apix=apix)
        df = pd.DataFrame({"tomogram": self.tomo_names, "particle": self.pick_indices})
        df.to_csv(os.path.join(outdir, "particle_map.csv"), index=False)


def make_particle_projections(
    in_vol: str,
    in_coords: str,
    out_dir: str,
    extract_shape: tuple,
    voxel_spacing: float,
    coords_scale: float = 1,
    col_name: str = "rlnMicrographName",
    tomo_type: str = None,
    particle_name: str = None,
    user_id: str = None,
    session_id: str = None,
    as_gallery: bool = True,
    as_stack: bool = False,
    gallery_shape: tuple = (16, 15),
    one_per_vol: bool = False,
    normalize: bool = True,
    invert: bool = True,
    radius: float = 0.9,
    live: bool = False,
    t_interval: float = 300,
    t_exit: float = 1800,
):
    """
    Live version of the generate_from_starfile function.

    Generate galleries or stacks based on coordinates in a starfile,
    with the volumes loaded either through copick or from a directory
    containing mrc files. Coordinates should be scaled to Angstroms.

    Parameters
    ----------
    in_vol: copick configuration file or directory of mrc files
    in_coords: starfile(s) of particle coordinates
    out_dir: output directory for mrc(s) and bookkeeping file
    extract_shape: subvolume extraction shape in Angstrom
    voxel_spacing: voxel spacing in Angstrom of tomograms
    coords_scale: factor to apply to coordinates and/or to store in stack starfile
    col_name: tomogram column label if coords from starfiles
    tomo_type: tomogram type if volumes from copick
    particle_name: particle name if coords from copick
    user_id: user ID if coords from copick
    session_id: session ID if coords from copick
    as_gallery: output minislabs as galleries
    as_stack: output minislabs as particle stacks
    gallery_shape: number of particles along gallery (row,col)
    one_per_vol: generate one gallery per tomogram
    normalize: normalize particle stack
    invert: invert contrast
    radius: fractional radius for normalization purposes
    t_interval: interval in seconds before checking for new files
    t_exit: interval in seconds after which to exit if no new files found
    """
    start_time = time.time()
    processed = []
    if not live:
        t_exit, t_interval = 0, 0

    # set up montage class, forcing extraction shape to have even dimensions
    extract_shape = (np.array(extract_shape) / voxel_spacing).astype(int)
    odd_dims = np.where(extract_shape % 2)[0]
    if len(odd_dims) > 0:
        extract_shape[odd_dims] -= 1
    montage = Minislab(tuple(extract_shape))

    while True:
        fnames = glob.glob(in_coords)
        fnames = [fn for fn in fnames if fn not in processed]
        print(time.strftime("%X %x %Z"), f": Found {len(fnames)} new files to process")

        if len(fnames) > 0:
            # handle different coordinate entrypoints
            if len(fnames) == 1 and os.path.splitext(in_coords)[-1] == ".json":
                cp_interface = CoPickWrangler(in_coords)
                coords = cp_interface.get_all_coords(particle_name, session_id, user_id)
            elif len(fnames) == 1 and os.path.splitext(in_coords)[-1] == ".star":
                coords = read_starfile(
                    fnames[0], coords_scale=coords_scale, col_name=col_name,
                )
            else:
                coords = combine_star_files(
                    fnames, coords_scale=coords_scale, col_name=col_name,
                )

            # handle different volume entrypoints
            if os.path.isfile(in_vol):
                load_method = "copick"
                cp_interface = CoPickWrangler(in_vol)
            else:
                load_method = "mrc"

            # generate particle projections from each run
            for run_name in coords:
                if load_method == "copick":
                    volume = cp_interface.get_run_tomogram(
                        run_name, voxel_spacing, tomo_type,
                    )
                if load_method == "mrc":
                    vol_name = os.path.join(in_vol, f"{run_name}.mrc")
                    volume = load_mrc(vol_name)

                coords_pixels = coords[run_name] / voxel_spacing
                montage.generate_slabs(volume, coords_pixels, run_name)

            processed.extend(fnames)
            start_time = time.time()

        time.sleep(t_interval)
        t_elapsed = time.time() - start_time
        if t_elapsed > t_exit:
            break
    print(f"Processed {len(processed)} sets of picks/tomograms.")

    # generate galleries and/or a particle stack
    if as_gallery:
        gallery_out_dir = os.path.join(out_dir, "gallery") if as_gallery else out_dir
        montage.make_galleries(
            gallery_shape, voxel_spacing, gallery_out_dir, one_per_vol,
        )
    if as_stack:
        stack_out_dir = os.path.join(out_dir, "stack") if as_gallery else out_dir
        montage.make_stacks(
            voxel_spacing,
            stack_out_dir,
            normalize=normalize,
            invert=invert,
            radius=radius,
        )
        make_stack_starfile(
            os.path.join(stack_out_dir, "particles.mrcs"),
            os.path.join(stack_out_dir, "particles.star"),
            apix=coords_scale,
        )





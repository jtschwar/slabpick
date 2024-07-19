import json
import os

import copick
import mrcfile
import numpy as np
import pandas as pd
import starfile
import zarr
from copick.models import CopickLocation, CopickPoint


def load_mrc(filename: str) -> np.ndarray:
    """
    Load the data in an mrc file into a numpy array.

    Parameters
    ----------
    filename: str, path to mrc file

    Returns
    -------
    np.ndarray, image or volume
    """
    with mrcfile.open(filename, "r", permissive=True) as mrc:
        return mrc.data


def save_mrc(
    data: np.ndarray,
    filename: str,
    overwrite: bool = True,
    apix: float = None,
):
    """
    Save a numpy array to mrc format.

    Parameters
    ----------
    data: np.ndarray, image or volume
    filename: str, save path
    overwrite: bool, overwrite filename if already exists
    apix: float, pixel size in Angstrom
    """
    if data.dtype != np.dtype("float32"):
        data = data.astype(np.float32)
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if apix:
            mrc.voxel_size = apix


def get_voxel_size(filename: str, isotropic: bool = True) -> float:
    """
    Extract voxel size from mrc file.

    Parameters
    ----------
    filename: str, path to mrc file
    isotropic: bool, extract single value assuming isotropic pixel size

    Returns
    -------
    apix: float, pixel size in Angstrom
    """
    apix = mrcfile.open(filename).voxel_size.tolist()
    if isotropic:
        return apix[0]
    return apix


def make_starfile(d_coords: dict, out_file: str, coords_scale: float = 1):
    """
    Write a Relion star file from the input coordinates. To convert
    from copick to Relion 4 conventions, coords_scale_factor should
    be the inverse tilt-series (not tomogram) pixel size in Angstrom.

    Parameters
    ----------
    d_coords: tomogram mapped to coordinates and optional score column
    out_file: output star file to write
    coords_scale: multiplicative factor to apply to coordinates
    """
    rln = {}
    rln["rlnTomoName"] = np.concatenate(
        [d_coords[tomo].shape[0] * [tomo] for tomo in d_coords],
    ).ravel()
    rln["rlnCoordinateX"] = (
        np.concatenate([d_coords[tomo][:, 0] for tomo in d_coords]).ravel()
        * coords_scale
    )
    rln["rlnCoordinateY"] = (
        np.concatenate([d_coords[tomo][:, 1] for tomo in d_coords]).ravel()
        * coords_scale
    )
    rln["rlnCoordinateZ"] = (
        np.concatenate([d_coords[tomo][:, 2] for tomo in d_coords]).ravel()
        * coords_scale
    )
    if np.all(np.array([d_coords[tomo].shape[1] for tomo in d_coords]) == 4):
        rln["rlnScore"] = np.concatenate(
            [d_coords[tomo][:, 3] for tomo in d_coords],
        ).ravel()
    for key in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]:
        rln[key] = np.zeros(len(rln["rlnCoordinateX"]))
    rln["rlnTomoManifoldIndex"] = np.ones(len(rln["rlnCoordinateX"])).astype(int)
    rln["rlnTomoParticleId"] = np.arange(len(rln["rlnCoordinateX"])).astype(int)

    rln_df = pd.DataFrame.from_dict(rln)
    starfile.write(rln_df, out_file)


def make_stack_starfile(
    in_stack: str,
    out_star: str,
    apix: float,
    ctf_precomputed: bool = True,
    voltage: float = 300.0,
    cs: float = 2.7,
    amplitude_contrast: float = 0.1,
):
    """
    Generate a Relion-compatible starfile for a particle stack.

    Parameters
    ----------
    in_stack: input particle stack in .mrcs format
    out_star: output star file
    apix: tilt-series pixel size
    ctf_precomputed: CTF has already been corrected
    voltage: microscope voltage
    cs: spherical aberration coefficient
    amplitude contrast: amplitude contrast
    """
    fname = os.path.basename(in_stack)
    apix_tomo = get_voxel_size(in_stack)
    n_particles, stack_dim, stack_dim1 = load_mrc(in_stack).shape
    assert stack_dim == stack_dim1

    grp_optics = {}
    grp_optics["rlnVoltage"] = [voltage]
    grp_optics["rlnSphericalAberration"] = [cs]
    grp_optics["rlnAmplitudeContrast"] = [amplitude_contrast]
    grp_optics["rlnTomoTiltSeriesPixelSize"] = [apix]
    grp_optics["rlnOpticsGroup"] = [1]
    grp_optics["rlnOpticsGroupName"] = ["optics1"]
    grp_optics["rlnCtfDataAreCtfPremultiplied"] = [1 if ctf_precomputed else 0]
    grp_optics["rlnImageDimensionality"] = [2]
    grp_optics["rlnImagePixelSize"] = [apix_tomo]
    grp_optics["rlnImageSize"] = [stack_dim]

    grp_particles = {}
    grp_particles["rlnImageName"] = [f"{num}@{fname}" for num in range(n_particles)]
    grp_particles["rlnOpticsGroup"] = n_particles * [1]
    grp_particles["rlnGroupNumber"] = n_particles * [1]

    dstack = {}
    dstack["optics"] = pd.DataFrame.from_dict(grp_optics)
    dstack["particles"] = pd.DataFrame.from_dict(grp_particles)

    starfile.write(dstack, out_star)


def read_starfile(
    in_star: str,
    col_name: str = "rlnTomoName",
    coords_scale: float = 1,
    extra_col_name: str = None,
) -> dict:
    """
    Extract tomogram-associated coordinates from a starfile.

    Parameters
    ----------
    in_star: Relion-4 style starfile
    col_name: column name for the tomograms
    coords_scale: multiplicative factor to apply to coordinates
    extra_col_name: column to extract and save as 4th column

    Returns
    -------
    d_coords: dictionary of tomogram name: particle XYZ coordinates
    """
    particles = starfile.read(in_star)
    if len(particles)==0:
        print(f"Warning: {in_star} appears to be an empty starfile")
        return {}
    if isinstance(particles, dict):
        particles = particles["particles"]

    tomo_names = np.unique(particles[col_name].values)
    d_coords = {}
    for tomo in tomo_names:
        tomo_indices = np.where(particles[col_name].values == tomo)[0]
        d_coords[tomo] = (
            np.array(
                [
                    particles.rlnCoordinateX.iloc[tomo_indices],
                    particles.rlnCoordinateY.iloc[tomo_indices],
                    particles.rlnCoordinateZ.iloc[tomo_indices],
                ],
            ).T
            * coords_scale
        )
        if extra_col_name is not None:
            d_coords[tomo] = np.hstack(
                (
                    d_coords[tomo],
                    particles[extra_col_name].iloc[tomo_indices].values[:, np.newaxis],
                ),
            )
    return d_coords


def combine_star_files(
    in_star: list,
    col_name: str = "rlnTomoName",
    coords_scale: float = 1,
) -> dict:
    """
    Combine multiple star files into a single dictionary
    of coordinates associated with tomograms.

    Parameters
    ----------
    in_star: list of star files to merge
    col_name: tomogram column name
    coords_scale: multiplicative factor to apply to coordinates

    Returns
    -------
    d_coords: tomogram mapped to particle coordinates
    """
    d_coords_list = [
        read_starfile(star_path, col_name=col_name, coords_scale=coords_scale)
        for star_path in in_star
    ]
    d_coords = {}
    for d in d_coords_list:
        for k, v in d.items():
            d_coords.setdefault(k, []).append(v)

    d_coords = {key: d_coords[key] for key in d_coords}
    for key in d_coords:
        if len(d_coords[key]) > 1:
            print(f"Warning! {key} spanned multiple star files")
        d_coords[key] = np.vstack(d_coords[key])

    return d_coords


def read_copick_json(fname: str) -> np.ndarray:
    """
    Read coordinates from an individual copick-formatted json file.

    Parameters
    ----------
    fname: str, path to json file

    Returns
    -------
    np.ndarray, xyz coordinates in Angstrom
    """
    with open(fname) as f:
        points = json.load(f)["points"]
    locs = [points[i]["location"] for i in range(len(points))]
    return np.array(
        [(locs[i]["x"], locs[i]["y"], locs[i]["z"]) for i in range(len(locs))],
    )


class CoPickWrangler:
    """
    Utilties to extract information from a copick project.
    copick documentation: https://uermel.github.io/copick/
    """

    def __init__(self, config: str):
        """
        Parameters
        ----------
        config: str, copick configuration file
        """
        self.root = copick.from_file(config)

    def get_run_coords(
        self,
        run_name: str,
        particle_name: str,
        session_id: str,
        user_id: str,
    ) -> np.ndarray:
        """
        Extract coordinates for a partciular run.

        Parameters
        ----------
        run_name: str, name of run
        particle_name: str, name of particle
        session_id: str, name of session
        user_id: str, name of user

        Returns
        -------
        np.array, shape (n_coords, 3) of coordinates in Angstrom
        """
        pick = self.root.get_run(run_name).get_picks(
            particle_name,
            session_id=session_id,
            user_id=user_id,
        )
        if len(pick) == 0:
            print(f"Picks json file may be missing for run: {run_name}")
            return np.empty(0)
        pick = pick[0]
        return np.array(
            [(p.location.x, p.location.y, p.location.z) for p in pick.points],
        )

    def get_run_tomogram(
        self,
        run_name: str,
        voxel_spacing: float,
        tomo_type: str,
    ) -> zarr.core.Array:
        """
        Get tomogram for a particular run.

        Parameters
        ----------
        run_name: str, name of run
        voxel_spacing: float, voxel spacing in Angstrom
        tomo_type: str, type of tomogram, e.g. denoised

        Returns
        -------
        array: zarr.core.Array, tomogram volume
        """
        run = self.root.get_run(run_name)
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(
            tomo_type=tomo_type,
        )
        arrays = list(zarr.open(tomogram.zarr(), "r").arrays())
        _, array = arrays[0]  # 0 corresponds to unbinned
        return array

    def get_run_names(self) -> list:
        """
        Extract all run names.

        Returns
        -------
        list, names of all available runs
        """
        return [run.name for run in self.root.runs]

    def get_all_coords(self, particle_name: str, session_id: str, user_id: str) -> dict:
        """
        Extract all coordinates for a particle across a dataset.

        Parameters
        ----------
        particle_name: str, name of particle
        session_id: str, name of session
        user_id: str, name of user

        Returns
        -------
        d_coords: dict, mapping of run name to particle coordinates
        """
        runs = self.get_run_names()
        d_coords = {}
        for run in runs:
            coords = self.get_run_coords(run, particle_name, session_id, user_id)
            if len(coords) > 0:
                d_coords[run] = coords
        return d_coords


def coords_to_copick(
    root: copick.models.CopickRoot,
    d_coords: dict,
    particle_name: str,
    session_id: str,
    user_id: str,
):
    """
    Convert a set of coordinates to copick format.

    Parameters
    ----------
    root: copick.impl.filesystem.CopickRootFSSpec object
    d_coords: dict, run_names mapped to coordinates
    particle_name: str, copick name of particle
    session_id: str, session id
    user_id: str, user id
    """
    for run_name in d_coords:
        pts = []
        coords = d_coords[run_name]
        for sc in coords:
            if coords.shape[1] == 3:
                pts.append(
                    CopickPoint(location=CopickLocation(x=sc[0], y=sc[1], z=sc[2])),
                )
            else:
                pts.append(
                    CopickPoint(
                        location=CopickLocation(x=sc[0], y=sc[1], z=sc[2]),
                        score=sc[3],
                    ),
                )

        run = root.get_run(run_name)
        if run is None:
            run = root.new_run(run_name)

        new_picks = run.new_picks(
            object_name=particle_name,
            session_id=session_id,
            user_id=user_id,
        )
        new_picks.points = pts
        new_picks.store()

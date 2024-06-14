from copick.impl.filesystem import CopickRootFSSpec
import numpy as np
import mrcfile
import zarr
import pandas as pd
import starfile

def load_mrc(filename):
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

def save_mrc(data, filename, overwrite=True, apix=None):
    """ 
    Save a numpy array to mrc format. 

    Parameters
    ----------
    data: np.ndarray, image or volume
    filename: str, save path
    overwrite: bool, overwrite filename if already exists
    apix: float, pixel size in Angstrom
    """
    if data.dtype != np.dtype('float32'):
        data = data.astype(np.float32)
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if apix:
            mrc.voxel_size = apix

def get_voxel_size(filename, isotropic=True):
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

def make_starfile(d_coords: dict, out_file: str, coords_scale: float=1):
    """
    Write a Relion star file from the input coordinates. To convert
    from copick to Relion 4 conventions, coords_scale_factor should 
    be the inverse tilt-series (not tomogram) pixel size in Angstrom.
    
    Parameters
    ----------
    d_coords: dict, tomogram name: particle XYZ coordinates
    out_file: str, output star file to write
    coords_scale: float, multiplicative factor to apply to coordinates
    """
    rln = {}
    rln['rlnTomoName'] = np.concatenate([d_coords[tomo].shape[0]*[tomo] for tomo in d_coords.keys()]).ravel()
    rln['rlnCoordinateX'] = np.concatenate([d_coords[tomo][:,0] for tomo in d_coords.keys()]).ravel() * coords_scale
    rln['rlnCoordinateY'] = np.concatenate([d_coords[tomo][:,1] for tomo in d_coords.keys()]).ravel() * coords_scale
    rln['rlnCoordinateZ'] = np.concatenate([d_coords[tomo][:,2] for tomo in d_coords.keys()]).ravel() * coords_scale
    for key in ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']:
        rln[key] = np.zeros(len(rln['rlnCoordinateX']))
    rln['rlnTomoManifoldIndex'] = np.ones(len(rln['rlnCoordinateX'])).astype(int)
    rln['rlnTomoParticleId'] = np.arange(len(rln['rlnCoordinateX'])).astype(int)

    rln_df = pd.DataFrame.from_dict(rln)
    starfile.write(rln_df, out_file)

def read_starfile(in_star: str, col_name: str = "rlnTomoName", coords_scale: float=1) -> dict:
    """
    Extract tomogram-associated coordinates from a starfile.
    
    Parameters
    ----------
    in_star: Relion-4 style starfile
    col_name: column name for the tomograms
    coords_scale: float, multiplicative factor to apply to coordinates 

    Returns
    -------
    d_coords: dictionary of tomogram name: particle XYZ coordinates
    """
    particles = starfile.read(in_star)
    if type(particles) == dict:
        particles = particles['particles']

    tomo_names = np.unique(particles[col_name].values)
    d_coords = {}
    for tomo in tomo_names:
        tomo_indices = np.where(particles[col_name].values==tomo)[0]
        d_coords[tomo] = np.array([particles.rlnCoordinateX.iloc[tomo_indices],
                                   particles.rlnCoordinateY.iloc[tomo_indices],
                                   particles.rlnCoordinateZ.iloc[tomo_indices]]).T * coords_scale
    return d_coords
    
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
        self.root = CopickRootFSSpec.from_file(config)

    def get_run_coords(self, run_name: str, particle_name: str, session_id: str) -> np.ndarray:
        """
        Extract coordinates for a partciular run.

        Parameters
        ----------
        run_name: str, name of run 
        particle_name: str, name of particle
        session_id: str, name of session   

        Returns
        -------
        np.array, shape (n_coords, 3) of coordinates in Angstrom
        """
        pick = self.root.get_run(run_name).get_picks(particle_name, session_id=session_id)
        if len(pick) == 0:
            print(f"Picks json file may be missing for run: {run_name}")
            return np.empty(0)
        pick = pick[0]
        return np.array([(p.location.x, p.location.y, p.location.z) for p in pick.points])

    def get_run_tomogram(self, run_name: str, voxel_spacing: float, tomo_type: str) -> zarr.core.Array:
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
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type=tomo_type)
        arrays = list(zarr.open(tomogram.zarr(), "r").arrays())
        _, array = arrays[0] # 0 corresponds to unbinned
        return array
    
    def get_run_names(self) -> list:
        """
        Extract all run names.

        Returns
        -------
        list, names of all available runs
        """
        return [run.name for run in self.root.runs]

    def get_all_coords(self, particle_name: str, session_id: str) -> dict:
        """
        Extract all coordinates for a particle across a dataset.

        Parameters
        ----------
        particle_name: str, name of particle   
        session_id: str, name of session

        Returns
        -------
        d_coords: dict, mapping of run name to particle coordinates
        """
        runs = self.get_run_names()
        d_coords = {}
        for run in runs:
            coords = self.get_run_coords(run, particle_name, session_id)
            if len(coords) > 0:
                d_coords[run] = coords
        return d_coords

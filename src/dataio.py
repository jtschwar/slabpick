from copick.impl.filesystem import CopickRootFSSpec
import numpy as np
import mrcfile
import zarr

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

class CoPickWrangler:
    """
    Utilties to extract information from a copick project. 
    copick documentation: https://uermel.github.io/copick/
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config: copick configuration file 
        """
        self.root = CopickRootFSSpec.from_file(config)

    def get_run_coords(self, run_name, particle_name):
        """
        Extract coordinates for a partciular run.

        Parameters
        ----------
        run_name: str, name of run 
        particle_name: str, name of particle

        Returns
        -------
        np.array, shape (n_coords, 3) of coordinates in Angstrom
        """
        try:
            pick = self.root.get_run(run_name).get_picks(particle_name)[0]
        except AttributeError:
            print("Run or particle name not found")
            return np.empty(0)
        return np.array([(p.location.x, p.location.y, p.location.z) for p in pick.points])

    def get_run_tomogram(self, run_name, voxel_spacing, tomo_type):
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
    
    def get_run_names(self):
        """
        Extract all run names.

        Returns
        -------
        list, names of all available runs
        """
        return [run.name for run in self.root.runs]

    def get_all_coords(self, particle_name):
        """
        Extract all coordinates for a particle across a dataset.

        Parameters
        ----------
        particle_name: str, name of particle   

        Returns
        -------
        d_coords: dict, mapping of run name to particle coordinates
        """
        runs = self.get_run_names()
        d_coords = {}
        for run in runs:
            coords = self.get_run_coords(run, 'apo-ferritin')
            if len(coords) > 0:
                d_coords[run] = coords
        return d_coords

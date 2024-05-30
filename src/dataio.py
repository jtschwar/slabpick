import numpy as np
import mrcfile

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

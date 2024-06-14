import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np
import starfile
import re
import os
from dataio import *

class Minislab:
    
    """
    Generate "galleries" or mosaics of tiled per-particle subvolume projections,
    or pile them up into a particle stack.
    """    
    
    def __init__(self, extract_shape: tuple[int,int,int]):
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
        
    def generate_filler(self, fill_shape: tuple[int,int], sigma=1.1)-> np.ndarray:
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
        filler = self.shape[2]*np.random.normal(loc=np.mean(np.array(self.vol_means)), 
                                                scale=np.mean(np.array(self.vol_stds)), 
                                                size=fill_shape)
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
        
        for i,c in enumerate(coords):
            # crop particle-centered subvolume and project along z, ensuring even dimensions once projected
            c = c.astype(int)
            xstart, xend = c[2]-int(self.shape[2]/2), c[2]+int(self.shape[2]/2) 
            ystart, yend = c[1]-int(self.shape[1]/2), c[1]+int(self.shape[1]/2) if self.shape[1]%2==0 else c[1]+int(self.shape[1]/2) + 1
            zstart, zend = c[0]-int(self.shape[0]/2), c[0]+int(self.shape[0]/2) if self.shape[0]%2==0 else c[0]+int(self.shape[0]/2) + 1
            if xstart < 0: xstart = 0
            if ystart < 0: ystart = 0
            if zstart < 0: zstart = 0
            subvol = np.array(vol[xstart:xend,ystart:yend,zstart:zend])
            
            if np.any(np.array(subvol.shape)==0) or np.any(c<0):
                print(f"Skipping entry with coordinates {coords[i]} due to out of bounds error")
                continue
            projection = np.sum(subvol, axis=0)
            
            # fill any missing rows/columns if particle is along tomogram x/y edge 
            if projection.shape[0] != self.shape[1]:
                edge = projection.shape[0]
                filler = self.generate_filler((self.shape[1]-edge, projection.shape[1]))
                if ystart == 0:
                    projection = np.vstack((filler, projection))
                    if self.shape[1]-edge > 4:
                        projection[self.shape[1]-edge-1:self.shape[1]-edge+1] = gaussian_filter(projection[self.shape[1]-edge-2:self.shape[1]-edge+2], sigma=1.1)[1:-1]
                else:
                    projection = np.vstack((projection, filler))
                    if self.shape[1]-edge > 4:
                        projection[edge-1:edge+1] = gaussian_filter(projection[edge-2:edge+2], sigma=1.1)[1:-1]
                        
            if projection.shape[1] != self.shape[0]:
                edge = projection.shape[1]
                filler = self.generate_filler((projection.shape[0], self.shape[0]-edge))
                if zstart == 0:
                    projection = np.hstack((filler, projection))
                    if self.shape[0]-edge > 4:
                        projection[:,self.shape[0]-edge-1:self.shape[0]-edge+1] = gaussian_filter(projection[:,self.shape[0]-edge-2:self.shape[0]-edge+2], sigma=1.1)[:,1:-1]
                else:
                    projection = np.hstack((projection, filler))
                    if self.shape[0]-edge > 4:
                        projection[:,edge-1:edge+1] = gaussian_filter(projection[:,edge-2:edge+2], sigma=1.1)[:,1:-1]
            
            self.minislabs[self.num_particles] = projection
            self.tomo_names.append(vol_name)
            self.pick_indices.append(i)
            self.num_particles += 1

    def make_one_gallery(self,
                         gshape: tuple[int, int], 
                         key_list: list) -> tuple[np.ndarray, list, list]:
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
        gallery = np.zeros((gshape[0]*pshape[0], gshape[1]*pshape[1])).astype(np.float32)
        row_idx, col_idx = [], []
        
        counter = 0
        for i in range(gshape[0]):
            for j in range(gshape[1]):
                if counter > len(key_list)-1:
                    filler = self.generate_filler((pshape[0], pshape[1]))
                    gallery[i*pshape[0]:i*pshape[0]+pshape[0], j*pshape[1]:j*pshape[1]+pshape[1]] = filler
                else:
                    gallery[i*pshape[0]:i*pshape[0]+pshape[0], j*pshape[1]:j*pshape[1]+pshape[1]] = self.minislabs[key_list[counter]]
                    row_idx.append(i)
                    col_idx.append(j)
                counter += 1

        return gallery, row_idx, col_idx

    def make_galleries(self, 
                       gshape: tuple[int,int], 
                       apix: float, 
                       outdir:str, 
                       one_per_vol: bool = False):
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
        if len(self.minislabs)==0:
            raise Exception("No slabs have been generated.")
        os.makedirs(outdir, exist_ok=True)
        
        if not one_per_vol:
            n_mgraphs = len(self.minislabs) // np.prod(gshape) + 1
        else:
            n_mgraphs = len(set(self.tomo_names))
            unique_names = pd.unique(np.array(self.tomo_names))
            
        for nm in range(n_mgraphs):
            if one_per_vol:
                key_list = list(np.where(np.array(self.tomo_names) == unique_names[nm])[0])
                filename = unique_names[nm]
            else:
                end_key = np.min([np.prod(gshape)*(nm+1), max(list(self.minislabs.keys()))+1])
                key_list = list(np.arange(nm*np.prod(gshape), end_key).astype(int))
                filename = 'particles'
            gallery, row_idx, col_idx = self.make_one_gallery(gshape, key_list)
            save_mrc(gallery, os.path.join(outdir, f"{filename}_{nm:03d}.mrc"), apix=apix)
            self.row_idx.extend(row_idx)
            self.col_idx.extend(col_idx)
            self.gallery_idx.extend(len(row_idx)*[nm])

        # save bookkeeping file in csv format
        df = pd.DataFrame({'tomogram': self.tomo_names,
                           'particle': self.pick_indices,
                           'gallery': self.gallery_idx,
                           'row': self.row_idx,
                           'col': self.col_idx})
        df.to_csv(os.path.join(outdir, f"particle_map.csv"), index=False)

def generate_from_copick(config: str,
                         out_dir: str,
                         particle_name: str,
                         session_id: str,
                         voxel_spacing: float,
                         tomo_type: str,
                         extract_shape: tuple,
                         gallery_shape: tuple = (16,15),
                         one_per_vol: bool = False):
    """
    Generate galleries from a copick project.

    Parameters
    ----------
    config: copick configuration file 
    out_dir: directory to write galleries and bookkeeping file to
    particle_name: particle name
    session_id: session id
    voxel_spacing: voxel spacing in Angstrom
    tomo_type: type of tomogram, e.g. 'denoised'
    extract_shape: subvolume extraction shape in Angstrom
    gallery_shape: number of particles along gallery (row,col)
    """
    cp_interface = CoPickWrangler(config)
    coords = cp_interface.get_all_coords(particle_name, session_id)
    extract_shape = tuple((np.array(extract_shape)/voxel_spacing).astype(int))
    
    montage = Minislab(extract_shape)
    for run_name in coords.keys():
        print(f"Processing tomogram {run_name}")
        volume = cp_interface.get_run_tomogram(run_name, voxel_spacing, tomo_type)
        coords_pixels = coords[run_name]/voxel_spacing
        montage.generate_slabs(volume, coords_pixels, run_name)
    montage.make_galleries(gallery_shape, voxel_spacing, out_dir, one_per_vol)

def generate_from_starfile(vol_path: str,
                           in_star: str,
                           out_dir: str,
                           extract_shape: tuple,
                           coords_scale: float = 1,
                           voxel_spacing: float = None,
                           tomo_type: str = None,
                           gallery_shape: tuple = (16,15),
                           one_per_vol: bool = False):
    """
    Generate galleries based on coordinates in a starfile,
    with the volumes loaded either indirectly through copick
    or directly from a directory containing mrc files. The
    coordinates scale factor should put the coordinates into A.

    Parameters
    ----------
    vol_path: copick configuration file or directory of mrc files
    in_star: starfile of particle coordinates
    out_dir: directory to write galleries and bookkeeping file to
    extract_shape: subvolume extraction shape in Angstrom
    coords_scale: multiplicative factor to apply to coordinates
    voxel_spacing: voxel spacing in Angstrom
    tomo_type: type of tomogram, e.g. 'denoised'
    gallery_shape: number of particles along gallery (row,col)
    one_per_vol: generate one gallery per tomogram
    """
    coords = read_starfile(in_star, coords_scale=coords_scale)
    extract_shape = tuple((np.array(extract_shape)/voxel_spacing).astype(int))
    
    load_method = ''
    if os.path.isfile(vol_path):
        cp_interface = CoPickWrangler(vol_path)
        load_method = "copick"
        assert voxel_spacing is not None and tomo_type is not None
    else:
        load_method = "mrc"

    montage = Minislab(extract_shape)
    for run_name in coords.keys():
        print(f"Processing tomogram {run_name}")
        if load_method == 'copick':
            volume = cp_interface.get_run_tomogram(run_name, voxel_spacing, tomo_type)
        if load_method == 'mrc':
            raise NotImplementedError
        coords_pixels = coords[run_name]/voxel_spacing
        montage.generate_slabs(volume, coords_pixels, run_name)
    montage.make_galleries(gallery_shape, voxel_spacing, out_dir, one_per_vol)

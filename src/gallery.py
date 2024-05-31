import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np
import starfile
import re
import os
from dataio import *

class Gallery:
    
    """
    Generate "galleries" or mosaics of tiled per-particle subvolume projections. 
    """    
    
    def __init__(self, extract_shape, coords_scale_factor=1.0):
        """
        Parameters
        ----------
        extract_shape: tuple, subvolume crop shape in Angstrom along (X,Y,Z)
        coords_scale_factor: float, factor to convert picked to extract tomogram pixel size
        """
        self.minislabs = {}
        self.num_particles = 0
        self.tomo_names = []
        self.pick_indices = []
        self.shape = extract_shape
        self.coords_scale_factor = coords_scale_factor
        
    def generate_slabs(self, vol, coords, vol_name):
        """
        Generate a per-particle slab, i.e. a 2d projection along z
        that is centered around each particle. 
        
        Parameters
        ----------
        vol: numpy or zarr array, tomogram
        coords: np.ndarray (n_coords, 3), xyz coordinates in pixels
        vol_name: str, name of volume
        """
        self.filler_mean, self.filler_std = np.mean(vol), np.std(vol)
        coords *= self.coords_scale_factor
        
        for i,c in enumerate(coords):
            # crop particle-centered subvolume and project along z
            c = c.astype(int)
            xstart, xend = c[2]-int(self.shape[2]/2), c[2]+int(self.shape[2]/2) if self.shape[2]%2==0 else c[2]+int(self.shape[2]/2) + 1
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
            std_scale_factor = 10
            if projection.shape[0] != self.shape[1]:
                edge = projection.shape[0]
                filler = gaussian_filter(np.random.normal(loc=self.filler_mean, scale=std_scale_factor*self.filler_std, 
                                                          size=(self.shape[1]-edge, projection.shape[1])), sigma=1.1)
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
                filler = gaussian_filter(np.random.normal(loc=self.filler_mean, scale=std_scale_factor*self.filler_std,
                                                          size=(projection.shape[0], self.shape[0]-edge)), sigma=1.1)
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
        
    def generate_galleries(self, gshape, apix, outdir, filename=None):
        """
        Generate galleries by tiling the minislabs and save the resulting projected
        particle mosaics in mrc format and a corresponding bookkeeping file as csv.
        
        Parameters
        ----------
        gshape: tuple, number of particles along (rows, cols)
        apix: float, pixel size in Angstrom   
        outdir: str, output directory for gallery mrcs and bookkeeping csv
        filename: str, specific prefix to include in output files
        """
        if len(self.minislabs)==0:
            raise Exception("No slabs have been generated.")
        os.makedirs(outdir, exist_ok=True)
        
        # generate galleries in mrc format
        gallery_idx = np.zeros(len(self.minislabs)).astype(int)
        row_idx = np.zeros(len(self.minislabs)).astype(int)
        col_idx = np.zeros(len(self.minislabs)).astype(int)
        
        counter = 0
        pshape = self.minislabs[0].shape
        std_scale_factor = 100
        n_mgraphs = len(self.minislabs) // np.prod(gshape) + 1
        for nm in range(n_mgraphs):
            gallery = np.zeros((gshape[0]*pshape[0], gshape[1]*pshape[1])).astype(np.float32)
            for i in range(gshape[0]):
                for j in range(gshape[1]):
                    if counter > len(self.minislabs)-1:
                        filler = gaussian_filter(np.random.normal(loc=self.filler_mean, scale=std_scale_factor*self.filler_std, 
                                                                  size=(pshape[0], pshape[1])), sigma=1.1)
                        gallery[i*pshape[0]:i*pshape[0]+pshape[0], j*pshape[1]:j*pshape[1]+pshape[1]] = filler
                    else:
                        gallery[i*pshape[0]:i*pshape[0]+pshape[0], j*pshape[1]:j*pshape[1]+pshape[1]] = self.minislabs[counter]
                        gallery_idx[counter], row_idx[counter], col_idx[counter] = nm, i, j
                        counter += 1
            if filename is None: filename = "gallery"
            save_mrc(gallery, os.path.join(outdir, f"{filename}_{nm:03d}.mrc"), apix=apix)

        # save bookkeeping file in csv format
        df = pd.DataFrame({'tomogram': self.tomo_names,
                           'particle': self.pick_indices,
                           'gallery': gallery_idx,
                           'row': row_idx,
                           'col': col_idx})
        if filename is None: filename = "bookkeeping"
        df.to_csv(os.path.join(outdir, f"{filename}.csv"), index=False)

def generate_from_copick(config: str,
                         out_dir: str,
                         particle_name: str,
                         voxel_spacing: float,
                         tomo_type: str,
                         extract_shape: tuple,
                         gallery_shape: tuple = (16,15)):
    """
    Generate galleries from a copick project.

    Parameters
    ----------
    config: copick configuration file 
    out_dir: directory to write galleries and bookkeeping file to
    particle_name: particle name
    voxel_spacing: voxel spacing in Angstrom
    tomo_type: type of tomogram, e.g. 'denoised'
    extract_shape: subvolume extraction shape in Angstrom
    gallery_shape: number of particles along gallery (row,col)
    """
    cp_interface = CoPickWrangler(config)
    coords = cp_interface.get_all_coords(particle_name)
    extract_shape = tuple((np.array(extract_shape)/voxel_spacing).astype(int))
    
    montage = Gallery(extract_shape)
    for run_name in coords.keys():
        volume = cp_interface.get_run_tomogram(run_name, voxel_spacing, tomo_type)
        coords_pixels = coords[run_name]/voxel_spacing
        montage.generate_slabs(volume, coords_pixels, run_name)
    montage.generate_galleries(gallery_shape, voxel_spacing, out_dir)

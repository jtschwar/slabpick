import numpy as np
import scipy.spatial

def cluster_coordinates(coords: np.ndarray, 
                        threshold: float) -> list:
    """
    Cluster coordinates based on a distance threshold.

    Parameters
    ----------
    coords: particle coordinates for one volume in Angstrom
    threshold: distance threshold in Angstrom

    Returns
    -------
    clusters: list of tuples containing replicate entries
    """
    distances = scipy.spatial.distance.cdist(coords, coords)
    indices = np.where((np.triu(distances)>0) & (np.triu(distances)<threshold))

    clusters = []
    for i,c in enumerate(coords):
        if i in indices[0] and i not in list(sum(clusters, ())):
            clusters.append(tuple(np.append([i], indices[1][indices[0]==i])))
        else:
            if i not in list(sum(clusters, ())):
                clusters.append((i,))
                
    return clusters

def merge_replicates(coords: np.ndarray,
                     threshold: float) -> tuple[np.ndarray, list]:
    """
    Generate a unique list of particle coordinates for one
    volume by clustering using a distance threshold.
        
    Parameters
    ----------
    coords: particle coordinates for one volume in Angstrom
    threshold: distance threshold in Angstrom
        
    Returns
    -------
    coords_unique: unique set of particle coordinates
    n_replicates: number of replicate entries per unique particle
    """
    cluster_ids = cluster_coordinates(coords, threshold)
    
    coords_unique = np.zeros((len(cluster_ids), 3))
    for i,c in enumerate(cluster_ids):
        coords_unique[i] = np.mean(coords[np.array(c)], axis=0)
    
    n_replicates = [len(entry) for entry in cluster_ids]
    return coords_unique, n_replicates

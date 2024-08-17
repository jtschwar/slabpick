import numpy as np
import scipy.signal
import scipy.spatial

from slabpick.minislab import get_subvolume


def map_coordinates(
    coords1: np.ndarray, coords2: np.ndarray, threshold: float,
) -> np.ndarray:
    """
    Map coordinates between two sets based on a distance threshold.

    Parameters
    ----------
    coords1: first set of coordinates in Angstrom
    coords2: second set of coordinates in Angstrom
    threshold: distance threshold in Angstrom

    Returns
    -------
    clusters: array of paired entries
    """
    distances = scipy.spatial.distance.cdist(coords1, coords2)
    clusters = np.where(distances < threshold)
    return np.array(clusters).T


def consolidate_coordinates(
    coords1: np.ndarray,
    coords2: np.ndarray,
    threshold: float,
    weights: list = None,
) -> np.ndarray:
    """
    Consolidate two sets of coordinates, merging duplicates
    between the sets and retaining the unique entries.

    Parameters
    ----------
    coords1: first set of coordinates in Angstrom
    coords2: second set of coordinates in Angstrom
    threshold: distance threshold in Angstrom
    weights: weights for merging coords1 and coords2 duplicates

    Returns
    -------
    coords_merge: consolidated set of coordinates in Angstrom
    clusters: array of paired entries
    """
    if weights is None:
        weights = [0.5, 0.5]
    clusters = map_coordinates(coords1, coords2, threshold)
    c1_unique_indices = np.setdiff1d(np.arange(coords1.shape[0]), clusters[:, 0])
    c2_unique_indices = np.setdiff1d(np.arange(coords2.shape[0]), clusters[:, 1])
    coords_cluster = np.average(
        (coords1[clusters[:, 0]], coords2[clusters[:, 1]]), axis=0, weights=weights,
    )
    coords_merge = np.concatenate(
        (coords1[c1_unique_indices], coords2[c2_unique_indices], coords_cluster),
    )
    return coords_merge, clusters


def consolidate_coordinates_sets(
    d_coords1: dict,
    d_coords2: dict,
    threshold: float,
    weights: list = None,
    ensure_unique: bool = True,
) -> tuple[dict, dict]:
    """
    Consolidate coordinates for every run in the dataset,
    retaining the unique coordinates and merging pairs of
    replicates represented in the input sets.

    Parameters
    ----------
    d_coords1: runs mapped to first set of coords in Angstrom
    d_coords2: runs mapped to second set of coords in Angstrom
    threshold: distance threshold in Angstrom for merging
    weights: relative weights to apply when merging coords
    ensure_unique: merge any lingering replicates

    Returns
    -------
    d_coords_merge: dict mapping runs to merged coords in Angstrom
    d_clusters: dict mapping runs to indices of replicates between sets
    """
    if weights is None:
        weights = [0.5, 0.5]
    tomo_list = list(d_coords1.keys()) + list(d_coords2.keys())
    tomo_list = sorted(set(tomo_list), key=tomo_list.index)

    d_coords_merge, d_clusters = {}, {}
    ini_count, final_count = 0, 0
    for _i, tomo in enumerate(tomo_list):
        if (tomo in d_coords1) and (tomo in d_coords2):
            d_coords_merge[tomo], d_clusters[tomo] = consolidate_coordinates(
                d_coords1[tomo],
                d_coords2[tomo],
                threshold,
                weights,
            )
        elif tomo in d_coords1:
            d_coords_merge[tomo] = d_coords1[tomo]
        elif tomo in d_coords2:
            d_coords_merge[tomo] = d_coords2[tomo]

        if ensure_unique:
            ini_count += d_coords_merge[tomo].shape[0]
            d_coords_merge[tomo], nr = merge_replicates(d_coords_merge[tomo], threshold)
        final_count += d_coords_merge[tomo].shape[0]

    print(f"Final particle count: {final_count}")
    if ensure_unique:
        print(f"Additional duplicate removal merged  {ini_count-final_count} particles")

    return d_coords_merge, d_clusters


def cluster_coordinates(coords: np.ndarray, threshold: float) -> list:
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
    indices = np.where((np.triu(distances) > 0) & (np.triu(distances) < threshold))

    clusters = []
    for i, _c in enumerate(coords):
        if i in indices[0] and i not in list(sum(clusters, ())):
            clusters.append(tuple(np.append([i], indices[1][indices[0] == i])))
        else:
            if i not in list(sum(clusters, ())):
                clusters.append((i,))

    return clusters


def merge_replicates(coords: np.ndarray, threshold: float) -> tuple[np.ndarray, list]:
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
    for i, c in enumerate(cluster_ids):
        coords_unique[i] = np.mean(coords[np.array(c)], axis=0)

    n_replicates = [len(entry) for entry in cluster_ids]
    return coords_unique, n_replicates


def generate_window(dim: int) -> np.ndarray:
    """
    Generate a 2d cosine window to mask the non-central
    region of a square box.

    Parameters
    ----------
    dim: box length

    Returns
    -------
    2d cosine window function
    """
    kx = scipy.signal.windows.cosine(dim)
    return kx[:, np.newaxis] * kx[np.newaxis, :]


def refine_z(
    coords: np.ndarray,
    volume: np.ndarray,
    extract_shape: np.ndarray | tuple,
    window: np.ndarray,
) -> np.ndarray:
    """
    Refine the z-coordinate of the input coordinates based on intensity
    statitics, specifically by finding the plane along the z-axis of a
    subvolume crop with the minimum integrated intensity.

    Parameters
    ----------
    coords: (X,Y,Z) coordinates of particle centers in pixels
    volume: particle-containing tomogram
    extract_shape: target (X,Y,Z) dimensions of extraction subvolumes
    window: 2d window function to apply to each plane of the subvolume

    Returns
    -------
    rcoords: coordinates with refined z-heights in pixels
    """
    rcoords = np.zeros_like(coords)
    for i, c in enumerate(coords):
        subvolume = get_subvolume(coords[i], volume, extract_shape)
        filt_subvolume = subvolume * window[np.newaxis, :, :]
        z_profile = np.sum(filt_subvolume, axis=(1, 2))
        z_delta = subvolume.shape[2] / 2 - np.argmin(z_profile)
        rcoords[i] = np.array([c[0], c[1], int(c[2] - z_delta)])

    return rcoords

import numpy as np


def normalize_stack(particles: np.ndarray, radius: float = 0.9) -> np.ndarray:
    """
    Normalize a stack on a per-particle basis. The central
    region inside the supplied fractional radius is masked
    to compute the per-particle mean and standard deviation.

    Parameters
    ----------
    particles: stack of particles
    radius: boundary for computing standard deviation

    Returns
    -------
    particles: normalized particle stack
    """
    pshape = particles[0].shape
    y, x = np.indices(pshape)
    center = pshape[1] / 2 - 0.5, pshape[0] / 2 - 0.5
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = np.broadcast_to(r < np.min(center) * radius, particles.shape).astype(int)
    particles_masked = np.ma.masked_array(particles, mask=mask)

    mu_background = np.mean(particles_masked, axis=(1,2)).data
    sigma_background = np.std(particles_masked, axis=(1,2)).data
    particles -= mu_background[:,np.newaxis,np.newaxis]
    particles /= sigma_background[:,np.newaxis,np.newaxis]
    return particles


def invert_contrast(particles: np.ndarray) -> np.ndarray:
    """
    Invert contrast of a particle stack.

    Parameters
    ----------
    particles: stack of particles

    Returns
    -------
    particle stack with inverted contrast
    """
    return -1.0 * particles

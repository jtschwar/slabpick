import numpy as np
import scipy.ndimage
import slabpick.minislab as minislab


def test_render_even():
    ref = np.array([4, 4, 4])
    in_array = ref + np.random.randint(low=0, high=2, size=3)
    assert np.array_equal(minislab.render_even(in_array), ref)


def test_get_subvolume():
    # extraction subvolume contained within volume
    a = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    sa = minislab.get_subvolume(np.array([3, 3, 3]), a, (4, 4, 4))
    assert sa.shape == (4, 4, 4)
    assert np.array_equal(sa, a[1:5, 1:5, 1:5])

    # extraction subvolume extends beyond volume
    sa = minislab.get_subvolume(np.array([3, 3, 3]), a, (5, 4, 4))
    assert sa.shape == (4, 4, 5)
    assert np.array_equal(sa[:, :, :-1], a[1:5, 1:5, 1:5])


def test_tilt_subvolume():
    # trivial case of 0 degrees
    a = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    b = minislab.tilt_subvolume(a, 0, a.shape)
    assert np.array_equal(b, a)

    # trivial case of 90 degrees
    a = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    b = minislab.tilt_subvolume(a, 90, a.shape)
    assert np.array_equal(b, np.rot90(a, axes=(0, 2)))

    # nontrivial case of random angle
    # note that this only works for even-dimension outputs where
    # the intentionally-retained regions rotated out of the field
    # can be cropped to match reference
    a = np.random.rand(20, 20, 20)
    angle = np.random.random() * 45
    b = minislab.tilt_subvolume(a, angle, a.shape)
    while b.shape[0] % 2 != 0:
        angle = np.random.random() * 45
        b = minislab.tilt_subvolume(a, angle, a.shape)
    b_ref = tilt_subvolume_reference(a, angle)
    crop = int((b.shape[0] - b_ref.shape[0]) / 2)
    if crop != 0:
        b = b[crop : -1 * crop, :, :]
    assert np.allclose(b, b_ref, rtol=1e-6)


def tilt_subvolume_reference(
    array: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Reference implementation of subvolume tilting
    about the x-axis, courtesy:
    https://stackoverflow.com/questions/59738230/apply-
    rotation-defined-by-euler-angles-to-3d-image-in-python

    Parameters
    ----------
    array: volume to be tilted
    angle:
    """
    theta = np.deg2rad(-1 * angle)  # negative turns CW -> CCW
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -1 * np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
    )

    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    xyz = np.vstack(
        [
            coords[0].reshape(-1) - float(dim[0]) / 2 + 0.5,  # x coordinate, centered
            coords[1].reshape(-1) - float(dim[1]) / 2 + 0.5,  # y coordinate, centered
            coords[2].reshape(-1) - float(dim[2]) / 2 + 0.5,
        ],
    )  # z coordinate, centered
    transformed_xyz = np.dot(Rx, xyz)

    x = transformed_xyz[0, :] + float(dim[0]) / 2 - 0.5
    y = transformed_xyz[1, :] + float(dim[1]) / 2 - 0.5
    z = transformed_xyz[2, :] + float(dim[2]) / 2 - 0.5

    x = x.reshape((dim[1], dim[0], dim[2]))
    y = y.reshape((dim[1], dim[0], dim[2]))
    z = z.reshape(
        (dim[1], dim[0], dim[2]),
    )  # reason for strange ordering: see next line

    new_xyz = [y, x, z]
    return scipy.ndimage.map_coordinates(array, new_xyz, order=1)

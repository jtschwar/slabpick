import os
import shutil

import numpy as np
import pandas as pd
import pytest
from slabpick.dataio import load_mrc, save_mrc
from slabpick.slab import Slab, determine_bounds


def test_determine_bounds():
    """
    Test various inputs for determining bounds with a sliding window.
    """
    assert np.array_equal(determine_bounds(10, 10, 1, 40), np.array([0, 10, 20, 30]))
    assert np.array_equal(determine_bounds(20, 20, 1, 40), np.array([0, 20]))
    assert np.array_equal(determine_bounds(10 * 2, 10 * 2, 2, 40), np.array([0, 10, 20, 30]))
    indices = determine_bounds(10, 8, 1, 40)
    assert indices[0] == 40 - indices[-1] - 10
    assert len(set(np.diff(indices))) < 2  # check same interval between items


class TestSlab:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.test_dir = "slab_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        self.make_mock_volumes()
        self.make_slabs(3, 2)

    def make_mock_volumes(self):
        """
        Make mock volumes for slab generation.
        """
        xdim = np.random.randint(3, 8)
        ydim = np.random.randint(3, 8)
        zdim = np.random.randint(3, 5) * 2
        self.vol_shape = (zdim, ydim, xdim)
        self.vol_names = []
        for i in range(3):
            test_vol = np.ones(self.vol_shape) * np.arange(zdim)[:, np.newaxis, np.newaxis] + i
            save_mrc(test_vol, f"{self.test_dir}/volume_{i:03d}.mrc", apix=1)
            self.vol_names.append(f"{self.test_dir}/volume_{i:03d}.mrc")

    def make_slabs(self, zthick, zslide):
        """
        Make slabs from mock volumes.
        """
        slabber = Slab(zthick, zslide)
        slabber.generate_slabs(self.test_dir, self.test_dir, include_tags=["volume"])

    def test_check_slab_mapping(self):
        """
        Check correctness of output slabs and mapping file.
        """
        df = pd.read_csv(os.path.join(self.test_dir, "slab_map.csv"))
        assert len(np.unique(df.tomogram.values)) == len(self.vol_names)
        assert len(set(df["zdepth"].values)) < 2

        for i in range(df.shape[0]):
            in_vol = load_mrc(df.iloc[i]["tomogram"])
            i_slab = load_mrc(df.iloc[i]["slab"])
            zstart, zdepth = df.iloc[i]["zstart"], df.iloc[i]["zdepth"]
            r_slab = np.sum(in_vol[zstart : zstart + zdepth], axis=0)
            assert np.array_equal(i_slab, r_slab)

        shutil.rmtree(self.test_dir)

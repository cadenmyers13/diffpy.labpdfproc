import os

import pytest

from diffpy.utils.parsers import load_data


# Test that our readable and unreadable files are indeed readable and
# unreadable by load_data (which is our definition of readable and unreadable)
def test_load_data_with_input_files(user_filesystem):
    os.chdir(user_filesystem)
    xarray_chi, yarray_chi = load_data("good_data.chi", unpack=True)
    xarray_xy, yarray_xy = load_data("good_data.xy", unpack=True)
    xarray_txt, yarray_txt = load_data("good_data.txt", unpack=True)
    with pytest.raises(ValueError):
        xarray_txt, yarray_txt = load_data("unreadable_file.txt", unpack=True)
    with pytest.raises(ValueError):
        xarray_pkl, yarray_pkl = load_data("binary.pkl", unpack=True)

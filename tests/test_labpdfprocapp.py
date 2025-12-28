from argparse import Namespace

import pytest

from diffpy.labpdfproc.labpdfprocapp import (
    resolve_wavelength,
    run_mud,
    run_sample,
    run_zscan,
)
from diffpy.labpdfproc.tools import WAVELENGTHS
from diffpy.utils.parsers.loaddata import loadData

sources = sorted(WAVELENGTHS.keys())


@pytest.mark.parametrize(
    "wavelength,expected",
    [
        # UC1
        # input: wavelength in angstroms, expected: same value
        (1.54, 1.54),
        # UC2
        # input: radiation source, expected: corresponding wavelength in Å
        ("MoKa1", 0.70930),
    ],
)
def test_resolve_wavelength(wavelength, expected):
    actual = resolve_wavelength(wavelength)
    assert actual == expected


@pytest.mark.parametrize(
    "bad_wavelength, expected",
    [
        (
            "invalid_source",
            "Unknown X-ray source 'invalid_source'. "
            f"Allowed sources are: {', '.join(sources)}.",
        ),
    ],
)
def test_resolve_wavelength_bad(bad_wavelength, expected):
    with pytest.raises(ValueError) as excinfo:
        resolve_wavelength(bad_wavelength)
    actual = str(excinfo.value)
    assert actual == expected


def test_run_mud(real_data_file, temp_output_dir):
    """Test that metadata is correctly stored after running run_mud."""
    args = Namespace(
        xray_data=str(real_data_file),
        wavelength="CuKa1",
        mud_value=2.5,
        xtype="tth",
        method="polynomial_interpolation",
        target_dir=str(temp_output_dir),
        output_correction=False,
        user_metadata=["facility=NSLS-II", "beamline=28-ID-2"],
        username="Test User",
        email="test@example.com",
        orcid="0000-0001-2345-6789",
        command="mud",
    )
    run_mud(args)
    output_file = temp_output_dir / "CeO2_635um_accum_0_corrected.chi"
    actual_raw = loadData(output_file, headers=True)
    # drop keys for test simplicity
    keys_to_drop = {"creation_time", "package_info"}
    actual = {k: v for k, v in actual_raw.items() if k not in keys_to_drop}
    expected = {
        "name": "Absorption corrected input_data: CeO2_635um_accum_0",
        "wavelength": 1.54056,
        "scat_quantity": "x-ray",
        "mud": 2.5,
        "output_directory": str(temp_output_dir),
        "xtype": "tth",
        "method": "polynomial_interpolation",
        "username": "Test User",
        "email": "test@example.com",
        "orcid": "0000-0001-2345-6789",
        # package_info removed
        # creation_time removed
        "input_file": str(real_data_file),
        "command": "mud",
        # metadata from user
        "facility": "NSLS-II",
        "beamline": "28-ID-2",
    }
    assert actual == expected


def test_run_sample():
    run_sample()
    assert False


def test_run_zscan():
    run_zscan()
    assert False


# [DiffractionObject]
# name = Absorption corrected input_data: CeO2_635um_accum_0
# wavelength = None
# scat_quantity = x-ray
# mud = 3.48870066530601
# output_directory = /Users/cadenmyers/billingelab/dev/diffpy.labpdfproc
# xtype = tth
# method = polynomial_interpolation
# username = Caden
# email = cjm2304@columbia.edu
# orcid =
# package_info = {'diffpy.labpdfproc': '0.0.1', 'diffpy.utils': '3.6.1'}
# input_directory = /Users/cadenmyers/billingelab/dev/diffpy.labpdfproc/doc
# /source/examples/example-data/CeO2_635um_accum_0.xy
# creation_time = 2025-12-27 11:03:52.175988

# def test_run_zscan(sample_data_file, zscan_file, temp_output_dir):
#     args = Namespace(
#         xray_data=str(sample_data_file),
#         z_scan_data=str(zscan_file),
#         wavelength="CuKa1",
#         mud_value=2.5,
#         xtype="tth",
#         method="polynomial_interpolation",
#         target_dir=str(temp_output_dir),
#         output_correction=False,
#         user_metadata=["myinfo=stuff"],
#         username="Test User",
#         email="test@example.com",
#         orcid="0000-0001-2345-6789",
#         command="zscan",
#     )

#     # Run the command
#     run_zscan(args)

#     # Load the corrected output file
#     output_file = temp_output_dir / "test_data_corrected.chi"
#     corrected = DiffractionObject.from_file(str(output_file))

#     # Expected metadata (excluding computed mud value)
#     expected = {
#         "name": "Absorption corrected input_data: test_data",
#         "wavelength": 1.54056,
#         "scat_quantity": "x-ray",
#         "mud": 2.5,
#         "output_directory": str(temp_output_dir),
#         "xtype": "tth",
#         "method": "polynomial_interpolation",
#         "username": "Test User",
#         "email": "test@example.com",
#         "orcid": "0000-0001-2345-6789",
#         # package_info removed
#         # creation_time removed
#         "input_file": str(sample_data_file),
#         "command": "zscan",
#         # metadata from user
#         "myinfo": "stuff",
#     }

#     # Extract only the keys we're testing
#     actual = {k: corrected.metadata[k] for k in expected.keys()}

#     assert actual == expected

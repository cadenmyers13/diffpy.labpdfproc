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
        # UC1: user give a numeric wavelength
        # input: wavelength in angstroms, expected: same value
        (1.54, 1.54),
        # UC2: user give a known radiation source
        # input: radiation source, expected: corresponding wavelength in Å
        ("MoKa1", 0.70930),
    ],
)
def test_resolve_wavelength(wavelength, expected):
    actual = resolve_wavelength(wavelength)
    assert actual == expected


@pytest.mark.parametrize(
    "bad_wavelength, expected",
    [  # bad UC1: user give a non-numeric, non-known source
        # input: invalid string, expected: ValueError error message
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


# UC: user corrects data with mud value
# input: input arguments for run_mud function
# expected: correct metadata in output file
def test_run_mud(real_data_file, temp_output_dir):
    """Test that metadata is correctly stored after running run_mud."""
    input_args = Namespace(
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
    run_mud(input_args)
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
        # metadata from user
        "facility": "NSLS-II",
        "beamline": "28-ID-2",
        # mud-specific metadata
        "command": "mud",
    }
    assert actual == expected


# UC: user corrects data with sample composition and density
# input: input arguments for run_sample function
# expected: correct metadata in output file
def test_run_sample(real_data_file, temp_output_dir):
    input_args = Namespace(
        xray_data=str(real_data_file),
        wavelength="Mo",
        xtype="tth",
        method="polynomial_interpolation",
        target_dir=str(temp_output_dir),
        output_correction=False,
        user_metadata=["facility=NSLS-II", "beamline=28-ID-2"],
        username="Test User",
        email="test@example.com",
        orcid="0000-0001-2345-6789",
        composition="CeO2",
        density=1,  # arbitrary density
        command="sample",
    )
    run_sample(input_args)
    output_file = temp_output_dir / "CeO2_635um_accum_0_corrected.chi"
    actual_raw = loadData(output_file, headers=True)
    # drop keys for test simplicity
    keys_to_drop = {"creation_time", "package_info"}
    actual = {k: v for k, v in actual_raw.items() if k not in keys_to_drop}

    expected = {
        "name": "Absorption corrected input_data: CeO2_635um_accum_0",
        "wavelength": 0.71073,
        "scat_quantity": "x-ray",
        "mud": 3.904202343614975,
        "output_directory": str(temp_output_dir),
        "xtype": "tth",
        "method": "polynomial_interpolation",
        "username": "Test User",
        "email": "test@example.com",
        "orcid": "0000-0001-2345-6789",
        # package_info removed
        # creation_time removed
        "input_file": str(real_data_file),
        # metadata from user
        "facility": "NSLS-II",
        "beamline": "28-ID-2",
        # sample-specific metadata
        "command": "sample",
        "composition": "CeO2",
        "density": 1.0,
    }
    assert actual == expected


# UC: user corrects data with z-scan file
# input: input arguments for run_zscan function
# expected: correct metadata in output file
def test_run_zscan(real_data_file, real_zscan_file, temp_output_dir):
    input_args = Namespace(
        xray_data=str(real_data_file),
        zscan_file=str(real_zscan_file),
        wavelength="Mo",
        xtype="tth",
        method="polynomial_interpolation",
        target_dir=str(temp_output_dir),
        output_correction=False,
        user_metadata=["facility=NSLS-II", "beamline=28-ID-2"],
        username="Test User",
        email="test@example.com",
        orcid="0000-0001-2345-6789",
        command="zscan",
    )
    run_zscan(input_args)
    output_file = temp_output_dir / "CeO2_635um_accum_0_corrected.chi"
    actual_raw = loadData(output_file, headers=True)
    # drop keys for test simplicity
    keys_to_drop = {"creation_time", "package_info", "mud"}
    actual = {k: v for k, v in actual_raw.items() if k not in keys_to_drop}

    expected = {
        "name": "Absorption corrected input_data: CeO2_635um_accum_0",
        "wavelength": 0.71073,
        "scat_quantity": "x-ray",
        # mud removed due to slight variability in calculation
        "output_directory": str(temp_output_dir),
        "xtype": "tth",
        "method": "polynomial_interpolation",
        "username": "Test User",
        "email": "test@example.com",
        "orcid": "0000-0001-2345-6789",
        # package_info removed
        # creation_time removed
        "input_file": str(real_data_file),
        # metadata from user
        "facility": "NSLS-II",
        "beamline": "28-ID-2",
        # zscan-specific metadata
        "command": "zscan",
        "z_scan_file": str(real_zscan_file),
    }
    assert actual == expected

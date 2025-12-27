import pytest

from diffpy.labpdfproc.labpdfprocapp import (
    resolve_wavelength,
    run_mud,
    run_sample,
    run_zscan,
)
from diffpy.labpdfproc.tools import WAVELENGTHS

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


def test_run_mud():
    run_mud()
    assert False


def test_run_zscan():
    run_zscan()
    assert False


def test_run_sample():
    run_sample()
    assert False

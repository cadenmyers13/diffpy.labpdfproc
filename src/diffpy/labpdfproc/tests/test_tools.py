import argparse
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import pytest

from diffpy.labpdfproc.tools import known_sources, load_user_metadata, set_output_directory, set_wavelength

params1 = [
    ([None], ["."]),
    (["."], ["."]),
    (["new_dir"], ["new_dir"]),
    (["existing_dir"], ["existing_dir"]),
]


@pytest.mark.parametrize("inputs, expected", params1)
def test_set_output_directory(inputs, expected, tmp_path):
    directory = Path(tmp_path)
    os.chdir(directory)

    existing_dir = Path(tmp_path).resolve() / "existing_dir"
    existing_dir.mkdir(parents=True, exist_ok=True)

    expected_output_directory = Path(tmp_path).resolve() / expected[0]
    actual_args = argparse.Namespace(output_directory=inputs[0])
    actual_args.output_directory = set_output_directory(actual_args)
    assert actual_args.output_directory == expected_output_directory
    assert Path(actual_args.output_directory).exists()
    assert Path(actual_args.output_directory).is_dir()


def test_set_output_directory_bad(tmp_path):
    directory = Path(tmp_path)
    os.chdir(directory)

    existing_file = Path(tmp_path).resolve() / "existing_file.py"
    existing_file.touch()

    actual_args = argparse.Namespace(output_directory="existing_file.py")
    with pytest.raises(FileExistsError):
        actual_args.output_directory = set_output_directory(actual_args)
        assert Path(actual_args.output_directory).exists()
        assert not Path(actual_args.output_directory).is_dir()


params2 = [
    ([None, None], [0.71]),
    ([None, "Ag"], [0.59]),
    ([0.25, "Ag"], [0.25]),
    ([0.25, None], [0.25]),
]


@pytest.mark.parametrize("inputs, expected", params2)
def test_set_wavelength(inputs, expected):
    expected_wavelength = expected[0]
    actual_args = argparse.Namespace(wavelength=inputs[0], anode_type=inputs[1])
    actual_wavelength = set_wavelength(actual_args)
    assert actual_wavelength == expected_wavelength


params3 = [
    (
        [None, "invalid"],
        [f"Anode type not recognized. Please rerun specifying an anode_type from {*known_sources, }."],
    ),
    ([0, None], ["No valid wavelength. Please rerun specifying a known anode_type or a positive wavelength."]),
    ([-1, "Mo"], ["No valid wavelength. Please rerun specifying a known anode_type or a positive wavelength."]),
]


@pytest.mark.parametrize("inputs, msg", params3)
def test_set_wavelength_bad(inputs, msg):
    actual_args = argparse.Namespace(wavelength=inputs[0], anode_type=inputs[1])
    with pytest.raises(ValueError, match=re.escape(msg[0])):
        actual_args.wavelength = set_wavelength(actual_args)


params5 = [
    ([[]], []),
    ([["toast=for breakfast"]], [["toast", "for breakfast"]]),
    ([["mylist=[1,2,3.0]"]], [["mylist", "[1,2,3.0]"]]),
    ([["weather=rainy", "day=tuesday"]], [["weather", "rainy"], ["day", "tuesday"]]),
]


@pytest.mark.parametrize("inputs, expected", params5)
def test_load_user_metadata(inputs, expected):
    actual_parser = ArgumentParser()
    actual_parser.add_argument("-u", "--user-metadata", action="append", metavar="KEY=VALUE", nargs="+")
    actual_args = actual_parser.parse_args([])
    expected_parser = ArgumentParser()
    expected_args = expected_parser.parse_args([])

    setattr(actual_args, "user_metadata", inputs[0])
    actual_args = load_user_metadata(actual_args)
    for expected_pair in expected:
        setattr(expected_args, expected_pair[0], expected_pair[1])
    assert actual_args == expected_args

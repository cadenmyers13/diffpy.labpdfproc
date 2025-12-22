#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from pathlib import Path

from gooey import Gooey, GooeyParser

from diffpy.labpdfproc.functions import apply_corr, compute_cve
from diffpy.labpdfproc.tools import load_metadata, preprocessing_args
from diffpy.utils.diffraction_objects import DiffractionObject
from diffpy.utils.parsers.loaddata import loadData

# -----------------------------------------------------------------------------
# GUI / CLI mode detection
# -----------------------------------------------------------------------------


def _use_gui():
    return len(sys.argv) == 1 or "--gui" in sys.argv


# -----------------------------------------------------------------------------
# Parser construction
# -----------------------------------------------------------------------------


def _add_common_args(p, use_gui=False):
    p.add_argument(
        "-t",
        "--target-dir",
        default=".",
        help="Directory to save corrected data",
        **({"widget": "DirChooser"} if use_gui else {}),
    )
    p.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )


def _register_mud(subp, use_gui=False):
    p = subp.add_parser(
        "mud",
        help="Apply absorption correction using a known μ·d value",
    )
    p.add_argument(
        "xray_data",
        help="Input X-ray diffraction data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    p.add_argument(
        "mud",
        type=float,
        help="Known μ·d value",
        **({"widget": "DecimalField"} if use_gui else {}),
    )
    _add_common_args(p, use_gui)


def _register_zscan(subp, use_gui=False):
    p = subp.add_parser(
        "zscan",
        help="Apply absorption correction using a z-scan measurement",
    )
    p.add_argument(
        "xray_data",
        help="Input X-ray diffraction data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    p.add_argument(
        "zscan_data",
        help="Z-scan data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    _add_common_args(p, use_gui)


def _register_sample(subp, use_gui=False):
    p = subp.add_parser(
        "sample",
        help="Apply absorption correction from sample properties",
        description=(
            "Compute absorption correction using tabulated values based on "
            "sample composition, X-ray energy or source, and mass density.\n\n"
            "Energy units: keV\n"
            "Density units: g/cm^3"
        ),
    )

    p.add_argument(
        "xray_data",
        help="Input X-ray diffraction data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )

    p.add_argument(
        "--composition",
        required=True,
        help="Chemical formula (e.g. Fe2O3)",
    )

    p.add_argument(
        "--energy",
        required=True,
        help=(
            "Incident X-ray energy in keV (e.g. 10) "
            "or a known source (e.g. CuKa, MoKa)"
        ),
    )

    p.add_argument(
        "--density",
        required=True,
        type=float,
        help="Sample mass density in g/cm^3",
        **({"widget": "DecimalField"} if use_gui else {}),
    )

    _add_common_args(p, use_gui)


def create_parser(use_gui=False):
    Parser = GooeyParser if use_gui else ArgumentParser
    p = Parser(
        prog="labpdfproc",
        description=(
            "Apply absorption corrections to laboratory X-ray diffraction "
            "data prior to PDF analysis.\n\n"
            "This tool computes and applies μ·d-based absorption corrections "
            "using experimental measurements or tabulated values, and writes "
            "corrected diffraction data suitable for use with diffpy.pdfgetx.\n\n"

            "To use the GUI, run without any arguments or with the --gui flag."
        ),
    )

    subp = p.add_subparsers(
        dest="command",
        required=True,
        title="Absorption correction method",
    )

    _register_mud(subp, use_gui)
    _register_zscan(subp, use_gui)
    _register_sample(subp, use_gui)

    return p


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def _load_pattern(filepath):
    x, y = loadData(filepath, unpack=True)
    return DiffractionObject(
        xarray=x,
        yarray=y,
        xtype="tth",
        scat_quantity="x-ray",
        name=Path(filepath).stem,
    )


def _save_corrected(pattern, input_path, target_dir, force):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    outfile = target_dir / f"{input_path.stem}_corrected.chi"

    if outfile.exists() and not force:
        print(
            f"WARNING: {outfile} already exists. " "Use --force to overwrite."
        )
        return

    pattern.dump(outfile)
    print(f"Saved corrected data to {outfile}")


# -----------------------------------------------------------------------------
# Workflows
# -----------------------------------------------------------------------------


def run_mud(args):
    args = preprocessing_args(args)
    path = Path(args.xray_data)

    pattern = _load_pattern(path)
    corr = compute_cve(pattern, args.mud)
    corrected = apply_corr(pattern, corr)

    _save_corrected(corrected, path, args.target_dir, args.force)


def run_zscan(args):
    args = preprocessing_args(args)
    path = Path(args.xray_data)

    pattern = _load_pattern(path)

    # z-scan-based μ·d estimation handled internally by compute_cve
    corr = compute_cve(pattern, args.zscan_data)
    corrected = apply_corr(pattern, corr)

    _save_corrected(corrected, path, args.target_dir, args.force)


def run_sample(args):
    args = preprocessing_args(args)
    path = Path(args.xray_data)

    pattern = _load_pattern(path)

    # Energy may be numeric (keV) or a named source
    try:
        energy = float(args.energy)
    except ValueError:
        energy = args.energy  # assume named source (e.g. CuKa)

    corr = compute_cve(
        pattern,
        (
            args.composition,
            energy,
            args.density,
        ),
    )

    corrected = apply_corr(pattern, corr)

    _save_corrected(corrected, path, args.target_dir, args.force)


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------


def dispatch(args):
    if args.command == "mud":
        return run_mud(args)
    if args.command == "zscan":
        return run_zscan(args)
    if args.command == "sample":
        return run_sample(args)

    raise ValueError(f"Unknown command: {args.command}")


# -----------------------------------------------------------------------------
# GUI entry
# -----------------------------------------------------------------------------


@Gooey(
    program_name="labpdfproc",
    show_sidebar=True,
    sidebar_title="Absorption correction",
    required_cols=1,
    optional_cols=1,
)
def run_gui():
    args = create_parser(use_gui=True).parse_args()
    dispatch(args)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    if _use_gui():
        return run_gui()

    args = create_parser(use_gui=False).parse_args()
    dispatch(args)


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

from gooey import Gooey, GooeyParser

from diffpy.labpdfproc.functions import CVE_METHODS, apply_corr, compute_cve
from diffpy.labpdfproc.tools import WAVELENGTHS, known_sources
from diffpy.utils.diffraction_objects import XQUANTITIES, DiffractionObject
from diffpy.utils.parsers.loaddata import loadData
from diffpy.utils.tools import compute_mu_using_xraydb, compute_mud

# -----------------------
# Helper functions
# -----------------------


def _add_common_args(parser, use_gui=False):
    parser.add_argument(
        "-t",
        "--target-dir",
        help=(
            "Directory to save corrected files (created if needed). "
            "Defaults to current directory."
        ),
        default=None,
        **({"widget": "DirChooser"} if use_gui else {}),
    )
    parser.add_argument(
        "-f", "--force", help="Overwrite existing files", action="store_true"
    )
    parser.add_argument(
        "--xtype",
        help="X-axis type for output (default: tth)",
        default="tth",
        choices=XQUANTITIES,
    )
    parser.add_argument(
        "--method",
        help="Method for CVE calculation (default: polynomial_interpolation)",
        default="polynomial_interpolation",
        choices=CVE_METHODS,
    )
    _add_credit_args(parser, use_gui)
    return parser


def _add_credit_args(parser, use_gui=False):
    parser.add_argument(
        "--username",
        help="Your name (optional, for dataset credit)",
        default=None,
        **({"widget": "TextField"} if use_gui else {}),
    )
    parser.add_argument(
        "--email",
        help="Your email (optional, for dataset credit)",
        default=None,
        **({"widget": "TextField"} if use_gui else {}),
    )
    parser.add_argument(
        "--orcid",
        help="Your ORCID ID (optional, for dataset credit)",
        default=None,
        **({"widget": "TextField"} if use_gui else {}),
    )


def _save_corrected(corrected, input_path, target_dir, force):
    target_dir = Path(target_dir) if target_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    outfile = target_dir / (input_path.stem + "_corrected.chi")

    if outfile.exists() and not force:
        print(f"WARNING: {outfile} exists. Use --force to overwrite.")
        return

    corrected.dump(str(outfile), xtype=corrected.xtype)
    print(f"Saved corrected data to {outfile}")


def _attach_credit_metadata(pattern, args):
    """Add optional user credit metadata if provided."""
    if pattern.metadata is None:
        pattern.metadata = {}
    for key in ("username", "email", "orcid"):
        value = getattr(args, key, None)
        if value:
            pattern.metadata[key] = value


def _load_pattern(path, wavelength=None):
    x, y = loadData(path, unpack=True)
    return DiffractionObject(
        xarray=x,
        yarray=y,
        xtype="tth",
        wavelength=wavelength,
        name=path.stem,
        metadata=None,
    )


def _get_wavelength(energy_or_source):
    """Convert energy (keV) or source name to wavelength (Angstrom)."""
    try:
        # Try to parse as numeric energy in keV
        energy_kev = float(energy_or_source)
        # Convert keV to wavelength in Angstrom: λ = 12.398 / E(keV)
        return 12.398 / energy_kev
    except ValueError:
        # It's a source name, look it up
        matched_source = next(
            (
                key
                for key in WAVELENGTHS
                if key.lower() == energy_or_source.lower()
            ),
            None,
        )
        if matched_source is None:
            raise ValueError(
                f"Source '{energy_or_source}' not recognized. "
                f"Allowed sources are {known_sources}."
            )
        return WAVELENGTHS[matched_source]


# -----------------------
# Subcommand functions
# -----------------------


def run_mud(args):
    path = Path(args.xray_data)
    pattern = _load_pattern(path)

    corr = compute_cve(pattern, args.mud, method=args.method, xtype=args.xtype)
    corrected = apply_corr(pattern, corr)
    _attach_credit_metadata(corrected, args)
    _save_corrected(corrected, path, args.target_dir, args.force)


def run_zscan(args):
    pattern_path = Path(args.xray_data)
    zscan_path = Path(args.zscan_file)

    # Compute mud from z-scan file
    mud = compute_mud(zscan_path)

    pattern = _load_pattern(pattern_path)
    corr = compute_cve(pattern, mud, method=args.method, xtype=args.xtype)
    corrected = apply_corr(pattern, corr)
    _attach_credit_metadata(corrected, args)
    _save_corrected(corrected, pattern_path, args.target_dir, args.force)


def run_sample(args):
    path = Path(args.xray_data)

    # Get wavelength from energy or source
    wavelength = _get_wavelength(args.energy)

    # Convert wavelength to energy in keV for xraydb
    energy_kev = 12.398 / wavelength

    # Compute mu*d from sample parameters
    mud = compute_mu_using_xraydb(
        args.composition, energy_kev, sample_mass_density=args.density
    )

    pattern = _load_pattern(path, wavelength=wavelength)
    corr = compute_cve(pattern, mud, method=args.method, xtype=args.xtype)
    corrected = apply_corr(pattern, corr)
    _attach_credit_metadata(corrected, args)
    _save_corrected(corrected, path, args.target_dir, args.force)


# -----------------------
# Parser construction
# -----------------------


def create_parser(use_gui=False):
    Parser = GooeyParser if use_gui else argparse.ArgumentParser
    parser = Parser(
        prog="labpdfproc",
        description=(
            "Apply absorption corrections to laboratory X-ray diffraction "
            "data prior to PDF analysis. "
            "Supports manual mu*d, z-scan, or sample-based corrections."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subp = parser.add_subparsers(
        dest="command", required=True, title="Correction method"
    )

    # -----------------------
    # MUD parser
    # -----------------------
    mud_parser = subp.add_parser("mud", help="Correct using known mu*d value")
    mud_parser.add_argument(
        "xray_data",
        help="Input X-ray diffraction data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    mud_parser.add_argument("mud", type=float, help="mu*d value")
    _add_common_args(mud_parser, use_gui)

    # -----------------------
    # ZSCAN parser
    # -----------------------
    zscan_parser = subp.add_parser(
        "zscan", help="Correct using a z-scan measurement"
    )
    zscan_parser.add_argument(
        "xray_data",
        help="Input X-ray diffraction data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    zscan_parser.add_argument(
        "zscan_file",
        help="Z-scan measurement file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    _add_common_args(zscan_parser, use_gui)

    # -----------------------
    # SAMPLE parser
    # -----------------------
    sample_parser = subp.add_parser(
        "sample", help="Correct using sample composition/density"
    )
    sample_parser.add_argument(
        "xray_data",
        help="Input X-ray diffraction data file",
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    sample_parser.add_argument(
        "--composition", required=True, help="Chemical formula, e.g. Fe2O3"
    )
    sample_parser.add_argument(
        "--energy",
        required=True,
        help=(
            "Incident X-ray energy in keV (numeric) or known "
            "source (CuKa, MoKa, etc.)"
        ),
    )
    sample_parser.add_argument(
        "--density",
        required=True,
        type=float,
        help="Sample mass density in g/cm^3",
    )
    _add_common_args(sample_parser, use_gui)

    return parser


# -----------------------
# CLI / GUI dispatch
# -----------------------


@Gooey(
    program_name="labpdfproc",
    required_cols=1,
    optional_cols=1,
    show_sidebar=True,
)
def get_args_gui():
    parser = create_parser(use_gui=True)
    return parser.parse_args()


def get_args_cli(override=None):
    parser = create_parser(use_gui=False)
    return parser.parse_args(override)


def main():
    use_gui = len(sys.argv) == 1 or "--gui" in sys.argv
    args = get_args_gui() if use_gui else get_args_cli()

    if args.command == "mud":
        run_mud(args)
    elif args.command == "zscan":
        run_zscan(args)
    elif args.command == "sample":
        run_sample(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

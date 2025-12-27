import argparse
import sys
from pathlib import Path

from gooey import Gooey, GooeyParser

from diffpy.labpdfproc.functions import CVE_METHODS, apply_corr, compute_cve
from diffpy.labpdfproc.tools import WAVELENGTHS
from diffpy.utils.diffraction_objects import XQUANTITIES, DiffractionObject
from diffpy.utils.parsers.loaddata import loadData
from diffpy.utils.tools import compute_mu_using_xraydb, compute_mud

# -----------------------
# Helper functions
# -----------------------


def _add_common_args(parser, use_gui=False):
    parser.add_argument(
        "-x",
        "--xtype",
        help=(
            "X-axis type (default: tth). Allowed values: "
            f"{', '.join(XQUANTITIES)}"
        ),
        default="tth",
        choices=XQUANTITIES,
    )
    parser.add_argument(
        "-m",
        "--method",
        help=(
            "Method for cylindrical volume element (CVE) calculation "
            "(default: polynomial_interpolation). Allowed methods: "
            f"{', '.join(CVE_METHODS)}"
        ),
        default="polynomial_interpolation",
        choices=CVE_METHODS,
    )
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
        "-c",
        "--output-correction",
        help="Also output the absorption correction to a separate file",
        action="store_true",
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


def _ensure_metadata(obj):
    obj.metadata = obj.metadata or {}


def _save_corrected(corrected, input_path, target_dir, force, xtype):
    target_dir = Path(target_dir) if target_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    outfile = target_dir / (input_path.stem + "_corrected.chi")
    if outfile.exists() and not force:
        print(f"WARNING: {outfile} exists. Use --force to overwrite.")
        return
    _ensure_metadata(corrected)
    corrected.dump(str(outfile), xtype=xtype)
    print(f"Saved corrected data to {outfile}")


def _save_correction(correction, input_path, target_dir, force, xtype):
    target_dir = Path(target_dir) if target_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    corrfile = target_dir / (input_path.stem + "_cve.chi")
    if corrfile.exists() and not force:
        print(f"WARNING: {corrfile} exists. Use --force to overwrite.")
        return
    _ensure_metadata(correction)
    correction.dump(str(corrfile), xtype=xtype)
    print(f"Saved correction data to {corrfile}")


def _attach_credit_metadata(pattern, args):
    """Add optional user credit metadata if provided."""
    if pattern.metadata is None:
        pattern.metadata = {}
    for key in ("username", "email", "orcid"):
        value = getattr(args, key, None)
        if value:
            pattern.metadata[key] = value


def _load_pattern(path, xtype, wavelength=None):
    x, y = loadData(path, unpack=True)
    return DiffractionObject(
        xarray=x,
        yarray=y,
        xtype=xtype,
        wavelength=wavelength,
        scat_quantity="x-ray",
        name=path.stem,
        metadata={},
    )


def resolve_wavelength(wavelength):
    """Resolve wavelength from user input.

    Parameters
    ----------
    w : str or float
        User input for wavelength. Can be numeric (in Angstroms) or
        a string X-ray source name (e.g. 'CuKa').

    Returns
    -------
    float
        Wavelength in Angstroms.
    """
    if wavelength is None:
        raise ValueError(
            "X-ray wavelength must be provided as a positional argument "
            "after the diffraction data file."
        )
    try:
        return float(wavelength)
    except (TypeError, ValueError):
        pass
    sources = sorted(WAVELENGTHS.keys())
    matched = next(
        (k for k in sources if k.lower() == str(wavelength).strip().lower()),
        None,
    )
    if matched is None:
        raise ValueError(
            f"Unknown X-ray source '{wavelength}'. "
            f"Allowed sources are: {', '.join(sources)}."
        )
    return WAVELENGTHS[matched]


# -----------------------
# Subcommand functions
# -----------------------


def run_mud(args):
    path = Path(args.xray_data)
    wavelength = resolve_wavelength(args.wavelength)
    pattern = _load_pattern(path, args.xtype, wavelength)
    correction = compute_cve(
        pattern, args.mud, method=args.method, xtype=args.xtype
    )
    _attach_credit_metadata(correction, args)
    corrected_data = apply_corr(pattern, correction)
    corrected_data.name = f"Absorption corrected input_data: {pattern.name}"
    _attach_credit_metadata(corrected_data, args)
    _save_corrected(
        corrected_data, path, args.target_dir, args.force, args.xtype
    )
    if args.output_correction:
        _save_correction(
            correction, path, args.target_dir, args.force, args.xtype
        )


def run_zscan(args):
    pattern_path = Path(args.xray_data)
    zscan_path = Path(args.zscan_file)
    wavelength = resolve_wavelength(args.wavelength)
    mud = compute_mud(zscan_path)
    print(f"Computed mu*D = {mud:.4f} from z-scan file")
    pattern = _load_pattern(pattern_path, args.xtype, wavelength)
    correction = compute_cve(
        pattern, mud, method=args.method, xtype=args.xtype
    )
    _attach_credit_metadata(correction, args)
    corrected_data = apply_corr(pattern, correction)
    corrected_data.name = f"Absorption corrected input_data: {pattern.name}"
    _attach_credit_metadata(corrected_data, args)
    _save_corrected(
        corrected_data, pattern_path, args.target_dir, args.force, args.xtype
    )
    if args.output_correction:
        _save_correction(
            correction, pattern_path, args.target_dir, args.force, args.xtype
        )


def run_sample(args):
    path = Path(args.xray_data)
    wavelength = resolve_wavelength(args.wavelength)
    energy_kev = 12.398 / wavelength  # Convert Å to keV
    mud = compute_mu_using_xraydb(
        args.composition,
        energy_kev,
        args.density,
    )
    print(
        f"Computed mu*D = {mud:.4f} for {args.composition} "
        f"at λ = {wavelength:.4f} Å"
    )
    pattern = _load_pattern(path, args.xtype, wavelength)
    correction = compute_cve(
        pattern, mud, method=args.method, xtype=args.xtype
    )
    _attach_credit_metadata(correction, args)
    corrected_data = apply_corr(pattern, correction)
    corrected_data.name = f"Absorption corrected input_data: {pattern.name}"
    _attach_credit_metadata(corrected_data, args)
    _save_corrected(
        corrected_data, path, args.target_dir, args.force, args.xtype
    )
    if args.output_correction:
        _save_correction(
            correction, path, args.target_dir, args.force, args.xtype
        )


def add_positional_wavelength(parser):
    parser.add_argument(
        "wavelength",
        help=(
            "X-ray wavelength in angstroms (numeric) or X-ray source name "
            f"(allowed: {', '.join(sorted(WAVELENGTHS.keys()))})."
        ),
    )


# -----------------------
# Parser construction
# -----------------------


def create_parser(use_gui=False):
    Parser = GooeyParser if use_gui else argparse.ArgumentParser
    parser = Parser(
        prog="labpdfproc",
        description=(
            "Apply absorption corrections to laboratory X-ray diffraction "
            "data prior to PDF analysis. Supports manual mu*d, "
            "z-scan, or sample-based corrections."
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
    add_positional_wavelength(mud_parser)
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
    add_positional_wavelength(zscan_parser)
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
    add_positional_wavelength(sample_parser)
    sample_parser.add_argument(
        "composition",
        help="Chemical formula, e.g. Fe2O3",
    )
    sample_parser.add_argument(
        "density",
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

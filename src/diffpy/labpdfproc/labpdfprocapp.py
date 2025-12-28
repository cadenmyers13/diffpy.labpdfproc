import argparse
import sys
from pathlib import Path

from gooey import Gooey, GooeyParser

from diffpy.labpdfproc.functions import CVE_METHODS, apply_corr, compute_cve
from diffpy.labpdfproc.tools import WAVELENGTHS, load_metadata
from diffpy.utils.diffraction_objects import XQUANTITIES, DiffractionObject
from diffpy.utils.parsers.loaddata import loadData
from diffpy.utils.tools import (
    check_and_build_global_config,
    compute_mu_using_xraydb,
    compute_mud,
    get_package_info,
    get_user_info,
)

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
    parser.add_argument(
        "-u",
        "--user-metadata",
        help=(
            "Specify key-value pairs to be loaded into metadata "
            "using the format key=value. "
            "Separate pairs with whitespace, "
            "and ensure no whitespaces before or after the = sign. "
            "Avoid using = in keys. If multiple = signs are present, "
            "only the first separates the key and value. "
            "If a key or value contains whitespace, enclose it in quotes. "
            "For example, facility='NSLS II', "
            "'facility=NSLS II', beamline=28ID-2, "
            "'beamline'='28ID-2', 'favorite color'=blue, "
            "are all valid key=value items."
        ),
        nargs="+",
        metavar="KEY=VALUE",
    )
    _add_credit_args(parser, use_gui)
    return parser


def _add_credit_args(parser, use_gui=False):
    parser.add_argument(
        "--username",
        help=(
            "Your name (optional, for dataset credit). "
            "Will be loaded from config files if not specified."
        ),
        default=None,
        **({"widget": "TextField"} if use_gui else {}),
    )
    parser.add_argument(
        "--email",
        help=(
            "Your email (optional, for dataset credit). "
            "Will be loaded from config files if not specified."
        ),
        default=None,
        **({"widget": "TextField"} if use_gui else {}),
    )
    parser.add_argument(
        "--orcid",
        help=(
            "Your ORCID ID (optional, for dataset credit). "
            "Will be loaded from config files if not specified."
        ),
        default=None,
        **({"widget": "TextField"} if use_gui else {}),
    )


def _load_user_metadata(args):
    """Load user-provided key=value metadata pairs into args."""
    if not args.user_metadata:
        return args
    reserved_keys = set(vars(args).keys())
    for item in args.user_metadata:
        if "=" not in item:
            raise ValueError(
                "Please provide key-value pairs in the format key=value. "
                "For more information, use `labpdfproc --help`."
            )
        items = item.split("=")
        key = items[0].strip()
        value = "=".join(items[1:]).strip() if len(items) > 1 else ""
        if key in reserved_keys:
            raise ValueError(
                f"{key} is a reserved name. "
                f"Please use a different key name."
            )
        if hasattr(args, key):
            raise ValueError(f"Duplicate key: {key}.")
        setattr(args, key, value)
    return args


def _load_user_info(args):
    """Load user info from config or prompt if not provided."""
    if args.username is None or args.email is None:
        check_and_build_global_config()
    config = get_user_info(
        owner_name=args.username,
        owner_email=args.email,
        owner_orcid=args.orcid,
    )
    args.username = config.get("owner_name")
    args.email = config.get("owner_email")
    args.orcid = config.get("owner_orcid")
    return args


def _load_package_info(args):
    """Load package info into args."""
    metadata = get_package_info("diffpy.labpdfproc")
    setattr(args, "package_info", metadata["package_info"])
    return args


def _save_corrected(corrected, input_path, args):
    target_dir = Path(args.target_dir) if args.target_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    outfile = target_dir / (input_path.stem + "_corrected.chi")
    if outfile.exists() and not args.force:
        print(f"WARNING: {outfile} exists. Use --force to overwrite.")
        return
    corrected.metadata = corrected.metadata or {}
    corrected.dump(str(outfile), xtype=args.xtype)
    print(f"Saved corrected data to {outfile}")


def _save_correction(correction, input_path, args):
    target_dir = Path(args.target_dir) if args.target_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    corrfile = target_dir / (input_path.stem + "_cve.chi")
    if corrfile.exists() and not args.force:
        print(f"WARNING: {corrfile} exists. Use --force to overwrite.")
        return
    correction.metadata = correction.metadata or {}
    correction.dump(str(corrfile), xtype=args.xtype)
    print(f"Saved correction data to {corrfile}")


def _load_pattern(path, xtype, wavelength, metadata):
    x, y = loadData(path, unpack=True)
    return DiffractionObject(
        xarray=x,
        yarray=y,
        xtype=xtype,
        wavelength=wavelength,
        scat_quantity="x-ray",
        name=path.stem,
        metadata=metadata,
    )


def resolve_wavelength(wavelength):
    """Resolve wavelength from user input.

    Parameters
    ----------
    wavelength : str or float
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


def _prepare_args(args, input_path):
    """Prepare args by loading metadata, user info, and package info."""
    args_copy = argparse.Namespace(**vars(args))
    args_copy.output_directory = (
        Path(args.target_dir) if args.target_dir else Path.cwd()
    )
    args_copy = _load_user_metadata(args_copy)
    args_copy = _load_user_info(args_copy)
    args_copy = _load_package_info(args_copy)
    metadata = load_metadata(args_copy, input_path)
    return metadata


# -----------------------
# Subcommand functions
# -----------------------


def run_mud(args):
    """Run mu*d based absorption correction."""
    path = Path(args.xray_data)
    wavelength = resolve_wavelength(args.wavelength)
    args.mud = args.mud_value
    metadata = _prepare_args(args, path)
    pattern = _load_pattern(path, args.xtype, wavelength, metadata)
    correction = compute_cve(
        pattern, args.mud, method=args.method, xtype=args.xtype
    )
    correction.metadata = metadata.copy()
    corrected_data = apply_corr(pattern, correction)
    corrected_data.name = f"Absorption corrected input_data: {pattern.name}"
    corrected_data.metadata = metadata.copy()
    _save_corrected(corrected_data, path, args)
    if args.output_correction:
        _save_correction(correction, path, args)


def run_zscan(args):
    """Run z-scan based absorption correction."""
    pattern_path = Path(args.xray_data)
    zscan_path = Path(args.zscan_file)
    wavelength = resolve_wavelength(args.wavelength)
    mud = compute_mud(zscan_path)
    print(f"Computed mu*D = {mud:.4f} from z-scan file")
    args.mud = mud
    args.z_scan_file = str(zscan_path)
    metadata = _prepare_args(args, pattern_path)
    pattern = _load_pattern(pattern_path, args.xtype, wavelength, metadata)
    correction = compute_cve(
        pattern, mud, method=args.method, xtype=args.xtype
    )
    correction.metadata = metadata.copy()
    corrected_data = apply_corr(pattern, correction)
    corrected_data.name = f"Absorption corrected input_data: {pattern.name}"
    corrected_data.metadata = metadata.copy()
    _save_corrected(corrected_data, pattern_path, args)
    if args.output_correction:
        _save_correction(correction, pattern_path, args)


def run_sample(args):
    """Run sample composition/density based absorption correction."""
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
    args.mud = mud
    args.sample_composition = args.composition
    args.sample_mass_density = args.density
    args.energy = energy_kev
    metadata = _prepare_args(args, path)
    pattern = _load_pattern(path, args.xtype, wavelength, metadata)
    correction = compute_cve(
        pattern, mud, method=args.method, xtype=args.xtype
    )
    correction.metadata = metadata.copy()
    corrected_data = apply_corr(pattern, correction)
    corrected_data.name = f"Absorption corrected input_data: {pattern.name}"
    corrected_data.metadata = metadata.copy()
    _save_corrected(corrected_data, path, args)
    if args.output_correction:
        _save_correction(correction, path, args)


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
    mud_parser.add_argument(
        "mud_value", type=float, help="mu*d value", metavar="mud"
    )
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

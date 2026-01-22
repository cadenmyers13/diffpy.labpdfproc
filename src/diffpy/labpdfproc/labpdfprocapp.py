import argparse
import sys

from gooey import Gooey, GooeyParser

from diffpy.labpdfproc.functions import CVE_METHODS, apply_corr, compute_cve
from diffpy.labpdfproc.tools import (
    WAVELENGTHS,
    known_sources,
    load_metadata,
    preprocessing_args,
)
from diffpy.utils.diffraction_objects import XQUANTITIES, DiffractionObject
from diffpy.utils.parsers.loaddata import loadData


def _wavelength_type(value):
    """Parse wavelength as float or named source (case-insensitive)."""
    try:
        return float(value)
    except ValueError:
        key = value.lower()
        wavelengths = {k.lower(): v for k, v in WAVELENGTHS.items()}
        try:
            return wavelengths[key]
        except KeyError:
            raise ValueError(
                f"Anode type 'invalid' not recognized. "
                "Please rerun specifying an anode type from "
                f"{*known_sources, }."
            )


def _add_common_args(parser, use_gui=False):
    parser.add_argument(
        "-w",
        "--wavelength",
        help=(
            "X-ray wavelength in angstroms (numeric) or X-ray source name "
            f"(allowed: {', '.join(sorted(WAVELENGTHS.keys()))}). "
            "Will be loaded from config files if not specified."
        ),
        default=None,
        type=_wavelength_type,
    )
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
        "-o",
        "--output-directory",
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


def _save_corrected(corrected, input_path, args):
    outfile = args.output_directory / (input_path.stem + "_corrected.chi")
    if outfile.exists() and not args.force:
        print(f"WARNING: {outfile} exists. Use --force to overwrite.")
        return
    corrected.metadata = corrected.metadata or {}
    corrected.dump(str(outfile), xtype=args.xtype)
    print(f"Saved corrected data to {outfile}")


def _save_correction(correction, input_path, args):
    corrfile = args.output_directory / (input_path.stem + "_cve.chi")
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


def _process_files(args):
    """Process all input files with absorption correction."""
    for path in args.input_paths:
        metadata = load_metadata(args, path)
        pattern = _load_pattern(path, args.xtype, args.wavelength, metadata)
        correction = compute_cve(
            pattern, args.mud, method=args.method, xtype=args.xtype
        )
        correction.metadata = metadata.copy()
        corrected_data = apply_corr(pattern, correction)
        corrected_data.name = (
            f"Absorption corrected input_data: {pattern.name}"
        )
        corrected_data.metadata = metadata.copy()
        _save_corrected(corrected_data, path, args)
        if args.output_correction:
            _save_correction(correction, path, args)


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

    # MUD parser
    mud_parser = subp.add_parser(
        "mud", help="Correct diffraction data using known mu*d value"
    )
    mud_parser.add_argument(
        "input",
        nargs="+",
        help=(
            "Input X-ray diffraction data file(s) or directory. "
            "Can specify multiple files, directories, or use wildcards."
        ),
        **({"widget": "MultiFileChooser"} if use_gui else {}),
    )
    mud_parser.add_argument(
        "mud_value", type=float, help="mu*d value", metavar="mud"
    )
    _add_common_args(mud_parser, use_gui)

    # ZSCAN parser
    zscan_parser = subp.add_parser(
        "zscan", help="Correct diffraction data using a z-scan measurement"
    )
    zscan_parser.add_argument(
        "input",
        nargs="+",
        help=(
            "Input X-ray diffraction data file(s) or directory. "
            "Can specify multiple files, directories, or use wildcards."
        ),
        **({"widget": "MultiFileChooser"} if use_gui else {}),
    )
    zscan_parser.add_argument(
        "z_scan_file",
        help=(
            "Z-scan measurement file. "
            "See diffpy.labpdfproc documentation for more information."
        ),
        **({"widget": "FileChooser"} if use_gui else {}),
    )
    _add_common_args(zscan_parser, use_gui)

    # SAMPLE parser
    sample_parser = subp.add_parser(
        "sample",
        help="Correct diffraction data using sample composition/density",
    )
    sample_parser.add_argument(
        "input",
        nargs="+",
        help=(
            "Input X-ray diffraction data file(s) or directory. "
            "Can specify multiple files, directories, or use wildcards."
        ),
        **({"widget": "MultiFileChooser"} if use_gui else {}),
    )
    sample_parser.add_argument(
        "sample_composition",
        help="Chemical formula, e.g. Fe2O3",
    )
    sample_parser.add_argument(
        "sample_mass_density",
        type=float,
        help="Sample mass density in capillary (g/cm^3)",
    )
    _add_common_args(sample_parser, use_gui)

    return parser


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
        args.mud = args.mud_value
    args = preprocessing_args(args)
    _process_files(args)


if __name__ == "__main__":
    main()

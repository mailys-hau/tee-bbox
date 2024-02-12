import click as cli

from pathlib import Path

from boxes import *
from loaders import *
from savers import SAVERS



@cli.command(context_settings={"help_option_names": ["-h", "--help"],
                               "show_default": True})
@cli.argument("pin", type=cli.Path(exists=True, file_okay=False, path_type=Path, resolve_path=True))
@cli.argument("ptg", type=cli.Path(exists=True, file_okay=False, path_type=Path, resolve_path=True))
@cli.option("--ge", "-G", "vendor", flag_value="ge", default=True,
            help="Use GE format loader and annotation process.")
@cli.option("--philips", "-P", "vendor", flag_value="philips",
            help="Use Philips format loader and annotation process.")
## Bounding box info: size, shape, ... ##
@cli.option("--bbox-shape", "-b", "bshape", default="rectangle",
            type=cli.Choice(["rectangle", "valve"], case_sensitive=False),
            help="Whether to have a rectangle-shaped bounding box or one that follows the mitral valve's shape.")
@cli.option("--thickness", "-t", "thick", default=3, type=cli.FloatRange(min=0),
            help="Minimal thickness of the bounding box in mm.")
@cli.option("--voxel-resolution", "-v", "voxres", default=0.5, type=cli.FloatRange(min=0),
            help="Voxel resolution in mm. Will be the same for every directions.")
@cli.option("--leaflets", "-L", "merge", flag_value=False,
            help="Generate one bounding box per leaflet.")
@cli.option("--mitral-valve", "-MV", "merge", flag_value=True, default=True,
            help="Generate one bounding box for the whole mitral valve.")
## Storage info ##
@cli.option("--output", "-o", "pout", default="bboxes",
            type=cli.Path(path_type=Path, resolve_path=True),
            help="Where created bounding boxes will be stored.")
@cli.option("--save-format", "-s", type=cli.Choice(list(SAVERS.keys()), case_sensitive=False),
            default="hdf", help="Whether to save processed data in HDFs or NIFTIs.")
def make_bbox(pin, ptg, vendor, bshape, thick, voxres, merge, pout, save_format):
    """
    Create a bounding box around the mitral valve and store it in desired format.

    PDATA    DIR    Where the data are stored.
    """
    if vendor == "ge" and OS_NAME == "Windows":
        raise ValueError("Can only run `--ge` option on Windows OS.")
    loader = load_ge if vendor == "ge" else load_philips
    shaper = normal_extrude if bshape == "valve" else rectangle_extrude
    saver = SAVERS[save_format.lower()]
    pout.mkdir(parents=True, exist_ok=True)
    voxres = [voxres, voxres, voxres]
    for fname in pin.iterdir():
        #FIXME: Load dicom
        with open(fname, "br") as fd:
            # Load (necessary data to create) surface
            # If merge, return mitral valve surface as a whole
            surface, vinp = loader(fname, ptg, voxres, merge)
        # Create bbox
        if not merge:
            bbox = [ shaper(s.mesh, s.voxinfo, thick) for s in surface ]
        else:
            bbox = shaper(surface.mesh, surface.voxinfo, thick)
        # Save all info
        saver(bbox, vinp, pout.joinpath(fname.stem), bshape)



if __name__ == "__main__":
    make_bbox()

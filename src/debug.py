import click as cli
import numpy as np
import trimesh as tm

from pathlib import Path, WindowsPath
from skimage.measure import marching_cubes
from trimesh.voxel.base import VoxelGrid

from boxes import *
from loaders import *



def bbox2mesh(bbox, vinfo, which):
    if bbox.sum() == 0:
        print(f"Generated bounding box of the {which} is empty, RiP.")
        print("Stopping debug here.")
        return
    verts, faces, vnormals, _ = marching_cubes(bbox, spacing=vinfo.spacing)
    verts = verts @ vinfo.directions + vinfo.origin
    mesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=vnormals)
    return mesh


@cli.command()
@cli.option("--philips", "-P", "vendor", flag_value="philips", default=True)
@cli.option("--ge", "-G", "vendor", flag_value="ge", default=False)
@cli.option("--shape", "-s", "bshape", default="rectangle",
            type=cli.Choice(["rectangle", "valve"], case_sensitive=False))
def debug_me(vendor, bshape):
    # Get data sample
    root = Path("~/Documents/data/debug-nxp").expanduser()
    vr = np.array([0.5, 0.5, 0.5])
    if vendor == "ge":
        pin = WindowsPath(root.joinpath("file.dcm"))
        pgt = pin.parent
        loader = load_ge
        thickness, midaxis = 0.003, 1
        vr /= 1e3
    else:
        pin = root.joinpath("images/file.nrrd")
        pgt = root.joinpath("masks_es")
        loader = load_philips
        thickness, midaxis = 3, 2

    meshes, vinp = loader(pin, pgt, vr, False)
    vinfo = meshes[0].voxinfo
    # Get all types of mesh
    amesh, pmesh = meshes[0].mesh, meshes[1].mesh
    mesh = tm.util.concatenate([amesh, pmesh])
    amesh.visual.face_colors = [203, 41, 214, 150]
    pmesh.visual.face_colors = [31, 204, 219, 150]
    mesh.visual.face_colors = [173, 127, 0, 150]

    # Get that bounding box
    shaper = rectangle_extrude if bshape == "rectangle" else normal_extrude
    abbox = shaper(amesh, vinfo, thickness, midaxis).voxel
    pbbox = shaper(pmesh, vinfo, thickness, midaxis).voxel
    mbbox = shaper(mesh, vinfo, thickness, midaxis).voxel
    abmesh = bbox2mesh(abbox, vinfo, "anterior leaflet")
    pbmesh = bbox2mesh(pbbox, vinfo, "posterior leaflet")
    mbmesh = bbox2mesh(mbbox, vinfo, "mitral valve")
    if abmesh is None or pbmesh is None or mbmesh is None:
        return
    abmesh.visual.face_colors = [203, 41, 214, 150]
    pbmesh.visual.face_colors = [31, 204, 219, 150]
    mbmesh.visual.face_colors = [173, 127, 0, 150]

    #  Plot thingies
    sc1 = tm.Scene()
    sc1.add_geometry(amesh), sc1.add_geometry(pmesh)
    sc1.add_geometry(abmesh), sc1.add_geometry(pbmesh)
    sc1.show(flags={"axis": True, "wireframe": True})
    sc2 = tm.Scene()
    sc2.add_geometry(mesh)
    sc2.add_geometry(mbmesh)
    sc2.show(flags={"axis": True, "wireframe": True})



if __name__ == "__main__":
    debug_me()

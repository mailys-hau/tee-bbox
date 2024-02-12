import click as cli
import numpy as np
import trimesh as tm

from pathlib import Path
from skimage.measure import marching_cubes
from trimesh.voxel.base import VoxelGrid

from boxes import *
from loaders import *


@cli.command()
@cli.option("--philips", "-P", "vendor", flag_value="philips", default=True)
@cli.option("--ge", "-G", "vendor", flag_value="ge", default=False)
def debug_me(vendor):
    # Get data sample
    root = Path("~/Documents/data/debug-nxp").expanduser()
    if vendor == "ge":
        pin = root.joinpath("file.dcm")
        pgt = pin.with_suffix('')
        loader = load_ge
    else:
        pin = root.joinpath("images/file.nrrd")
        pgt = root.joinpath("masks_es")
        loader = load_philips

    vr = np.array([0.5, 0.5, 0.5])
    meshes, vinp, voxels = loader(pin, pgt, vr, False)
    vinfo = meshes[0].voxinfo
    # Get all types of mesh
    amesh, pmesh = meshes[0].mesh, meshes[1].mesh
    mesh = tm.util.concatenate([amesh, pmesh])
    amesh.visual.face_colors = [203, 41, 214, 150]
    pmesh.visual.face_colors = [31, 204, 219, 150]
    mesh.visual.face_colors = [173, 127, 0, 150]
    # Get all types of voxel grids, Trimesh type
    avg = VoxelGrid(voxels[0].voxel, transform=meshes[0].affine)
    pvg = VoxelGrid(voxels[1].voxel, transform=meshes[1].affine)
    mv = voxels[0].voxel | voxels[1].voxel
    vg = VoxelGrid(mv, transform=meshes[1].affine)

    # Get that bounding box
    bbox = rectangle_extrude(mesh, vinfo, midaxis=2).voxel
    if bbox.sum() == 0:
        print("Generated bounding box is empty, RiP.")
        print("Stopping debug here.")
        return
    verts, faces, vnormals, _ = marching_cubes(bbox, spacing=vinfo.spacing)
    bmesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=vnormals)
    bmesh.visual.face_colors = [173, 127, 0, 150]

    #  Plot thingies
    sc = tm.Scene()
    sc.add_geometry(amesh), sc.add_geometry(pmesh)
    #sc.add_geometry(bmesh)
    sc.show(flags={"axis": True, "wireframe": True})
    #sc.add_geometry(mesh)
    #sc.show(flags={"axis": True, "wireframe": True})



if __name__ == "__main__":
    debug_me()

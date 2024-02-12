"""
Load input & mitral annulus *surface* from GE. #Put everything in an RAS coordinate system.
"""

import trimesh as tm

from dicoms import dcm2vox
from loaders.ply import full_load_ply
from utils import label2onehot, MeshTEE



def load_ge(fname, pply, voxres, merge=True):
    pply = pply.joinpath(fname.stem)
    # Select middle frame time *from meshes* (aka mid-systole)
    times = [ float(fn.stem.split('-')[-1]) for fn in pply.glob("anterior*")]
    times.sort()
    mf = times[int(len(times) / 2) + 1]
    # Load necessary info from DICOM as VoxelTEE (=Input US)
    vinp = dcm2vox(fname, mf, voxres) # Already converted in mm
    # Load PLY meshes
    with open(fname.joinpath(f"anterior-{mf}.ply"), "br") as fd:
        damesh = full_load_ply(fd, prefer_color="faces")
    amesh = MeshTEE(tm.Trimesh(**damesh), vinp.voxinfo)
    with open(pply.joinpath(f"posterior-{mf}.ply"), "br") as fd:
        dpmesh = full_load_ply(fd, prefer_color="faces")
    pmesh = MeshTEE(tm.Trimesh(**dpmesh), vinp.voxinfo)
    #amesh, pmesh = amesh.apply_affine(), pmesh.apply_affine()
    # Merge meshes if needed
    mesh = MeshTEE.concat([amesh, pmesh]) if merge else [amesh, pmesh]
    return mesh, vinp

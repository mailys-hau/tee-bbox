"""
Load input & mitral annulus *surface* from GE. #Put everything in an RAS coordinate system.
"""

import trimesh as tm

from loaders.dicoms import dcm2vox
from loaders.plys import full_load_ply
from utils import label2onehot, MeshTEE



def load_ge(fname, pply, voxres, merge=True):
    pply = pply.joinpath(fname.stem)
    if not pply.exists():
        print(f"No surface found for {fname.name}, skipping it.")
        return None, None
    # Select middle frame time *from meshes* (aka mid-systole)
    times = [ float(fn.stem.split('-')[-1]) for fn in pply.glob("anterior*")]
    times.sort()
    try:
        mf = times[int(len(times) / 2) + 1]
    except IndexError: # If there's not enough frames we're gonna end up heretimes
        mf = times[-1]
    # Load necessary info from DICOM as VoxelTEE (=Input US)
    vinp = dcm2vox(fname, mf, voxres) # Already converted in mm
    # Load PLY meshes
    with open(pply.joinpath(f"anterior-{mf}.ply"), "br") as fd:
        damesh = full_load_ply(fd, prefer_color="faces")
    amesh = MeshTEE(tm.Trimesh(**damesh), voxinfo=vinp.voxinfo)
    with open(pply.joinpath(f"posterior-{mf}.ply"), "br") as fd:
        dpmesh = full_load_ply(fd, prefer_color="faces")
    pmesh = MeshTEE(tm.Trimesh(**dpmesh), voxinfo=vinp.voxinfo)
    #amesh.apply_affine(), pmesh.apply_affine()
    # Merge meshes if needed
    mesh = MeshTEE.concat([amesh, pmesh]) if merge else [amesh, pmesh]
    return mesh, vinp

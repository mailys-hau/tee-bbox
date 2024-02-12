"""
Load input & GT from various vendors. Put everything in an RAS coordinate system.
"""

import nrrd
import numpy as np
import trimesh as tm

from scipy.ndimage import affine_transform
from skimage.measure import marching_cubes

from utils import label2onehot, get_affine, MeshTEE, VoxelInfo, VoxelTEE
from utils.voxels import mid_vox_grid



def load_philips(fname, dmask, voxres, merge=True):
    iimg, info = nrrd.read(fname)
    # Should only yield one file
    gtname = next(dmask.glob(f"{fname.stem}*{fname.suffix}"))
    fid = int(gtname.stem.split('_')[-1])
    vinp = VoxelTEE(iimg[fid], origin=info["space origin"],
                    directions=info["space directions"][1:],
                    spacing=abs(info["space directions"][1:].diagonal()),
                    shape=iimg.shape, frame_time=fid)
    vinp.resample(voxres)
    gtimg, gtinfo = nrrd.read(gtname) #FIXME: We get LSP, want RAS
    # Philips GT contains 0=background, 1=MA, 2=anterior, 3=posterior
    # => Only fetch 2 and 3
    gtarr = label2onehot(gtimg, [0, 1])
    # GT affine may be different than input, so use `gtinfo`
    avox = VoxelTEE(mid_vox_grid(gtarr[0]), origin=gtinfo["space origin"],
                    directions=gtinfo["space directions"], spacing=voxres,
                    shape=gtarr[0].shape, frame_time=fid)
    pvox = VoxelTEE(mid_vox_grid(gtarr[1]), origin=gtinfo["space origin"],
                    directions=gtinfo["space directions"],
                    spacing=abs(gtinfo["space directions"].diagonal()),
                    shape=gtarr[1].shape, frame_time=fid)
    avox.resample(voxres), pvox.resample(voxres)
    amesh, pmesh = avox.to_mesh(), pvox.to_mesh()
    amesh.apply_affine(), pmesh.apply_affine()
    mesh = MeshTEE.concat([amesh, pmesh]) if merge else [amesh, pmesh]
    return mesh, vinp

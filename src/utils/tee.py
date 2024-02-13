"""
Imported from https://github.com/mailys-hau/echovox (except `VoxelGrid`)
"""
import numpy as np
import scipy.ndimage as scpn
import trimesh as tm
import warnings

from skimage.measure import marching_cubes

from utils.voxels import VoxelInfo



warnings.simplefilter("default", UserWarning)




class _TEE:
    """ Input TEE as a mesh """
    def __init__(self, tee, tee_type, **voxinfo):
        if tee_type == "voxel":
            self._voxel = tee
            self._mesh = None
        elif tee_type == "mesh":
            self._mesh = tee
            self._voxel = None
        else:
            raise ValueError(f"Unknown TEE type, got `{tee_type}`.")
        self.voxinfo = voxinfo["voxinfo"] if "voxinfo" in voxinfo else VoxelInfo(**voxinfo)

    @property
    def spacing(self):
        return self.voxinfo.spacing

    @property
    def mesh(self):
        if self._mesh is None:
            verts, faces, vnormals, _ = marching_cubes(self.voxel, spacing=self.spacing)
            self._mesh = tm.Trimesh(vertices=verts, faces=faces,
                                    vertex_normals=vnormals)
        return self._mesh

    @property
    def voxel(self):
        if self._voxel is None:
            raise NotImplementedError #TODO
        return self._voxel

    def to_m(self): #FIXME? Add resample
        self.voxinfo.to_m()

    def to_mm(self):
        self.voxinfo.to_mm()



class VoxelTEE(_TEE):
    def __init__(self, tee, **voxinfo):
        if voxinfo.get("shape", ()) == ():
            voxinfo["shape"] = tee.shape
        super(VoxelTEE, self).__init__(tee, "voxel", **voxinfo)

    def resample(self, new_spacing):
        if isinstance(new_spacing, list):
            new = np.array(new_spacing)
        elif isinstance(new_spacing, float):
            new = np.array([new_spacing, new_spacing, new_spacing])
        else:
            if not isinstance(new_spacing, np.ndarray):
                raise ValueError(f"Spacing must be a `float`, `list`, or `numpy.ndarray`, got {type(new_spacing)}.")
            new = new_spacing
        zoom = self.voxinfo.spacing / new
        self._voxel = scpn.zoom(self.voxel, zoom)
        self.voxinfo.spacing = new
        self.shape = self.voxel.shape
        self.voxinfo._affine()

    def to_mesh(self):
        return MeshTEE(self.mesh, voxinfo=self.voxinfo)

class MeshTEE(_TEE):
    def __init__(self, tee, **voxinfo):
        super(MeshTEE, self).__init__(tee, "mesh", **voxinfo)

    @property
    def affine(self):
        return self.voxinfo.affine

    def apply_affine(self, affine=None):
        if affine is not None and self.affine is not None:
            warnings.warn("Mesh already had an affine, it will be overwritten by the given one", UserWarning)
        else:
            affine = self.affine
        self.mesh.apply_transform(affine)

    @staticmethod
    def concat(meshes):
        tmeshes = [ m.mesh for m in meshes ]
        # Assume same voxinfo for all
        return MeshTEE(tm.util.concatenate(tmeshes), voxinfo=meshes[0].voxinfo)

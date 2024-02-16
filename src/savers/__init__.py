import h5py
import nibabel as nib
import nrrd
import numpy as np

from utils import onehot2label



def save_as_hdf(bbox, vinput, fname, bshape="rectangle"):
    hdf = h5py.File(fname.with_suffix(".h5"), 'w')
    hdf.create_dataset("CartesianVolume", data=vinput.voxel)
    tg = hdf.create_group("/GroundTruth")
    if isinstance(bbox, list):
        tg.create_dataset("anterior", data=bbox[0].voxel)
        tg.create_dataset("posterior", data=bbox[1].voxel)
    else:
        tg.create_dataset("mitralValve", data=bbox.voxel)
    # Assume same info for input & GT
    vinfo = bbox[0].voxinfo if isinstance(bbox, list) else bbox.voxinfo
    info = hdf.create_group("/VolumeGeometry")
    info.create_dataset("frameTime", data=vinfo.frame_time)
    info.create_dataset("origin", data=vinfo.origin)
    info.create_dataset("directions", data=vinfo.directions)
    info.create_dataset("resolution", data=vinfo.spacing)
    info.create_dataset("shape", data=vinfo.shape)
    info.create_dataset("bboxShape", data=bshape)
    info.create_dataset("unit", data=vinfo.unit)
    cmap = ''  if vinfo.colormap is None else vinfo.colormap
    hdf.create_dataset("ColorMap", data=cmap)
    hdf.close()

def save_as_nii(bbox, vinput, fname, bshape="rectangle"):
    # Don't save input as it's separated from GT in this system
    vinfo = bbox[0].voxinfo if isinstance(bbox, list) else bbox.voxinfo
    header = nib.Nifti1Header()
    header.set_xyzt_units(xyz=vinfo.unit_as_nifti)
    if isinstance(bbox, list): # Is already a one hot if only MV
        bbox = onehot2label(np.stack([b.voxel for b in bbox]))
    else:
        bbox = bbox.voxel
    gtimg = nib.Nifti1Image(bbox, vinfo.affine, header=header)
    fname = fname.with_stem(f"{fname.stem}-{bshape}").with_suffix(".nii")
    nib.save(gtimg, fname)

def save_as_nrrd(bbox, vinput, fname, bshape="rectangle"):
    # Don't save input as it's separated from GT in this system
    vinfo = bbox[0].voxinfo if isinstance(bbox, list) else bbox.voxinfo
    if isinstance(bbox, list): # Is already a one hot if only MV
        bbox = onehot2label(np.stack([b.voxel for b in bbox]))
    else:
        bbox = bbox.voxel
    header = {"type": "short",
              "dimension": bbox.ndim,
              "space": "right-anterior-superior", # RAS, hopefully
              "sizes": np.array(bbox.shape),
              "space directions": vinfo.directions,
              "kinds": ["domain", "domain", "domain"],
              "endian": "litle",
              "encoding": "gzip",
              "space origin": vinfo.origin}
    fname = fname.with_stem(f"{fname.stem}-{bshape}").with_suffix(".nrrd")
    # Use same type as when loading Philips
    nrrd.write(str(fname), bbox.astype(np.int16), header)


SAVERS = {"hdf": save_as_hdf, "nii": save_as_nii, "nrrd": save_as_nrrd}

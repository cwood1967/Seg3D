import numpy as np
import pandas as pd
import parse_roi

import tifffile
import scipy.ndimage as ndi
from matplotlib import pyplot as plt

def getstartstop(v, dv, size):
    v0 = max(v - dv, 0)
    vf = v + dv
    if vf > size:
        v0 = size - 2*dv
        vf = size

    return v0, vf
def rois_to_mask(image, roidf, dx=10, dy=10, dz=2):
    """
    
    Parameters
    ----------
    image : array
        A numpy array with (z, y, x)
    """
    mask = np.zeros_like(image)
    bbox = list()
    labels = list()
    #fgf
    for row in roidf.itertuples():
        x0, xf = getstartstop(row.x, dx, image.shape[1])
        y0, yf = getstartstop(row.y, dy, image.shape[0])
        z0, zf = getstartstop(row.z, dz, image.shape[2])
         
        mask[y0:yf, x0:xf, z0:zf] = 1
        bbox.append([y0, x0, yf, xf, z0, zf])
        labels.append(1)
    return {'seg':mask, 'bb_targets':bbox, 'roi_labels':labels} 

def rois_to_spheres(image, roidf, sigma):
    mask = np.zeros_like(image).astype(np.float32)
    
    for row in roidf.itertuples():
        mask[row.z, row.y, row.x] = 1
    mask = ndi.gaussian_filter(mask, sigma)
    mask = mask/mask.max()
    mask = np.where(mask > .1, 1.0, 0)
    return mask

roizip = '/Users/cjw/DropBox/Work/3D/A.zip'
imagefile = '/Users/cjw/DropBox/Work/3D/A.tif'
maskfile = '/Users/cjw/DropBox/Work/3D/Amask.tif'

image = tifffile.imread(imagefile)
print(image.shape)
roidf = parse_roi.points_to_df(roizip)
print(roidf.head())
mask = rois_to_spheres(image[:,0,:,:], roidf, (2, 3, 3))
tifffile.imwrite(maskfile, mask.astype(np.float32()))
print(mask.max(), mask.mean())
plt.imshow(mask.max(axis=0))
plt.show()
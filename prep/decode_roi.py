#%%
import  os
import numpy as np
from read_roi import read_roi_file, read_roi_zip
from matplotlib import pyplot as plt
from pims import ND2_Reader
import tifffile 
print(tifffile.__version__)
path = '/Volumes/core/micro/jeg/ac1692/jjl/20190419_3PO_IMARE-91237/deep learning'

files = sorted(os.listdir(path))

#%%
zips = [s for s in files if s.endswith('zip')]
rois = [s for s in files if s.endswith('roi')]
nd2s = [s for s in files if s.endswith('nd2')]
tifs = [s for s in files if s.endswith('tif')]

#%%
from skimage.draw import polygon
image = np.zeros((512, 512))
inum = 11
z = read_roi_zip(path + "/" + zips[inum])

nd = "-"
for a in nd2s:
    a1 = a.split('.')[0]
    print(a1, zips[inum])
    if a1 in zips[inum]:
        nd = path + "/" + a
        break

print(nd)
print(zips[inum])
with ND2_Reader(nd, channel=0) as frames:
    frames.iter_axes = 'z'
    for frame in frames:
        tb = frame.sum(axis=(0))
        print(frame.max(), frame.shape, tb.shape)

for roi in z.keys():
    c = z[roi]['x']
    r = z[roi]['y']
    rr, cc = polygon(r, c)
    image[rr, cc] = 255

plt.figure(figsize=(8,16))
plt.subplot(2,1,1)
plt.imshow(image)
plt.subplot(2,1,2)
plt.imshow(tb)
#%%

d = ND2_Reader(nd, channel=1)
d[0].sum(axis=0)

#%%






#%%


#%%


import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import tifffile

from models import unet3d
#from prep import target
from matplotlib import pyplot as plt

def random_slice(image, mask, size):
    '''
    
    image: array
        Normalized image with CZYX dims
    mask: array
        Normalizes image with ZYX dims
    '''
    
    nx = image.shape[-1]
    ny = image.shape[-2]
    nz = image.shape[1]
    nc = image.shape[0]
    
    px = size[2]
    py = size[1]
    pz = size[0]
    
    _x = np.random.randint(0, nx - px)
    _y = np.random.randint(0, ny - py)
    _z = np.random.randint(0, nz - pz)
    
    _bx = image[:,_z:_z+pz, _y:_y+py, _x:_x+px]
    _bx = _bx[np.newaxis]
    _bx = torch.from_numpy(_bx)
    _bm = mask[_z:_z+pz, _y:_y+py, _x:_x+px]
    _bm = _bm[np.newaxis]
    _bm = _bm[np.newaxis]
    _bm = torch.from_numpy(_bm) 
    return _bx, _bm  

def get_one_mem(bs, image, mask):
    _bx = image[:,6:14, 100:228, 200:328]
    _bx = _bx[np.newaxis]
    _bx = torch.from_numpy(_bx)
    _bm = mask[6:14, 100:228, 200:328]
    _bm = _bm[np.newaxis]
    _bm = _bm[np.newaxis]
    _bm = torch.from_numpy(_bm)
    
    return _bx, _bm
    
    
def get_batch(batchsize, image, mask):
    _xlist = list()
    _mlist = list()
    for i in range(batchsize):
        _x, _m = random_slice(image, mask, (8, 128, 128))
        _xlist.append(_x)
        _mlist.append(_m)
        
    return torch.cat(_xlist), torch.cat(_mlist)

def train():
    image = tifffile.imread('/Users/cjw/DropBox/Work/3D/A.tif')
    mask = tifffile.imread('/Users/cjw/DropBox/Work/3D/Amask.tif')
    x = np.moveaxis(image, 1, 0)
    x = x[1:,:,:,:]
    xmin = x.min(axis=(1,2,3), keepdims=True)
    xmax = x.max(axis=(1,2,3), keepdims=True)
    x = (x - xmin)/(xmax - xmin)
    x = x.astype(np.float32)
    mask = mask.astype(np.float32)
           
    net = unet3d(params={'nchannels':2})
    lr=2e-3
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8,
    #                           momentum=0.9)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                weight_decay=0.0005)
    
    criterion = nn.BCEWithLogitsLoss()
    
    for i in range(200):
        
        _bx, _bm = get_batch(12, x, mask)
        # _bx, _bm = get_one_mem(12, x, mask)
        # if i == 0:
        #     plt.imshow(_bx[0, 0,2,:,:])
        #     plt.show()  
        res = net(_bx)
        loss = criterion(res, _bm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(i, loss)
    
    # return net(torch.from_numpy(x[:, 5:13, 100:228, 100:228][np.newaxis]))
    return res #net(_bx)
def test():
    print(unet3d)
    net = unet3d(params={'nchannels':2})
    print(net)


res = torch.sigmoid(train()).detach().numpy()

plt.imshow(res[0,0,:,:,:].max(axis=0))
plt.show()
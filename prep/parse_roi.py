import  os

import numpy as np
import pandas as pd
from read_roi import read_roi_file, read_roi_zip


def points_to_df(roizip):
    rois = read_roi_zip(roizip)
    
    x = list()
    y = list()
    z = list()
    c = list()
    name = list()
    slice = list()
    
    for k, v in rois.items():
        if v['type'] != 'point':
            print(f"{k} is not a point")
            continue
        x.append(v['x'][0])        
        y.append(v['y'][0])        
        z.append(v['position']['slice'])
        c.append(v['position']['channel'])
        name.append(v['name'])        
        slice.append(v['slices'][0])
        
    df = pd.DataFrame({'name':name,
                       'y':y,
                       'x':x,
                       'z':z,
                       'c':c,
                       'slice':slice})
    
    return df        



import random
from numba import cuda
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numba as nb
from numba import cuda
import sys
import time




@nb.jit(fastmath=True,nopython=True)
def generate_dummy_mask(matrix:np.ndarray,threshold = 200):
    mask = np.copy(matrix)
    for row_id,row in enumerate(mask):
        for pixel_id,pixel in enumerate(row):
            mask[row_id][pixel_id] = pixel>threshold
    return mask


@nb.jit(parallel=True,fastmath = True,nopython = True)
def sequential_machine_code(pixelsMatrix,mask=None,grayLevels = 256) -> np.ndarray:
    H,W = pixelsMatrix.shape
    size = H*W
    occurencies = np.zeros(grayLevels,np.int32)

    for row_id,row in enumerate(pixelsMatrix):
        for pixel_id,pixel in enumerate(row):
            if mask is None or mask[row_id][pixel_id]==0:
                occurencies[pixel]+=1

    cdf = np.zeros(grayLevels,np.int32)
    for level in range(grayLevels):
        cdf[level] = sum(occurencies[:level+1])

    i = 0
    cdfmin = cdf[i]
    while cdfmin==0:
        i+=1
        cdfmin = cdf[i]
    h = np.zeros(grayLevels,np.int32)
    for v in range(grayLevels):
        h[v] = round((cdf[v]-cdfmin)/(size-cdfmin) * (grayLevels-1)) 


    for row_id,row in enumerate(pixelsMatrix):
        for pixel_id,pixel in enumerate(row):
            if mask is None or mask[row_id][pixel_id]==0:
                pixelsMatrix[row_id][pixel_id] = h[pixel]

    return pixelsMatrix




if __name__ == '__main__':
    times = []
    fnames = ['bigbig.png']
    for _ in range(10):
        rand_name = random.choice(fnames)
        fxd = np.array(Image.open(rand_name).convert('L'))
        mask = generate_dummy_mask(fxd)
        start= time.time()
        par = sequential_machine_code(fxd,mask,256)
        end = time.time()
        times.append((rand_name,round(end-start,3)))
    print(times)
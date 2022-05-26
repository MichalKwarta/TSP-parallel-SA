import sys
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numba as nb
from numba import cuda






def sequential(pixelMatrix,mask=None,grayLevels = 256) -> np.ndarray:
    H,W = pixelMatrix.shape
    assert mask is None or mask.shape==pixelMatrix.shape,"Invalid mask size"
    size = H*W
    occurencies = [0.0 for _ in range(grayLevels)]
    for row_id,row in enumerate(pixelMatrix):
        for pixel_id,pixel in enumerate(row):
            if mask is None or mask[row_id][pixel_id]==0:
                occurencies[pixel]+=1

    cdf = [int(sum(occurencies[:i+1])) for i in range(grayLevels)]
    cdfmin = next((x for x in cdf if x),-1)

    h = [round((cdf[v]-cdfmin)/(size-cdfmin) * (grayLevels-1)) for v in range(grayLevels)]

    for row_id,row in enumerate(pixelMatrix):
        for pixel_id,pixel in enumerate(row):
            if mask is None or mask[row_id][pixel_id]==0:
                pixelMatrix[row_id][pixel_id] = h[pixel]

    return pixelMatrix
    






@nb.jit(fastmath=True,nopython=True)
def generate_dummy_mask(matrix:np.ndarray,threshold = 200):
    mask = np.copy(matrix)
    for row_id,row in enumerate(mask):
        for pixel_id,pixel in enumerate(row):
            mask[row_id][pixel_id] = pixel>threshold
    return mask





if __name__ == '__main__':
    img = Image.open(sys.argv[1]).convert('L')
    pixelMatrix = np.array(img)
    mask = generate_dummy_mask(pixelMatrix)
    times = []
    for _ in range(10):
        start= time.time()
        par = sequential(pixelMatrix,mask)
        end = time.time()
        times.append(round(end-start,3))
    print(times)
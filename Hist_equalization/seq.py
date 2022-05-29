import time
from PIL import Image
import numpy as np
import numba as nb






def sequential(pixelsMatrix,mask,grayLevels = 256) -> np.ndarray:
    H,W = pixelsMatrix.shape
    assert  mask.shape==pixelsMatrix.shape,"Invalid mask size"
    size = H*W
    occurencies = [0.0 for _ in range(grayLevels)]
    for row_id,row in enumerate(pixelsMatrix):
        for pixel_id,pixel in enumerate(row):
            if  mask[row_id][pixel_id]==1:
                occurencies[pixel]+=1

    cdf = [int(sum(occurencies[:i+1])) for i in range(grayLevels)]
    cdfmin = next((x for x in cdf if x),-1)

    h = [round((cdf[v]-cdfmin)/(size-cdfmin) * (grayLevels-1)) for v in range(grayLevels)]

    for row_id,row in enumerate(pixelsMatrix):
        for pixel_id,pixel in enumerate(row):
            if  mask[row_id][pixel_id]==1:
                pixelsMatrix[row_id][pixel_id] = h[pixel]

    return pixelsMatrix

@nb.jit(fastmath=True,nopython=True)
def generate_dummy_mask(matrix:np.ndarray,threshold = 200):
    mask = np.copy(matrix)
    for row_id,row in enumerate(mask):
        for pixel_id,pixel in enumerate(row):
            mask[row_id][pixel_id] = pixel>threshold
    return mask




def benchmark(file,iters):
    times = []
    filename = np.array(Image.open(file).convert('L'))
    mask = generate_dummy_mask(filename)

    for _ in range(iters):
        start= time.time()
        seq = sequential(filename,mask)
        end = time.time()
        times.append(round(end-start,3))
    return times

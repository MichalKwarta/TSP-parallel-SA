import time
from PIL import Image
import numpy as np
import numba as nb



def sequential(pixelsMatrix,mask,grayLevels=256):
    occurencies = [0.0 for _ in range(grayLevels)]
    pixels_in_mask_count = 0
    for row_id,row in enumerate(pixelsMatrix):
        for pixel_id,pixel in enumerate(row):
            if  mask[row_id][pixel_id]!=0:
                occurencies[pixel]+=1 
                pixels_in_mask_count+=1 

    cdf = [sum(occurencies[:i+1]) for i in range(grayLevels)]
    cdfmin = next((x for x in cdf if x),-1) 
    h = [round((cdf[v]-cdfmin)/(pixels_in_mask_count-cdfmin) * (grayLevels-1)) for v in range(grayLevels)] 
    print(h)
    for row_id,row in enumerate(pixelsMatrix):
        for pixel_id,pixel in enumerate(row):
            if  mask[row_id][pixel_id] != 0: #piksel w masce nie jest czarny, wstawiam wartość z transformacji h
                pixelsMatrix[row_id][pixel_id] = h[pixel]
            else: #piksel jest czarny w masce, więc wstawiam czarny kolor 
                pixelsMatrix[row_id][pixel_id] = 0
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

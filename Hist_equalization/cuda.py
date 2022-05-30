from numba import cuda
from PIL import Image
import numpy as np
import numba as nb
from numba import cuda
import time




@nb.jit(fastmath=True,nopython=True)
def generate_dummy_mask(matrix:np.ndarray,threshold = 200):
    mask = np.copy(matrix)
    for row_id,row in enumerate(mask):
        for pixel_id,pixel in enumerate(row):
            mask[row_id][pixel_id] = pixel>threshold
    return mask


@cuda.jit(fastmath = True)
def countOccurencies(pixelsMatrix,occurencies,mask):
    x,y = cuda.grid(2)
    if x>=pixelsMatrix.shape[0] or y>=pixelsMatrix.shape[1]:
        return
    if mask[x][y] == 0:
        return
    cuda.atomic.add(occurencies,pixelsMatrix[x][y],1)

@cuda.jit(fastmath = True)
def calcCDF(occurencies_d,cdf): 
    acc = 0
    for i in range(cuda.grid(1)+1):
        acc+=occurencies_d[i]
    cdf[cuda.grid(1)] = acc

@cuda.jit(fastmath = True)
def calcH(h,cdf_d,cdfmin_d,size_d,grayLevels_d):
    nominator = (cdf_d[cuda.grid(1)]-nb.int32(cdfmin_d))

    denominator = (size_d-cdfmin_d)
    multiplier = (grayLevels_d-1)
    result = round(nominator/denominator*multiplier)

    h[cuda.grid(1)] = result




@cuda.jit(fastmath=True)
def changeOriginalValues(h_d,pixelsMatrix,mask):
    x,y = cuda.grid(2)
    if x>=pixelsMatrix.shape[0] or y>=pixelsMatrix.shape[1]:
        return
    if mask[x][y] != 0:
        pixelsMatrix[x][y] = nb.int32( h_d[pixelsMatrix[x][y]] )
    else:
        pixelsMatrix[x][y] = 0


@cuda.reduce
def sum_reduce(a,b):
    return a+b
            
        


def parallel(pixelsMatrix,mask,grayLevels = 256):
    H,W = pixelsMatrix.shape
    blockdim = (32, 32)
    mask_d = cuda.to_device(mask)

    griddim = (H // blockdim[0] + 1,W // blockdim[1] + 1)
    occurencies = np.zeros(grayLevels,np.int32)
    pixelsMatrix = pixelsMatrix.astype(np.int32)
    pixelsMatrix_d = cuda.to_device(pixelsMatrix)
    del pixelsMatrix
    occurencies_d = cuda.to_device(occurencies)
    countOccurencies[griddim,blockdim](pixelsMatrix_d,occurencies_d,mask_d)
    cdf = np.zeros(grayLevels,np.int32)
    cdf_d = cuda.to_device(cdf)
    threadsperblock = 32
    blockspergrid = (grayLevels + (threadsperblock - 1)) // threadsperblock
    calcCDF[threadsperblock,blockspergrid](occurencies_d,cdf_d)


    cdfmin = next((x for x in cdf_d if x),-1)

    h = np.zeros(grayLevels,np.int32)
    h_d = cuda.to_device(h)
    pixelsInMask =nb.int32(sum_reduce(occurencies_d))
    cuda.synchronize()
    calcH[threadsperblock,blockspergrid](h_d,cdf_d,nb.int32(cdfmin),pixelsInMask,nb.int32(grayLevels))
    del occurencies_d
    del cdfmin
    del cdf_d
    changeOriginalValues[griddim,blockdim](h_d,pixelsMatrix_d,mask_d)
    
    del h_d


    return pixelsMatrix_d



def benchmark(file,iters):
    times = []
    filename = np.array(Image.open(file).convert('L'))
    mask = generate_dummy_mask(filename)

    for _ in range(iters):

        start= time.time()
        par = parallel(filename,mask)
        end = time.time()
        times.append(round(end-start,3))
    return times

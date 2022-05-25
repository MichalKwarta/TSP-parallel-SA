from numba import cuda


@cuda.jit
def parallel(pixelsMatrix,matrix,occurencies,cdf,h,grayLevels):

    for i in range(0,len(pixelsMatrix[cuda.grid(1)])):
            matrix[cuda.grid(1)][pixelsMatrix[cuda.grid(1),i]]+=1
        
    
    cuda.syncthreads()
    width = len(pixelsMatrix[0])
    height = len(pixelsMatrix)
    for j in range(height):
        occurencies[cuda.grid(1)]+=matrix[j][cuda.grid(1)]

    cuda.syncthreads()

    if cuda.grid(1)<len(cdf):
        for cdf_iterator in range(cuda.grid(1)+1):
            cdf[cuda.grid(1)]+=occurencies[cdf_iterator] #CDF DZIWNIE LICZY

    cdfmin_id = 0
    while cdf[cdfmin_id]==0:
        cdfmin_id+=1
        
    cdfmin = cdf[cdfmin_id]    


    cuda.syncthreads()
    
    h[cuda.grid(1)] = round((cdf[cuda.grid(1)]-cdfmin)/(height*width-cdfmin) * (grayLevels-1))

    cuda.syncthreads()

    for col_id in range(width):
        pixelsMatrix[cuda.grid(1)][col_id]  = h[pixelsMatrix[cuda.grid(1)][col_id]]

    
        

    
    


threads = 256
threadsperblock = 32
blockspergrid = (threads + (threadsperblock - 1)) // threadsperblock

img = Image.open('baby.png').convert('L')
arr = np.array(img)
arr_d = cuda.to_device(arr)

W,H = img.size
grayLevels = 256
matrix = np.matrix([np.zeros(grayLevels,np.int32) for _ in range(threads)])
occurencies = np.zeros(grayLevels,np.int32)
cdf = np.zeros(grayLevels,np.int32)
h = np.zeros(grayLevels,np.int32)

img = parallel[threadsperblock,blockspergrid](arr_d,matrix,occurencies,cdf,h,grayLevels)
# parallel[blockspergrid,threadsperblock](arr_d)
# cuda.synchronize()
print(occurencies)
# Image.fromarray(np.array(arr_d))
# Image.fromarray(np.reshape(arr_d,(W,H)))







def sequential(path:str,grayLevels = 256) -> np.ndarray:
    img = Image.open(path).convert('L')
    H,W = img.size
    size = H*W
    occurencies = [0.0 for _ in range(grayLevels)]
    pixels_flattened = np.array(img).flatten()
    for pixel in pixels_flattened:
        occurencies[pixel]+=1
    cdf = [int(sum(occurencies[:i+1])) for i in range(grayLevels)]
    print(list(map(int,occurencies)))
    cdfmin = next((x for x in cdf if x),-1)
    h = [round((cdf[v]-cdfmin)/(size-cdfmin) * (grayLevels-1)) for v in range(grayLevels)]
    for id,pixel in enumerate(pixels_flattened):
        pixels_flattened[id] = h[pixel]
    return np.reshape(pixels_flattened,(W,H))
    
silence = sequential('baby.png')

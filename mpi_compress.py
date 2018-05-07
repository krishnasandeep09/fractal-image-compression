import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
from mpi4py import MPI
#from tempfile import TemporaryFile as TF
#from decompress import decompress
#from numba import jit
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule as SM

# Manipulate channels

def get_greyscale_image(img):
    return np.mean(img[:,:,:2], 2)


# Transforms

def reduce(img, factor):
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] += np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
    return result

def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)

def flip(img, direction):
    return img[::direction,:]

def apply_transform(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast*rotate(flip(img, direction), angle) + brightness

# Contrast and brightness

def find_contrast_and_brightness1(D, S):
    # Fix the contrast and only fit the brightness
    contrast = 0.75
    brightness = (np.sum(D - contrast*S)) / D.size
    return contrast, brightness 

def find_contrast_and_brightness2(D, S):
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
    return x[1], x[0]

# Compression for greyscale images

def generate_all_transformed_blocks(img, source_size, destination_size, step):
    factor = source_size // destination_size
    transformed_blocks = []
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            S = reduce(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append((k, l, direction, angle, apply_transform(S, direction, angle)))
    return transformed_blocks

def searchLoop(img, source_size, destination_size, step, rank, numRblocks, size):
    transforms = []
    #print(rank)
    count = -1
    transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
    for i in range((rank*numRblocks),((rank+1)*numRblocks)):
        transforms.append([])
        count = count + 1
        for j in range((img.shape[1] // destination_size)):
            transforms[count].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i*destination_size:(i+1)*destination_size,((j )*destination_size):((j + 1 )*destination_size)]
            # Test all possible transforms and take the best one

            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = find_contrast_and_brightness2(D, S)
                S = contrast*S + brightness
                d = np.sum(np.square(D - S))
                if d < min_d:
                    min_d = d
                    transforms[count][j] = (k, l, direction, angle, contrast, brightness)
    return transforms#, rank
def compress(img, source_size, destination_size, step):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    numRblocks = (img.shape[0]//destination_size)//size
    if(rank == 1):
        transforms1 = searchLoop(img, source_size, destination_size, step, rank, numRblocks, size)
        #transforms0
        trf = None
        comm.send(transforms1, dest=0, tag=10)
    if(rank == 0):
        transforms0 = searchLoop(img, source_size, destination_size, step, rank, numRblocks, size)
        #transforms0 = np.zeros_like(transforms1, dtype=float64)
        
        transforms1 = comm.recv(source=1, tag=10)
    comm.Barrier()
        
    #transforms = searchLoop(img, source_size, destination_size, step, rank, numRblocks, size)
    if(rank == 0):
        trf = transforms0+transforms1
        #print(trf[-1][-1])
    trf = comm.bcast(trf, root=0)
    comm.Barrier()
    #MPI.Finalize()
    return trf

def decompress(transforms, source_size, destination_size, step, nb_iter=8):
    factor = source_size // destination_size
    height = len(transforms) * destination_size
    width = len(transforms[0]) * destination_size
    #R = np.random.randint(0, 256, (height, width))
    R = np.ones((height, width))*150
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        #print(i_iter)
        for m in range(len(transforms)):
            for n in range(len(transforms)):
                # Apply transform
                #print(R.shape)
                k, l, flip, angle, contrast, brightness = transforms[m][n]
                S = reduce(R[k*step:k*step+source_size,l*step:l*step+source_size], factor)
                D = apply_transform(S, flip, angle, contrast, brightness)
#                print(m,n, transforms[m][n])
#                print(S)
#                print(R[k*step:k*step+source_size,l*step:l*step+source_size])
#                print(D)
                cur_img[m*destination_size:(m+1)*destination_size,n*destination_size:(n+1)*destination_size] = D
        R = cur_img
    return R









# Parameters

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = list(zip(directions, angles))




                    
if __name__ == '__main__':
    
    
    #img = misc.imread('monkey.gif')
    img = mpimg.imread('monkey.gif')
    #print(img.shape)
    img = get_greyscale_image(img)
    img = reduce(img, 4)
    
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    #MPI.Init
    transforms = compress(img, 8, 4, 8)
    MPI.Finalize()
    R = decompress(transforms, 8, 4, 8)
    plt.figure()
    plt.imshow(R, cmap='gray', interpolation='none')
    #plot_iterations(iterations, img)
    plt.show()
    

    

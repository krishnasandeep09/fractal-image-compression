# fractal-image-compression


This is mpi implementation of simple fractal image compression algorithm.

This currently works only for image sizes that are multiples of 64x64 such as 128x128, 256x256

Serial code was taken from https://github.com/pvigier/fractal-image-compression


How to run:

  mpiexec -n no_of_threads python mpi_crompress.py image_file_name

Requirements:

  python 3
  
  mpi4py
  
  numpy
  
  matplotlib
  
  scipy



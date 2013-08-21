CC = nvcc
INCLUDES = -I./include/
SOURCES = src/*.cu

CUDA_LIB = /usr/local/cuda/lib
CUDA_ARCH = -gencode=arch=compute_20,code=sm_20 \
			-gencode=arch=compute_30,code=sm_30
FLAGS = -O3 -v $(CUDA_ARCH) --compiler-options '-fPIC' --shared
LINK = -L$(CUDA_LIB) -lcublas
TARGET = libdecafcuda.so

USECUBLAS := 1

all: $(SOURCES)
	$(CC) $(INCLUDES) $(FLAGS) $(LINK) -o $(TARGET) $(SOURCES)

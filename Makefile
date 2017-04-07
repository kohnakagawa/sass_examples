NVCC = nvcc

PTX = smem.ptx cmem.ptx
CUBIN = smem.cubin cmem.cubin
SASS = smem.sass cmem.sass

TARGET = smem.out cmem.out $(PTX) $(CUBIN) $(SASS)

CUDA_HOME=/home/app/cuda/cuda-7.0
# CUDA_HOME=/usr/local/cuda

ARCH = -arch=sm_35
# ARCH = -arch=sm_60

OPT_FLAGS = -O3
# OPT_FLAGS = -O0 -g

NVCCFLAGS= $(OPT_FLAGS) -std=c++11 $(ARCH) -Xcompiler "-Wall -Wextra -Wunused-variable -Wsign-compare $(OPT_FLAGS)" -ccbin=g++ -Xptxas -v
INCLUDE= -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc

all: $(TARGET)

.SUFFIXES:
.SUFFIXES: .cu .cubin
.cu.cubin:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -cubin $< $(LIBRARY) -o $@

.SUFFIXES: .cubin .sass
.cubin.sass:
	cuobjdump -sass $< | c++filt > $@

.SUFFIXES: .cu .ptx
.cu.ptx:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

.SUFFIXES: .cu .out
.cu.out:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

clean:
	rm -f $(TARGET) *~

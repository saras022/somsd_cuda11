###############################################
     ###########################################
      #########################################
       #####                             #####
        ####   Makefile for the package: ####
        ####                             ####
        ####        -= GPU_FRNN =-       ####
       #####                             #####
      #########################################
     ###########################################
   ###############################################
#
# Modernized 2026: CUDA 8 -> CUDA 11.x, added OpenMP CPU path.
#
# Usage:
#   make                 Auto-detect CUDA; build GPU+CPU binary if nvcc found,
#                         otherwise build a CPU-only (OpenMP) binary.
#   make CPU_ONLY=1       Force a CPU-only (OpenMP) build even if nvcc is present.
#                         Useful for dev boxes without a GPU, or for comparing
#                         CPU vs GPU throughput.
#
# Runtime: use "-cpu <N>" on the somsd command line (or the OMP_NUM_THREADS
# environment variable) to control the number of CPU threads used by the
# OpenMP-parallel training path.

 #############
### Options ###
 #############
CC          = gcc
CXX         = g++
NVCC        = nvcc
CFLAGS      = -O3 -fomit-frame-pointer -Wall -D__INTERACTIVE__ -fopenmp
LDFLAGS     = -fopenmp -lpthread
# CUDA 11.x supports compute capability 3.5 (deprecated) through 8.6.
# sm_35/37/50/52 are deprecated-but-supported on CUDA 11.0-11.7; drop them if
# you only target Turing/Ampere+ and want faster nvcc builds.
NVARCH      = -gencode arch=compute_50,code=sm_50 \
              -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_86,code=compute_86
NVFLAGS     = -O3 -std=c++14 $(NVARCH) -Xcompiler -fopenmp
AR          = ar r
RANLIB      = ranlib

ifndef CPU_ONLY
HAS_CUDA := $(shell which nvcc 2>/dev/null)
CUDA_CHECK := $(notdir $(HAS_CUDA))
CUDA_HOME := $(shell dirname $(shell dirname $(HAS_CUDA)) 2>/dev/null)
else
CUDA_CHECK := disabled
endif

HAS_ZLIB := $(shell ls /usr/include/zlib.h 2>/dev/null)
ZLIB_CHECK := $(notdir $(HAS_ZLIB))

OBJS = common.o data.o fileio.o train.o train_cpu.o utils.o somsd.o testsom.o
GPU_OBJS = train_gpu.o cuda_utils.o


 ########################
### Check dependencies ###
 ########################
#check if Cuda is available (and not disabled via CPU_ONLY=1)
ifeq ($(CUDA_CHECK),nvcc)
CUDA_MSG="\n\e[32mGREAT:\e[0m System supports CUDA. Building with GPU support."
OPT = -D__USECUDA__
OBJS += $(GPU_OBJS)
LINKER = $(NVCC)
else
ifdef CPU_ONLY
CUDA_MSG="\n\e[36mCPU_ONLY=1:\e[0m Building CPU-only (OpenMP) binary as requested."
else
CUDA_MSG="\n\e[35mCUDA UNAVAILABLE:\e[0m Will compile without Cuda support (CPU/OpenMP only)."
endif
OPT =
LINKER = $(CXX)
endif

#check if NVML is available (via the CUDA toolkit's own headers - no more
#vendoring a private copy of nvml.h in this repo)
ifeq ($(OPT), -D__USECUDA__)
NVML_HEADER := $(wildcard $(CUDA_HOME)/include/nvml.h)
ifneq ("$(wildcard /usr/lib64/nvidia/libnvidia-ml.so)","")
ifneq ("$(NVML_HEADER)","")
NVML_MSG="\e[32mGREAT:\e[0m NVML available ($(NVML_HEADER))"
OPT += -D__USENVML__
LIBS += -lnvidia-ml
LPATH += -L/usr/lib64/nvidia/
CFLAGS += -I$(CUDA_HOME)/include
else
NVML_MSG="\e[35mWARNING:\e[0m NVML not available. Cannot find nvml.h under $(CUDA_HOME)/include. Autoselect on multi-gpu systems may not work without it."
endif
else
NVML_MSG="\e[35mWARNING:\e[0m NVML not available. Cannot find libnvidia-ml.so. Autoselect on multi-gpu systems may not work without it."
endif
else
NVML_MSG=""
endif

#Check if zlib is installed
ifneq ($(ZLIB_CHECK),zlib.h)
ZLIB_MSG="\e[35mWARNING:\e[0m No zlib detected!\n\t Go and install the zlib_devel package. Compiling without zlib support..."
else
ZLIB_MSG="\e[32mGREAT:\e[0m zlib support detected"
OPT += -D__USEZLIB__
LIBS += -lz
endif


 ####################
### Build sequence ###
 ####################
all: printInfo somsd

printInfo:
	@echo -e $(CUDA_MSG)
	@echo -e $(ZLIB_MSG)
	@echo -e -n $(NVML_MSG)
	@echo


%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $(CFLAGS) $(OPT) $<

train_gpu.o: train_gpu.cu
	$(NVCC) -c -o $@ $(NVFLAGS) $(OPT) $<

cuda_utils.o: cuda_utils.cu
	$(NVCC) -c -o $@ $(NVFLAGS) $(OPT) $<

somsd: $(OBJS)
	$(LINKER) -o $@ $(OBJS) $(LPATH) $(LIBS) $(LDFLAGS)

clean:
	rm -f *.o *.cu.o somsd

backup:
	tar -zcvf backup.tgz *.cu *.c *.h Makefile

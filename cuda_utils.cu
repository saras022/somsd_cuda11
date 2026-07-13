#include <stdio.h>
#include <string.h>
#include "common.h"
#include "cuda_utils.h"

#ifdef __USECUDA__
#include <cuda_runtime.h>

#ifdef __USENVML__
#include "nvml.h"
#endif

#define MAX_NUM_GPUs 64

typedef struct{
  int GPUmode;
  unsigned long long compute_perf;
  unsigned int usage;
} GPUs;

/*****************************************************************************
Function: 

Parameters: 

Return:void
******************************************************************************/
int GetNumGPUs()
{
  int num = 0;

  cudaGetDeviceCount(&num);
  return num;
}

/*****************************************************************************
Function: Print information provided by CUDA.

Parameters: Pointer to an output stream. If the pointer is NULL then output
          will be printed to stdout.

Return:void
******************************************************************************/
void PrintCUDAInformation(FILE *ofile)
{
  int n, i, num;
  struct cudaDeviceProp prop;

  if (ofile == NULL)
    ofile = stdout;

  cudaGetDevice(&n);
  num = GetNumGPUs();
  cudaGetDeviceProperties(&prop, n);

  i=3;
  fprintf(ofile, "Device: %d out of %d\n", n, num);
  fprintf(ofile, "Name: %s\n", prop.name);
  fprintf(ofile, "totalGlobalMem: %zu\n", prop.totalGlobalMem);
  fprintf(ofile, "sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock);
  fprintf(ofile, "regsPerBlock: %d\n", prop.regsPerBlock);
  fprintf(ofile, "warpSize: %d\n", prop.warpSize);
  fprintf(ofile, "memPitch: %zu\n", prop.memPitch);
  fprintf(ofile, "maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
  fprintf(ofile, "maxThreadsDim[3]: %d\n", prop.maxThreadsDim[i]);
  fprintf(ofile, "maxGridSize[3]: %d\n", prop.maxGridSize[i]);
  fprintf(ofile, "clockRate: %d\n", prop.clockRate);
  fprintf(ofile, "totalConstMem: %zu\n", prop.totalConstMem);
  fprintf(ofile, "major: %d\n", prop.major);
  fprintf(ofile, "minor: %d\n", prop.minor);
  fprintf(ofile, "textureAlignment: %d\n", prop.textureAlignment);
  fprintf(ofile, "texturePitchAlignment: %d\n", prop.texturePitchAlignment);
  fprintf(ofile, "deviceOverlap: %d\n", prop.deviceOverlap);
  fprintf(ofile, "multiProcessorCount: %d\n", prop.multiProcessorCount);
  fprintf(ofile, "kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);
  fprintf(ofile, "integrated: %d\n", prop.integrated);
  fprintf(ofile, "canMapHostMemory: %d\n", prop.canMapHostMemory);
  fprintf(ofile, "computeMode %d\n", prop.computeMode);
  fprintf(ofile, "maxTexture1D: %d\n", prop.maxTexture1D);
  fprintf(ofile, "maxTexture1DLinear: %d\n", prop.maxTexture1DLinear);
  fprintf(ofile, "maxTextureCubemap: %d\n", prop.maxTextureCubemap);
  fprintf(ofile, "maxSurface1D: %d\n", prop.maxSurface1D);
  fprintf(ofile, "maxSurfaceCubemap: %d\n", prop.maxSurfaceCubemap);
  fprintf(ofile, "surfaceAlignment: %d\n", prop.surfaceAlignment);
  fprintf(ofile, "concurrentKernels: %d\n", prop.concurrentKernels);
  fprintf(ofile, "ECCEnabled: %d\n", prop.ECCEnabled);
  fprintf(ofile, "pciBusID: %d\n", prop.pciBusID);
  fprintf(ofile, "pciDeviceID: %d\n", prop.pciDeviceID);
  fprintf(ofile, "pciDomainID: %d\n", prop.pciDomainID);
  fprintf(ofile, "tccDriver: %d\n", prop.tccDriver);
  fprintf(ofile, "asyncEngineCount: %d\n", prop.asyncEngineCount);
  fprintf(ofile, "unifiedAddressing: %d\n", prop.unifiedAddressing);
  fprintf(ofile, "memoryClockRate: %d\n", prop.memoryClockRate);
  fprintf(ofile, "memoryBusWidth: %d\n", prop.memoryBusWidth);
  fprintf(ofile, "l2CacheSize: %d\n", prop.l2CacheSize);
  fprintf(ofile, "maxThreadsPerMultiProcessor: %d\n",prop.maxThreadsPerMultiProcessor);
}

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

/*****************************************************************************
Function: 

Parameters: 

Return:void
******************************************************************************/
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
    {
      fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
      DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)          check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err){
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err));
    DEVICE_RESET
      exit(EXIT_FAILURE);
  }
}

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif



/*****************************************************************************
Function: 

Parameters: 

Return:void
******************************************************************************/
inline int _ConvertSMVer2Cores(int major, int minor)
{
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20B class
    { 0x60, 64 }, // Pascal Generation  (SM 6.0) GP100 class
    { 0x61, 128}, // Pascal Generation  (SM 6.1) GP10x class
    { 0x62, 128}, // Pascal Generation  (SM 6.2) GP10x class
    { 0x70, 64 }, // Volta Generation   (SM 7.0) GV100 class
    { 0x72, 64 }, // Volta Generation   (SM 7.2) GV10B class
    { 0x75, 64 }, // Turing Generation  (SM 7.5) TU10x class
    { 0x80, 64 }, // Ampere Generation  (SM 8.0) GA100 class
    { 0x86, 128}, // Ampere Generation  (SM 8.6) GA10x class
    {   -1, -1 }
  };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1){
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)){
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }

  // If we don't find the value then we default to latest
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
  return nGpuArchCoresPerSM[index-1].Cores;
}

/*****************************************************************************
Function: 

Parameters: 

Return:void
******************************************************************************/
int GetNumCores(int dev)
{
  int sm;

  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  sm = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

  return sm;
}


/*****************************************************************************
Function: General GPU Device CUDA Initialization

Parameters: 

Return:
******************************************************************************/
int gpuDeviceInit(int devID)
{
  int device_count;

  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0){
    fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0)
    devID = 0;

  if (devID > device_count-1){
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count);
    fprintf(stderr, ">> gpuDeviceInit (-gpu:%d) is not a valid GPU device. <<\n", devID);
    if (device_count > 0)
      fprintf(stderr, ">> Try (-gpu:0) <<\n", devID);

    fprintf(stderr, "\n");
    return -devID;
  }

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  if (deviceProp.computeMode == cudaComputeModeProhibited){
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    return -1;
  }

  if (deviceProp.major < 1){
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(devID));
  printf("Using CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

  return devID;
}

/*****************************************************************************
Function: This function returns the ID of the best GPU (with maximum GFLOPS)

Parameters: 

Return: -1 if no GPU is detected
        <num>: the ID of the fastest GPU
******************************************************************************/
int gpuGetMaxGflopsDeviceId(GPUs *gpus)
{
  int current_device     = 0, sm_per_multiproc  = 0;
  int max_perf_device    = -1;
  int device_count       = 0, best_SM_arch      = 0;
  int devices_prohibited = 0;
  cudaDeviceProp deviceProp;
  unsigned long long max_compute_perf = 0;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0)
    return -1;

  if (device_count >= MAX_NUM_GPUs){
    fprintf(stderr, "System with %d GPUs detected. Can only work with a the first %d GPUs\n", device_count, MAX_NUM_GPUs);
    device_count = MAX_NUM_GPUs;
  }
  
  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited
    if (deviceProp.computeMode != cudaComputeModeProhibited){
      if (deviceProp.major > 0 && deviceProp.major < 9999){
	best_SM_arch = MAX(best_SM_arch, deviceProp.major);
      }
    }
    else{
      devices_prohibited++;
    }
    current_device++;
  }

  if (devices_prohibited == device_count){
    fprintf(stderr, "CUDA error: all devices have compute mode prohibited.\n");
    return -1;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count){
    cudaGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
    if (deviceProp.computeMode != cudaComputeModeProhibited){
      if (deviceProp.major == 9999 && deviceProp.minor == 9999){
	sm_per_multiproc = 1;
      }
      else{
	sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
      }

      gpus[current_device].compute_perf  = (unsigned long long)deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

      if (gpus[current_device].compute_perf  > max_compute_perf){
	// If we find GPU with SM major > 2, search only these
	if (best_SM_arch > 2){
	  // If our device==dest_SM_arch, choose this, or else pass
	  if (deviceProp.major == best_SM_arch){
	    max_compute_perf  = gpus[current_device].compute_perf;
	    max_perf_device   = current_device;
	  }
	}
	else{
	  max_compute_perf  = gpus[current_device].compute_perf;
	  max_perf_device   = current_device;
	}
      }
    }
    ++current_device;
  }
  return max_perf_device;
}

 

/*****************************************************************************
Function: This function computes the GPU utilization rate of all GPUs

Parameters: 

Return:
******************************************************************************/
#ifdef __USENVML__
int gpuGetUtilizationNVML(GPUs *gpus)
{
  int current_device     = 0;
  int device_count       = 0;
  int result;
  nvmlDevice_t device;
  nvmlUtilization_t deviceUtil;

  checkCudaErrors(cudaGetDeviceCount(&device_count));
  if (device_count == 0)
    return 0;
  else if (device_count >= MAX_NUM_GPUs)
    device_count = MAX_NUM_GPUs;

  result = nvmlInit();
  if (NVML_SUCCESS != result)
    return -1;

  current_device = 0;
  while (current_device < device_count){
    nvmlDeviceGetHandleByIndex(current_device, &device);
    nvmlDeviceGetUtilizationRates(device, &deviceUtil);
    gpus[current_device].usage = (int)deviceUtil.gpu;
    ++current_device;
  }
  return 1;
}
#endif //NVML


/*****************************************************************************
Function: This function computes the GPU utilization rate of all GPUs

Parameters: 

Return:
******************************************************************************/
void gpuGetUtilizationSMI(GPUs *gpus)
{
  FILE *ifile;
  int currentDevice, res, usage;
  char buf[256];

  if (system("nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null >/dev/null") == 0){
    ifile = popen("nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader", "r");

    while(1){
      fgets(buf, 256, ifile);
      if (feof(ifile))
	break;
      res = sscanf(buf, "%d,%d", &currentDevice, &usage);
      if (res == 2)
	gpus[currentDevice].usage = usage;
    }
    pclose(ifile);
  } 
  else{
    fprintf(stderr, PRT_ERROR "Command 'nvidia-smi' not found.\n");
    fprintf(stderr, "       This is a multi-GPU system and this program cannot decide which GPU to\n       use without the command nvidia-smi. You will need to select the GPU\n       manually by using the command line option -gpu:<num>.\n");
    exit(EXIT_FAILURE);    
  }
}

/*****************************************************************************
Function: This function returns the ID of the best available GPU

Parameters: 

Return:
******************************************************************************/
int gpuGetFastestAvailableGPUDeviceId()
{
  int currentDevice     = 0;
  int max_perf_device    = 0;
  int device_count       = 0;
  unsigned long long max_compute_perf = 0, computePerf;
  GPUs gpus[MAX_NUM_GPUs] = {0};

  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  gpuGetMaxGflopsDeviceId(gpus);
#ifdef __USENVML__
  gpuGetUtilizationNVML(gpus);
#else
  gpuGetUtilizationSMI(gpus);
#endif

  for (currentDevice = 0; currentDevice < MAX_NUM_GPUs; currentDevice++){
    computePerf = (100-gpus[currentDevice].usage)*gpus[currentDevice].compute_perf;
    if (computePerf > 0)
      printf("Device %d: Speed: %ld GFLOPS, Usage: %d\\% -> %ld\n", currentDevice, gpus[currentDevice].compute_perf, gpus[currentDevice].usage, computePerf);
    
    if (computePerf >= max_compute_perf){
      max_compute_perf = computePerf;
      max_perf_device = currentDevice;
    }
  }
  if (max_compute_perf == 0){
    fprintf(stderr, PRT_WARNING "Failed to compute fastest available GPU. Will try GPU device 0.\n");
  }

  return max_perf_device;
}


/*****************************************************************************
Function: Initialization code to find the best CUDA Device

Parameters: 

Return:
******************************************************************************/
int findCudaDevice(int device)
{
  cudaDeviceProp deviceProp;
  int devID, sm;

  // If the command-line has a device number specified, use it
  if (device >= 0){
    if (gpuDeviceInit(device) < 0){
      fprintf(stderr, "exiting...\n");
      exit(EXIT_FAILURE);
    }
  }
  else{
    // Otherwise pick the device with highest Gflops/s
    if (GetNumGPUs() == 1)
      return 0;

    if (GetNumGPUs() > 1)
      printf("%d supported GPU devices detected.\n", GetNumGPUs());
    devID = gpuGetFastestAvailableGPUDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    sm = GetNumCores(devID);
    printf("Using GPU Device %d: \"%s\" with compute capability %d.%d (%d cores)\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor, sm);
    device = devID;
  }

  return device;
}

#endif  //__USECUDA__

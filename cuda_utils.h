//#ifndef __CUDA_UTILS_H__
//#define __CUDA_UTILS_H__


/**************/
/**PROTOTYPES**/
/**************/
int GetNumGPUs();
int GetNumCores(int dev);
void PrintCUDAInformation(FILE *ofile);
int findCudaDevice(int device);

//#endif

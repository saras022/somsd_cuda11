//#ifdef __USECUDA__
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "data.h"
#include "fileio.h"
#include "utils.h"
#include "train.h"
#include "testsom.h"
#include "train_gpu.h"
#include "cuda_utils.h"

#define GAUSS_ALPHA_CONSTANT 4.0
#define MAX_DATA_DIM 110

#define MAX_BLOCK_4MINMAX 1024
#define MAX_GRID_4MINMAX 1024

//New setting would not be tested
//#define BLOCK_4MINMAX 512
//#define BLOCK_FIX 512
//#define GRID_FIX 512

#define BLOCK_4MINMAX 128
#define BLOCK_FIX 128
#define GRID_FIX 128

#define DATA_DIM 100
#define NEIGHBOR_MAX_BOUND 50000000
#define NEIGHBOR_THRESHOLD 0.0 //0.00001

int grid_minmax;
extern char *fileresult;
cudaEvent_t gpu_s, gpu_e;

float *dev_codebook, *dev_sample;
int *dev_nodesample;
int *dev_coor_x, *dev_coor_y;
float *dev_diff, *dev_child_diff, *dev_learningrate, *dev_miu;
float *d_minmax;
int *d_minmax_pos;

int noc,data_dim, ssdim, num_samples, max_pnum; //number of codebook/neuron and dimension of codebook/sample
int map_xdim, map_ydim;
int fan_in, fan_out;

__device__ int winner_codeno;
__device__ float diffsf;

__device__ int neighbor_list[NEIGHBOR_MAX_BOUND];
__device__ float neighbor_func[NEIGHBOR_MAX_BOUND];
__device__ int noc_now;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
	    file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int GetUpperBoundPowerOfTwo(unsigned int n) {
  float p = ceil(std::log(n) / std::log(2));
  return static_cast<unsigned int>(std::pow(2, p));
}

__global__ void Min_Block_reduce_kernel(float* d_array, float *d_minmax, int *d_minmax_pos, int array_dim, int mode)
{
    __shared__ float shared[MAX_BLOCK_4MINMAX];
    __shared__ int shared_ind[MAX_BLOCK_4MINMAX];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = FLT_MAX;
    shared_ind[tid] = -1;
    //__syncthreads();

    //printf("INSIDE\n");
    //First time calling this function
    if (mode==1){
		while (gid < array_dim) {
			shared[tid] = d_array[gid];
			shared_ind[tid] = gid;
			gid += gridDim.x*blockDim.x;
		}
		__syncthreads();

    }

    //Second time calling this function
    if (mode==2){
     	while (gid < array_dim) {
    		shared[tid] =  d_minmax[gid];
    		shared_ind[tid] =  d_minmax_pos[gid];
    		gid += gridDim.x*blockDim.x;
    	}
        __syncthreads();
    }
    //printf("Shared_ind[%d] = %d\n", tid,shared_ind[tid]);

    gid = (blockDim.x * blockIdx.x) + tid;
    int bid = blockIdx.x;

	for (unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
		if (tid < s){
			if (shared[tid]>shared[tid + s]){
				shared[tid] = shared[tid + s];

				shared_ind[tid] = shared_ind[tid+s];
			}
		}
		__syncthreads();
	}

	if (tid == 0){//Each block retures a partial MIN/MAX
		d_minmax[bid]= shared[0];
		d_minmax_pos[bid] = shared_ind[0];
		//printf("Min pos = %d\n",shared_ind[0]);
	}
}

__global__ void Min_Thread_reduce_kernel(float *d_minmax, int *d_minmax_pos)
{
  __shared__ float shared[MAX_GRID_4MINMAX];
  __shared__ int shared_ind[MAX_GRID_4MINMAX];

  int tid = threadIdx.x;
  shared[tid] = 9999999;
  shared_ind[tid] = -1;

  //printf("INSIDE\n");
  //printf("d_minmax = %f\n",d_minmax[tid]);
  //printf("d_minmax_pos = %d\n",d_minmax_pos[tid]);

  shared[tid] =  d_minmax[tid];
  __syncthreads();
  shared_ind[tid] =  d_minmax_pos[tid];
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s){
	if (shared[tid]> shared[tid + s]){
	  shared[tid] = shared[tid + s];
	  shared_ind[tid] = shared_ind[tid+s];
	}
      }
      __syncthreads();
    }

  if (tid == 0){//MIN/MAX value is stored in the first element
    d_minmax[0]= shared[0];
    d_minmax_pos[0]= shared_ind[0];
    winner_codeno = shared_ind[0];


    //printf("Winner = %d - %f\n",shared_ind[0],shared[0]);
  }
}

template <unsigned int  blocksize>
__global__ void Add_reduce_kernel(float *d_array_in,float *d_array_out, int array_dim){
  extern __shared__ float shared[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blocksize*2)+tid;
  unsigned int gridsize = (blocksize*2)*gridDim.x;
  int s;

  while (i < array_dim) {
    shared[tid] += d_array_in[i]+d_array_in[i+blocksize];
    i += gridsize;
  }
  __syncthreads();

  for (s = blocksize/ 2; s > 32; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid < 32) {//We do not need to use __syncthreads if tid<warp size
    if (blocksize>=64) shared[tid] += shared[tid + 32];
    if (blocksize>=32) shared[tid] += shared[tid + 16];
    if (blocksize>=16) shared[tid] += shared[tid + 8];
    if (blocksize>=8) shared[tid] += shared[tid + 4];
    if (blocksize>=4) shared[tid] += shared[tid + 2];
    if (blocksize>=2) shared[tid] += shared[tid + 1];
  }

  if (tid==0){
    d_array_out[blockIdx.x] = shared[0];
    //printf("dev_diff[block:%d] = %f\n", blockIdx.x,dev_diff[blockIdx.x]);
  }
}
__global__ void Add_Block_reduce_kernel(float *g_odata, int array_dim)
{
  __shared__ float sdata[MAX_BLOCK_4MINMAX];
  int s;
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int blocksize = blockDim.x;

  while (gid < array_dim) {
    sdata[tid] = g_odata[gid];
    gid += gridDim.x*blockDim.x;
  }
  __syncthreads();

  for (s = blocksize/ 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32) {//We do not need to use __syncthreads if tid<warp size
    if (blocksize>=64) 	sdata[tid] += sdata[tid + 32];
    if (blocksize>=32) 	sdata[tid] += sdata[tid + 16];
    if (blocksize>=16)	sdata[tid] += sdata[tid + 8];
    if (blocksize>=8)	sdata[tid] += sdata[tid + 4];
    if (blocksize>=4)	sdata[tid] += sdata[tid + 2];
    if (blocksize>=2)	sdata[tid] += sdata[tid + 1];
  }

  if (tid == 0){//Each block retures a partial SUM
    g_odata[blockIdx.x] = sdata[0];
    //printf("Partial Sum = %f\n",sdata[0]);
  }
}

__global__ void Add_Thread_reduce_kernel(float *g_odata)
{
  int s;
  __shared__ float sdata[MAX_GRID_4MINMAX];
  unsigned int tid = threadIdx.x;
  int blocksize = blockDim.x;

  sdata[tid] = g_odata[tid];
  __syncthreads();

  for (s = blocksize/ 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32) {//We do not need to use __syncthreads if tid<warp size
    if (blocksize>=64) 	sdata[tid] += sdata[tid + 32];
    if (blocksize>=32) 	sdata[tid] += sdata[tid + 16];
    if (blocksize>=16)	sdata[tid] += sdata[tid + 8];
    if (blocksize>=8)	sdata[tid] += sdata[tid + 4];
    if (blocksize>=4)	sdata[tid] += sdata[tid + 2];
    if (blocksize>=2)	sdata[tid] += sdata[tid + 1];
  }

  if (tid == 0){//SUS is stored in the first element
    g_odata[0] = sdata[0];
    //printf("SUM = %f\n",sdata[0]);
  }

}

__global__ void PrintData_Kernel(float *sample, float *codebook, int noc, int data_dim, int num_samples,int *dev_coor_x, int *dev_coor_y){
  int i,j;
  //Printing the codebook
  printf("CODEBOOK...\n");
  for (i = 0; i < noc; i++){
    for (j = 0; j < data_dim; j++)
      printf("%0.4f ",codebook[i*data_dim+j]);
    printf("\n");
  }
  //Printing the data sample
  /*printf("SAMPLE...\n");
    for (i = 0; i < num_samples; i++){
    for (j = 0; j < data_dim; j++)
    printf("%0.4f ",sample[i*data_dim+j]);
    printf("\n");
    }*/

  for (i = 0; i < noc; i++) printf("X%d ",dev_coor_x[i]);
  printf("\n");
  for (i = 0; i < noc; i++) printf("Y%d ",dev_coor_y[i]);
}

void CopyData_H2D(struct Parameters *parameters){
  struct Map *map;
  struct Graph *gptr;
  struct Node *node;
  int i,j,k;

  map = &parameters->map;
  map_xdim = map->xdim;
  map_ydim = map->ydim;
  noc = map->xdim * map->ydim;
  data_dim = map->dim; // = codebook_dim

  fan_in = parameters->train->FanIn; //Pre-definded but normally not known yet
  fan_out = parameters->train->FanOut; //Pre-defined

  num_samples = 0;
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next)
    num_samples += gptr->numnodes;

  fprintf(stderr, "Start copying data from CPU to GPU ... \n");
  //fprintf(stderr, "OK1\n");

  HANDLE_ERROR(cudaMalloc((void**)&dev_codebook, noc * data_dim * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_sample, num_samples * data_dim * sizeof(float)));//SOM case or 1 graph case

  HANDLE_ERROR(cudaMalloc((void**)&dev_coor_x, noc * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_coor_y, noc * sizeof(int)));

  HANDLE_ERROR(cudaMalloc((void**)&dev_diff, noc * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_child_diff, data_dim * sizeof(float)));

  HANDLE_ERROR(cudaMalloc((void**)&dev_learningrate, noc * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_miu, data_dim * sizeof(float)));

  grid_minmax = GetUpperBoundPowerOfTwo((noc + BLOCK_4MINMAX - 1) / BLOCK_4MINMAX);
  HANDLE_ERROR(cudaMalloc(&d_minmax, grid_minmax * sizeof(float)));
  HANDLE_ERROR(cudaMalloc(&d_minmax_pos, grid_minmax * sizeof(int)));

  /*
  //OPTIM: Too many small transfer --> need banching all together
  for (i = 0; i < noc; i++){
  codebookIdx[i] = data_dim*i;
  HANDLE_ERROR(cudaMemcpy(dev_codebook+codebookIdx[i], map->codes[i].points, data_dim * sizeof(float), cudaMemcpyHostToDevice));
  }
  //Not working if there are more than 1 graphs

  //OPTIM: Too many small transfer --> need banching all together
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
  for (i = 0; i < gptr->numnodes; i++){
  sampleIdx[i]= data_dim*i;
  HANDLE_ERROR(cudaMemcpy(dev_sample+sampleIdx[i], gptr->nodes[i]->points, data_dim *sizeof(float), cudaMemcpyHostToDevice));
  }
  }
  */
  //Batch transferring data
  float *batch_codebook;
  float *batch_sample;
  int *batch_nodesample;

  batch_codebook =  (float *) malloc(noc * data_dim* sizeof(float));

  for (i = 0; i < noc; i++){  /* For every codebook of the map */
    for (j = 0; j< data_dim; j++)
      batch_codebook[i*data_dim + j] = map->codes[i].points[j];
  }

  //To store the g_nodeIdx of any nodes within any graph g_node[graphID][nodeID]
  int num_graphs=0;
  int max_num_nodes = 0;
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
    num_graphs++;
    max_num_nodes = max(max_num_nodes, gptr->numnodes);
  }
  int **g_node;
  g_node =  (int **) malloc(num_graphs * sizeof(int*));
  for (i = 0; i < num_graphs; i++) g_node[i] = (int *) calloc(max_num_nodes, sizeof(int));

  int gp_c = 0;
  int g_nodeIdx = 0;


  int pnum = 0;
  max_pnum = 0;

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
    for (i = 0; i < gptr->numnodes; i++){
      node = gptr->nodes[i];
      g_node[gp_c][node->nnum] =  g_nodeIdx;
      g_nodeIdx++;

      //This will be used for the PMGraphSOM case
      pnum = node->numparents;
      if (pnum>max_pnum)
	max_pnum = pnum;
    }
    gp_c++;
  }
  //NOTE: should check if the g_node[][] is in incremental order, say with reversed node id order

  fan_in = max_pnum;
  ssdim =  (2+fan_in+fan_out);
  /*
    The first 2 points are winning coors of corresponding node,
    all following points are all g_nodeIdx of children/parent nodes
  */
  HANDLE_ERROR(cudaMalloc((void**)&dev_nodesample, num_samples * ssdim * sizeof(int)));
  batch_nodesample =  (int *) malloc(num_samples * ssdim * sizeof(int));
  batch_sample =  (float *) malloc(num_samples * data_dim* sizeof(float));
  //In the SOMSD and more, data_dim includes data label dim + all offset of children/parent winning coordinates or so

  gp_c = 0;
  g_nodeIdx = 0;
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
    for (i = 0; i < gptr->numnodes; i++){
      node = gptr->nodes[i];

      //if (map->type == TYPE_SOMSD){
      if (map->type != TYPE_SOM){
	//Contains two winning coors
	batch_nodesample[g_nodeIdx*ssdim] = node->x;
	batch_nodesample[g_nodeIdx*ssdim+1] = node->y;

	for (k = 0; k < fan_out; k++){
	  if (node->children[k] != NULL){
	    batch_nodesample[g_nodeIdx*ssdim+2+k] = g_node[gp_c][node->children[k]->nnum];
	    //printf("child %d ",g_node[gp_c][node->children[k]->nnum]);
	  }
	  else batch_nodesample[g_nodeIdx*ssdim+2+k] = -1;
	}
	for (k = 0; k < node->numparents; k++){
	  if (node->parents[k] != NULL){
	    batch_nodesample[g_nodeIdx*ssdim+2+fan_out+k] = g_node[gp_c][node->parents[k]->nnum];
	    //printf("parents %d ",g_node[gp_c][node->parents[k]->nnum]);
	  }
	  else batch_nodesample[g_nodeIdx*ssdim+2+fan_out+k] = -1;
	}
	//printf("\n");
      }

      for (j = 0; j< data_dim; j++)
	batch_sample[g_nodeIdx*data_dim + j] =  gptr->nodes[i]->points[j];

      g_nodeIdx++;
    }
    gp_c++;
  }

  //printf("COPYING CPU->GPU\n");

  HANDLE_ERROR(cudaMemcpy(dev_codebook, batch_codebook, noc*data_dim * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_sample, batch_sample, num_samples*data_dim *sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_nodesample, batch_nodesample, num_samples*ssdim *sizeof(int), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemcpy(dev_miu, parameters->train->nodes[0]->mu,  data_dim*sizeof(int), cudaMemcpyHostToDevice));
  //for (i=0;i<data_dim;i++) printf("%f:", parameters->train->nodes[0]->mu[i]);
  //-->Need to set value for both mu1 and mu2 in terminal command

  free(batch_codebook);
  free(batch_sample);
  free(batch_nodesample);

  for (i = 0; i < num_graphs; i++) free(g_node[i]);
  free(g_node);

  printf("\nFinished COPYING CPU->GPU\n");
}

void CopyTestDataSample_H2D(struct Parameters *parameters, struct Graph *data)
{
  struct Map *map;
  struct Graph *gptr;
  //  struct Node *node;
  int i,j;

  map = &parameters->map;
  map_xdim = map->xdim;
  map_ydim = map->ydim;
  noc = map->xdim * map->ydim;
  data_dim = map->dim; // = codebook_dim

  fan_in = data->FanIn; //Pre-definded but normally not known yet
  fan_out = data->FanOut; //Pre-defined

  num_samples = 0;
  for (gptr = data; gptr != NULL; gptr = gptr->next)
    num_samples += gptr->numnodes;

  fprintf(stderr, "Start copying Test data sample from CPU to GPU ... \n");
  //fprintf(stderr, "OK1\n");

  float *batch_sample;
  batch_sample =  (float *) malloc(num_samples * data_dim* sizeof(float));

  int gp_c = 0;
  int g_nodeIdx = 0;
  for (gptr = data; gptr != NULL; gptr = gptr->next){
    for (i = 0; i < gptr->numnodes; i++){

      for (j = 0; j< data_dim; j++)
	batch_sample[g_nodeIdx*data_dim + j] =  gptr->nodes[i]->points[j];

      g_nodeIdx++;
    }
    gp_c++;
  }

  //printf("COPYING CPU->GPU\n");

  HANDLE_ERROR(cudaMemcpy(dev_sample, batch_sample, num_samples*data_dim *sizeof(float), cudaMemcpyHostToDevice));
  free(batch_sample);

}


void CopyData_D2H(struct Parameters *parameters, Winner *winner){
  int i,j;

  struct Map *map;
  map = &parameters->map;

  fprintf(stderr, "Start copying data from GPU back to CPU ...\n ");

  //Batch transferring data
  float *batch_codebook;
  batch_codebook =  (float *) malloc(noc * data_dim* sizeof(float));

  HANDLE_ERROR(cudaMemcpy(batch_codebook, dev_codebook, noc*data_dim * sizeof(float), cudaMemcpyDeviceToHost));

  for (i = 0; i < noc; i++){  /* For every codebook of the map */
    for (j = 0; j< data_dim; j++)
      map->codes[i].points[j] = batch_codebook[i*data_dim + j];
  }

  free(batch_codebook);

  //HANDLE_ERROR(cudaMemcpy(&winner->codeno, &d_minmax_pos[0], sizeof(int), cudaMemcpyDeviceToHost));
  fprintf(stderr, "END\n");
}

void FreeCuda(){
  cudaFree(dev_codebook);
  cudaFree(dev_sample);
  cudaFree(dev_nodesample);

  cudaFree(dev_coor_x);
  cudaFree(dev_coor_y);
  cudaFree(dev_diff);
  cudaFree(dev_learningrate);
  cudaFree(dev_miu);
  cudaFree(d_minmax);
  cudaFree(d_minmax_pos);

}

/* Use when the ydim<1024
 * Need  to declare: dim3 grids(xdim,1,1);
 * 					 dim3 blocks(ydim,1,1);
 */

__global__ void Gpu_Init_diffsf(){
  diffsf = FLT_MAX;
}

__global__ void Gpu_Print_Test(){
  printf("Winner = %d - %f\n", winner_codeno, diffsf);
}

__global__ void Gpu_Init_Variables(int *dev_coor_x, int *dev_coor_y){
  unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  dev_coor_x[tid] = threadIdx.x;
  dev_coor_y[tid] = blockIdx.x;
}

/* Use when the ydim>1024
 * Need  to declare: dim3 grids(xdim,ydim,1);
 * 					 dim3 blocks(1,1,1);
 */
__global__ void Gpu_Init_Variables_2(int *dev_coor_x, int *dev_coor_y){
  unsigned int tidx = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int tidy = threadIdx.y+blockIdx.y*blockDim.y;
  unsigned int offset = tidx + tidy * blockDim.x * gridDim.x;
  dev_coor_x[offset] = blockIdx.x;
  dev_coor_y[offset] = blockIdx.y;
}

__global__ void FindWinnerEucledian_Child_Kernel(int nodeIdx, float* mu, float* dev_child_diff,
						 float* dev_codebook, float* dev_sample, int data_dim){
  //	int s;
  __shared__ float sdiff[MAX_DATA_DIM];//If put fixed size, then need to care about the sdiff range over tid

  unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

  while (tid < data_dim) {
    sdiff[tid] = dev_codebook[data_dim*blockIdx.x+tid] - dev_sample[data_dim*nodeIdx+tid];
    sdiff[tid] *= sdiff[tid]*mu[tid];

    tid += blockDim.x * gridDim.x;
  }
  __syncthreads();

  dev_child_diff[tid] = sdiff[tid];
}

__global__ void FindWinnerEucledian_Parent_Kernel(int nodeIdx, float* mu, float* dev_diff, float* dev_child_diff,
						  float* dev_codebook, float* dev_sample, int data_dim, int noc){

  //float sdiff[MAX_DATA_DIM];//If put fixed size, then need to care about the sdiff range over tid
  //  floaf sdiff_sum;
  unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
  //printf("Inside KERNEL\n");
  //	int gridsi;

  while (tid < noc) {
    //	sdiff_sum = 0;

    /*#pragma unroll
      for (i=0;i<data_dim;i++){
      sdiff = dev_codebook[data_dim*tid+i] - dev_sample[data_dim*nodeIdx+i];
      sdiff *= sdiff*mu[i];
      sdiff_sum += sdiff;
      }*/

    //Invoking the Dynamic parallel programming here

    //FindWinnerEucledian_Child_Kernel<<<1,128>>>(nodeIdx, mu, dev_child_diff, dev_codebook, dev_sample, data_dim);
    //Add_Block_reduce_kernel<<<1, 128>>>(dev_child_diff,data_dim);
    //Reduction applied on a single block
    //Add_Thread_reduce_kernel<<<1, data_dim>>>(dev_child_diff);

    dev_diff[tid] = dev_child_diff[0];

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void List_Fixed_Neighboor_kernel(int nodex, int nodey, int radius_fix, int xdim, int ydim){
  //	int dx,dy;
  noc_now = 0;
  //unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;

  //BlockDim = 2*radius
  //GridDim = 2*radius
  //Consider the last winner codeno of the working node
  int radius  = radius_fix;

  int offsetx = threadIdx.x - radius;
  int offsety = blockIdx.x - radius;

  //printf("Center X Y: %d %d\n", x,y);
  int dxx = offsetx+nodex;
  int dyy = offsety+nodey;

  if (dxx >= 0 && dxx < xdim){
    if (dyy >= 0 && dyy < ydim){
      int count = atomicAdd(&noc_now,1);
      neighbor_list[count] = dxx + dyy*xdim;
      //printf("%d -- ", neighbor_list[count]);
    }
  }
  //printf("\n");
  //printf("%d \n", noc_now);
}

__global__ void FindWinnerEucledian_Neighbor_Kernel(int nodeIdx, float* mu, float* dev_diff,
						    float* dev_codebook, float* dev_sample, int data_dim){
  int i;

  //float sdiff[MAX_DATA_DIM];//If put fixed size, then need to care about the sdiff range over tid
  float sdiff, sdiff_sum;
  unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

  while (tid < noc_now) {
    sdiff_sum = 0;

    //#pragma unroll: Use to store index, variables in registers, for speeding up
    //But here this does not help since data_dim (also tid, nodeIdx) is not known at compiling time
    int id1 = data_dim* neighbor_list[tid];
    int id2 = data_dim*nodeIdx;

#pragma unroll 32
    for (i=0;i<data_dim;i++){
      sdiff = dev_codebook[id1+i] - dev_sample[id2+i];
      sdiff *= sdiff*mu[i];
      sdiff_sum += sdiff;
    }

    dev_diff[tid] = sdiff_sum;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void FindWinnerEucledian_Kernel(int nodeIdx, float* mu, float* dev_diff,
					   float* dev_codebook, float* dev_sample, int data_dim, int noc){
  int i;

  //float sdiff[MAX_DATA_DIM];//If put fixed size, then need to care about the sdiff range over tid
  float sdiff, sdiff_sum;

  unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
  //printf("Inside KERNEL\n");

  while (tid < noc) {
    sdiff_sum = 0;

    //#pragma unroll: Use to store index, variables in registers, for speeding up
    //But here this does not help since data_dim (also tid, nodeIdx) is not known at compiling time

    int id1 = data_dim*tid;
    int id2 = data_dim*nodeIdx;

#pragma unroll 32
    for (i=0;i<data_dim;i++){
      sdiff = dev_codebook[id1+i] - dev_sample[id2+i];
      sdiff *= sdiff*mu[i];
      sdiff_sum += sdiff;
    }

    dev_diff[tid] = sdiff_sum;

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void FindWinnerEucledian_Kernel0(int nodeIdx, float* mu, float* dev_diff,
					    float* dev_codebook, float* dev_sample, int data_dim){
  int s;
  __shared__ float sdiff[MAX_DATA_DIM];//If put fixed size, then need to care about the sdiff range over tid
  unsigned int tid = threadIdx.x;
  int blocksize = blockDim.x;
  //unsigned int n = threadIdx.x + blockIdx.x*blockDim.x;
  //printf("Inside KERNEL\n");
  sdiff[tid]=0;

  if (tid < data_dim)
    sdiff[tid] = dev_codebook[data_dim*blockIdx.x+tid] - dev_sample[data_dim*nodeIdx+tid];
  __syncthreads();

  if (tid < data_dim)
    sdiff[tid] *= sdiff[tid];
  __syncthreads();

  if (tid < data_dim)
    sdiff[tid] *= mu[tid];
  __syncthreads();

  //printf("sdiff[thread:%d] = %f\n", tid, sdiff[tid]);
  //Sum of all elements: the number of elements is no more than blockDim (or num of threads)
  //Implement reduction 2, blockDim.x has already set to a power of 2
  for (s = blocksize/ 2; s > 32; s >>= 1) {
    if (tid < s)
      sdiff[tid] += sdiff[tid + s];
    __syncthreads();
  }

  if (tid < 32) {//We do not need to use __syncthreads if tid<warp size
    if (blocksize>=64) 	sdiff[tid] += sdiff[tid + 32];
    if (blocksize>=32) 	sdiff[tid] += sdiff[tid + 16];
    if (blocksize>=16)	sdiff[tid] += sdiff[tid + 8];
    if (blocksize>=8)	sdiff[tid] += sdiff[tid + 4];
    if (blocksize>=4)	sdiff[tid] += sdiff[tid + 2];
    if (blocksize>=2)	sdiff[tid] += sdiff[tid + 1];
  }

  if (tid==0){
    dev_diff[blockIdx.x] = sdiff[0];
    //printf("dev_diff[block:%d] = %f\n", blockIdx.x,dev_diff[blockIdx.x]);
  }

}

__global__ void FindWinnerEucledian_Kernel_LargeDataDim(int nodeIdx, float* mu, float* dev_diff,
							float* dev_codebook, float* dev_sample, int data_dim, int noc){
  __shared__ float sdiff[MAX_DATA_DIM];//If put fixed size, then need to care about the sdiff range over tid
  float temp;
  unsigned int tid = threadIdx.x;
  int blocksize = blockDim.x;
  unsigned int i,s;

  unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int bid = blockIdx.x;
  //printf("Inside KERNEL\n");
  sdiff[tid]=0;
  while (gid < noc) {

    for (i = tid; i < data_dim; i+=blocksize) {
      temp = dev_codebook[data_dim*blockIdx.x+i] - dev_sample[data_dim*nodeIdx+i];
      temp *= temp;
      temp *= mu[i];

      sdiff[tid] += temp;
      __syncthreads();
    }

    for (s = blocksize/ 2; s > 32; s >>= 1) {
      if (tid < s)
	sdiff[tid] += sdiff[tid + s];
      __syncthreads();
    }

    if (tid < 32) {//We do not need to use __syncthreads if tid<warp size
      if (blocksize>=64) 	sdiff[tid] += sdiff[tid + 32];
      if (blocksize>=32) 	sdiff[tid] += sdiff[tid + 16];
      if (blocksize>=16)	sdiff[tid] += sdiff[tid + 8];
      if (blocksize>=8)	sdiff[tid] += sdiff[tid + 4];
      if (blocksize>=4)	sdiff[tid] += sdiff[tid + 2];
      if (blocksize>=2)	sdiff[tid] += sdiff[tid + 1];
    }

    if (tid==0){
      dev_diff[bid] = sdiff[0];
      //printf("dev_diff[block:%d] = %f\n", blockIdx.x,dev_diff[blockIdx.x]);
    }

    gid += blockDim.x * gridDim.x;
    //bid += gridDim.x;
  }
}
/******************************************************************************
 Description: Find best matching codebook using the Eucledian distance meassure.

 Return value: The best matching codebook is returned to parameter "winner".
******************************************************************************/
void FindWinnerEucledian_GPU(int nodeIdx, struct Winner *winner) {
  //fprintf(stderr, "Node ID: %d\n", nodeIdx);

  //Using Fixed radius neighboor for finding the winning codebook
  //int radiusf = 10;
  //List_Fixed_Neighboor_kernel<<<radiusf*2,radiusf*2>>>(node->x, node->y, radiusf, map_xdim, map_ydim);

  //OPTIM: vary grids and blocks to get the optimum
  //int gridofblocks = noc;
  //int blockofthreads = GetUpperBoundPowerOfTwo(data_dim);
  //FindWinnerEucledian_Kernel0<<<gridofblocks,blockofthreads>>>(nodeIdx, dev_miu, dev_diff, dev_codebook, dev_sample, data_dim);

  //FindWinnerEucledian_Kernel_LargeDataDim<<<gridofblocks,blockofthreads>>>(nodeIdx, dev_miu, dev_diff, dev_codebook, dev_sample, data_dim,noc);

  //The most accurate
  FindWinnerEucledian_Kernel<<<GRID_FIX,BLOCK_FIX>>>(nodeIdx, dev_miu, dev_diff, dev_codebook, dev_sample, data_dim,noc);
  //FindWinnerEucledian_Neighbor_Kernel<<<GRID_FIX,BLOCK_FIX>>>(nodeIdx, dev_miu, dev_diff, dev_codebook, dev_sample, data_dim);

  grid_minmax = GetUpperBoundPowerOfTwo((noc + BLOCK_4MINMAX - 1) / BLOCK_4MINMAX);
  Min_Block_reduce_kernel<<<grid_minmax, BLOCK_4MINMAX>>>(dev_diff,d_minmax,d_minmax_pos,noc,1);
  while (grid_minmax>1024){
    int grid_minmax_new = GetUpperBoundPowerOfTwo((grid_minmax + BLOCK_4MINMAX - 1) / BLOCK_4MINMAX);
    Min_Block_reduce_kernel<<<grid_minmax_new, BLOCK_4MINMAX>>>(d_minmax,d_minmax,d_minmax_pos,grid_minmax,2);
    grid_minmax = grid_minmax_new;
  }

  Min_Thread_reduce_kernel<<<1, grid_minmax>>>(d_minmax,d_minmax_pos);

  HANDLE_ERROR(cudaMemcpy(&winner->diff, &d_minmax[0], sizeof(float), cudaMemcpyDeviceToHost));
  /*/Only for checking correctness
    HANDLE_ERROR(cudaMemcpy(&winner->diff, &d_minmax[0], sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&winner->codeno, &d_minmax_pos[0], sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&node->x, &dev_coor_x[d_minmax_pos[0]], sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&node->y, &dev_coor_y[d_minmax_pos[0]], sizeof(int), cudaMemcpyDeviceToHost));
    //fprintf(stderr, "Winner %d - %f\n", winner->codeno, winner->diff);
    */
      }

void FindWinnerEucledian_GPU1(int nodeIdx, struct Node *node, struct Winner *winner) {
  //int nodeIdx= node->nnum;
  //fprintf(stderr, "Node ID: %d\n", nodeIdx);

  //OPTIM: vary grids and blocks to get the optimum
  //int gridofblocks = noc;
  //int blockofthreads = GetUpperBoundPowerOfTwo(data_dim);

  FindWinnerEucledian_Kernel<<<GRID_FIX,BLOCK_FIX>>>(nodeIdx, dev_miu, dev_diff, dev_codebook, dev_sample, data_dim,noc);
  //FindWinnerEucledian_Kernel0<<<gridofblocks,blockofthreads>>>(nodeIdx, dev_miu, dev_diff, dev_codebook, dev_sample, data_dim);

  grid_minmax = GetUpperBoundPowerOfTwo((noc + BLOCK_4MINMAX - 1) / BLOCK_4MINMAX);
  Min_Block_reduce_kernel<<<grid_minmax, BLOCK_4MINMAX>>>(dev_diff,d_minmax,d_minmax_pos,noc,1);
  while (grid_minmax>1024){
    int grid_minmax_new = GetUpperBoundPowerOfTwo((grid_minmax + BLOCK_4MINMAX - 1) / BLOCK_4MINMAX);
    Min_Block_reduce_kernel<<<grid_minmax_new, BLOCK_4MINMAX>>>(d_minmax,d_minmax,d_minmax_pos,grid_minmax,2);
    grid_minmax = grid_minmax_new;
  }

  Min_Thread_reduce_kernel<<<1, grid_minmax>>>(d_minmax,d_minmax_pos);

  //Only for the last iteration to copy data to file
  HANDLE_ERROR(cudaMemcpy(&winner->diff, &d_minmax[0], sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&winner->codeno, &d_minmax_pos[0], sizeof(int), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&node->x, &dev_coor_x[winner->codeno], sizeof(int), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&node->y, &dev_coor_y[winner->codeno], sizeof(int), cudaMemcpyDeviceToHost));
  //fprintf(stderr, "Winner %d - %f\n", winner->codeno, winner->diff);

}
__global__ void LearningRate_Kernel(int nodeIdx, float *learningrate, float alpha, float radius,
				    int* dev_coor_x, int* dev_coor_y, int noc) {

  /*
    The number of threads is set to be equal to the number of neuron = noc
    Each thread compute a distance from the winner_codeno pos to a single neuron on the map
  */
  //unsigned int tidx = threadIdx.x+blockIdx.x*blockDim.x;
  //unsigned int tidy = threadIdx.y+blockIdx.y*blockDim.y;
  //unsigned int tid = tidx + tidy * blockDim.x * gridDim.x;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < noc) {


    //if (blockIdx.x==0) printf("Winner_codeno = %d\n",winner_codeno);

    int bx= dev_coor_x[winner_codeno];
    int by= dev_coor_y[winner_codeno];
    int tx = dev_coor_x[tid];
    int ty = dev_coor_y[tid];

    //printf("bx,bt,tx,ty = %d %d %d %d\n", bx,by,tx,ty);

    //Compute Hexa distance

    //OPTIM: reduce multiple access to global variables
    int  dx;
    float  dy;

    dx= bx - tx;
    dy = by - ty;

    if (dx & 1) {
      if (tx & 1)
	dy += 0.5;
      else
	dy -= 0.5;
    }

    float hexa_distance = (float)(dy*dy + 0.75*dx*dx);
    //printf("hexa_distance: %f\n",hexa_distance);

    //OPTIM: Using fast math runtine version
    learningrate[tid] = alpha* __expf((hexa_distance/(-2.0 * radius)));
    //printf("Alpha: %f\n",alpha1);

    tid += blockDim.x * gridDim.x;
  }

}
__global__ void LearningRate_Kernel0(int nodeIdx, float *learningrate, float alpha, float radius,
				     int* dev_coor_x, int* dev_coor_y) {

  /*
    The number of threads is set to be equal to the number of neuron = noc
    Each thread compute a distance from the winner_codeno pos to a single neuron on the map
  */
  unsigned int tidx = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int tidy = threadIdx.y+blockIdx.y*blockDim.y;
  unsigned int tid = tidx + tidy * blockDim.x * gridDim.x;
  //if (blockIdx.x==0) printf("Winner_codeno = %d\n",winner_codeno);

  int bx= dev_coor_x[winner_codeno];
  int by= dev_coor_y[winner_codeno];
  int tx = dev_coor_x[tid];
  int ty = dev_coor_y[tid];

  //printf("bx,bt,tx,ty = %d %d %d %d\n", bx,by,tx,ty);

  //Compute Hexa distance

  //OPTIM: reduce multiple access to global variables
  int  dx;
  float  dy;

  dx= bx - tx;
  dy = by - ty;

  if (dx & 1) {
    if (tx & 1)
      dy += 0.5;
    else
      dy -= 0.5;
  }

  float hexa_distance = (float)(dy*dy + 0.75*dx*dx);
  //printf("hexa_distance: %f\n",hexa_distance);

  //OPTIM: Using fast math runtine version
  learningrate[tid] = alpha* __expf((hexa_distance/(-2.0 * radius)));
  //printf("Alpha: %f\n",alpha1);
}

/*
  List all the neighboring nodes that locate within the neighborhood region, that means
  the nodes associated with the non-rezo (or larger than a threshold value) neighborhood function results
  Purpose: only working on a decreasing small region in updating stage: resulting in speed-up
*/

__global__ void List_Neighboor_kernel(float radius, int* dev_coor_x, int* dev_coor_y, int noc) {

  /*
    The number of threads is set to be equal to the number of neuron = noc
    Each thread compute a distance from the winner_codeno pos to a single neuron on the map
  */
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  noc_now = 0;
  while (tid < noc) {

    int bx= dev_coor_x[winner_codeno];
    int by= dev_coor_y[winner_codeno];
    int tx = dev_coor_x[tid];
    int ty = dev_coor_y[tid];

    //printf("bx,bt,tx,ty = %d %d %d %d\n", bx,by,tx,ty);

    //Compute Hexa distance

    //OPTIM: reduce multiple access to global variables
    int  dx;
    float  dy;

    dx= bx - tx;
    dy = by - ty;

    if (dx & 1) {
      if (tx & 1)
	dy += 0.5;
      else
	dy -= 0.5;
    }

    float hexa_distance = (float)(dy*dy + 0.75*dx*dx);
    //printf("hexa_distance: %f\n",hexa_distance);

    //OPTIM: Using fast math runtine version
    float neif = __expf((hexa_distance/(-2.0 * radius)));
    //printf("Alpha: %f\n",alpha1);

    if (neif > NEIGHBOR_THRESHOLD){
      int count = atomicAdd(&noc_now,1);
      neighbor_list[count] = tid;
      neighbor_func[count] = neif;
    }

    tid += blockDim.x * gridDim.x;
  }

}
__global__ void GaussianAdapt_Neighbor_Kernel(int nodeIdx, float alpha, float* dev_codebook, float* dev_sample, int data_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float diff;
  int i;

  //printf("Inside KERNEL\n");
  while (tid < noc_now) {
    float d_learningrate = alpha * neighbor_func[tid];

    int id1 = data_dim*nodeIdx;
    int id2 = data_dim*neighbor_list[tid];

#pragma unroll 32
    for (i=0;i<data_dim;i++){
      diff = (dev_sample[id1+i]- dev_codebook[id2+i]);

      diff = diff*d_learningrate;

      dev_codebook[id2+i] += diff; //update with step;
    }

    tid += blockDim.x * gridDim.x;
  }

}

__global__ void GaussianAdapt_Kernel(int nodeIdx, float* dev_learningrate,
				     float* dev_codebook, float* dev_sample, int data_dim, int noc) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float diff;
  int i;
  //printf("Inside KERNEL\n");
  while (tid < noc) {
#pragma unroll
    for (i=0;i<data_dim;i++){
      diff = (dev_sample[data_dim*nodeIdx+i]- dev_codebook[data_dim*tid+i]);

      diff = diff*dev_learningrate[tid];

      dev_codebook[data_dim*tid+i] += diff; //step;
    }

    tid += blockDim.x * gridDim.x;
  }

}

__global__ void GaussianAdapt_Kernel0(int nodeIdx, float* dev_learningrate,
				      float* dev_codebook, float* dev_sample, int data_dim) {
  unsigned int tid = threadIdx.x;
  __shared__ float diff;

  //printf("Inside KERNEL\n");

  diff = (dev_sample[data_dim*nodeIdx+tid]- dev_codebook[data_dim*blockIdx.x+tid]);
  //printf("Diff: %f\n",diff);

  diff = diff*dev_learningrate[blockIdx.x];
  //float step = diff*dev_learningrate[blockIdx.x];
  //printf("Step: %f\n",step);

  //Updating the codebook's elements
  dev_codebook[data_dim*blockIdx.x+tid] += diff; //step;

}
__global__ void GaussianAdapt_Kernel1(int nodeIdx, float alpha, float radius,
				      float* dev_codebook, float* dev_sample, int* dev_coor_x, int* dev_coor_y, int data_dim) {
  unsigned int tid = threadIdx.x;
  //unsigned int n = threadIdx.x + blockIdx.x*blockDim.x;
	
  //printf("Inside KERNEL\n");

  int bx= dev_coor_x[winner_codeno];
  int by= dev_coor_y[winner_codeno];
  int tx = dev_coor_x[blockIdx.x];
  int ty = dev_coor_y[blockIdx.x];

  //printf("bx,bt,tx,ty = %d %d %d %d\n", bx,by,tx,ty);

  //Compute Hexa distance
  int  dx = bx - tx;
  float  dy = by - ty;

  if (dx & 1) {
    if (tx & 1)
      dy += 0.5;
    else
      dy -= 0.5;
  }

  float hexa_distance = (float)(dy*dy + 0.75*dx*dx);
  //printf("hexa_distance: %f\n",hexa_distance);

  float alpha1 = alpha* expf((hexa_distance/(-2.0 * radius)));
  //printf("Alpha: %f\n",alpha1);

  float diff = (dev_sample[data_dim*nodeIdx+tid]- dev_codebook[data_dim*blockIdx.x+tid]);
  //printf("Diff: %f\n",diff);

  float step = diff*alpha1;
  //printf("Step: %f\n",step);
	
  dev_codebook[data_dim*blockIdx.x+tid] += step;

}
/******************************************************************************
 Description: Adapt all codebook vectors assuming a gaussian neighborhood
 relationship between the codebooks.

 Return value: none
******************************************************************************/
void GaussianAdapt_GPU(int nodeIdx, FLOAT radius, FLOAT alpha) {

  //OPTIM: Only update the neighboring node defined by the neighborhood function
  List_Neighboor_kernel<<<GRID_FIX,BLOCK_FIX>>>(radius, dev_coor_x, dev_coor_y, noc);
  GaussianAdapt_Neighbor_Kernel<<<GRID_FIX,BLOCK_FIX>>>(nodeIdx, alpha, dev_codebook, dev_sample, data_dim);

  //OPTIM: vary grids and blocks to get the optimum
  //OPTIM: loop tiling, break down large loop to take advantage of using cache (which is small)
  //int gridofblocks = noc;
  //int blockofthreads = data_dim;

  //printf("Before kernel CALL\n");
  //GaussianAdapt_Kernel1<<<gridofblocks,blockofthreads>>>(nodeIdx,alpha,radius,
  //	  dev_codebook, dev_codebookIdx, dev_sample, dev_sampleIdx, dev_coor_x, dev_coor_y);


  /*if (map_ydim<1024)
    LearningRate_Kernel<<<map_xdim, map_ydim>>>(nodeIdx, dev_learningrate, alpha, radius,
    dev_coor_x, dev_coor_y);
    else{
    dim3 grids(map_xdim,map_ydim,1);
    LearningRate_Kernel<<<grids,1>>>(nodeIdx, dev_learningrate, alpha, radius,
    dev_coor_x, dev_coor_y);
    }*/

  //LearningRate_Kernel<<<GRID_FIX,BLOCK_FIX>>>(nodeIdx, dev_learningrate, alpha, radius, dev_coor_x, dev_coor_y, noc);
  //GaussianAdapt_Kernel<<<GRID_FIX,BLOCK_FIX>>>(nodeIdx, dev_learningrate,dev_codebook, dev_sample, data_dim,noc);

  //LearningRate_Kernel0<<<map_xdim, map_ydim>>>(nodeIdx, dev_learningrate, alpha, radius, dev_coor_x, dev_coor_y);
  //GaussianAdapt_Kernel0<<<gridofblocks,blockofthreads>>>(nodeIdx, dev_learningrate,dev_codebook, dev_sample, data_dim);


  //printf("After kernel CALL\n");
}

void ComputeHRSOMPerformance(struct Parameters *parameters) {
  int nnum = 0;
  struct Graph *gptr;
  struct Node *node;
  struct Winner winner = { 0 };
  struct Map *map;
  map = &parameters->map;

  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  int g_nodeIdx=0;
  int gcnt=0;
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      node = gptr->nodes[nnum];

      FindWinnerEucledian_GPU1(g_nodeIdx, node, &winner);/* Find best matching codebook*/
      fprintf(file_res,"%d %d %d %d %s\n", gcnt, g_nodeIdx, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

      g_nodeIdx++;
    }
    gcnt++;
  }
  fclose(file_res);

  ComputeNetPerformance(*parameters);

  if (parameters->valid != NULL){
	  CopyTestDataSample_H2D(parameters, parameters->valid);

	  char fileval[200];
	  sprintf(fileval, "%s_val", fileresult);
	  FILE *file_resval = fopen(fileval,"w");
	  if (file_resval==NULL) fprintf(stderr, "Error open file result validation\n");

	  gcnt=0;
	  g_nodeIdx = 0;
	  //For (every node in the test set){
	  for (gptr = parameters->valid; gptr != NULL; gptr = gptr->next) {
		  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
			  node = gptr->nodes[nnum];

			  FindWinnerEucledian_GPU1(g_nodeIdx, node, &winner);
			  fprintf(file_resval,"%d %d %d %d %s\n", gcnt, g_nodeIdx, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

			  g_nodeIdx++;

		  }
		  gcnt++;
	  }
	  fclose(file_resval);

  }
  //parameters->valid = NULL; // Restore pointer

  if (parameters->test != NULL){
	  CopyTestDataSample_H2D(parameters, parameters->test);

	  char filetest[200];
	  sprintf(filetest, "%s_test", fileresult);
	  FILE *file_restest = fopen(filetest,"w");
	  if (file_restest==NULL) fprintf(stderr, "Error open file result test\n");

	  gcnt=0;
	  g_nodeIdx = 0;
	  //For (every node in the test set){
	  for (gptr = parameters->test; gptr != NULL; gptr = gptr->next) {
		  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
			  node = gptr->nodes[nnum];

			  FindWinnerEucledian_GPU1(g_nodeIdx, node, &winner);
			  fprintf(file_restest,"%d %d %d %d %s\n", gcnt, g_nodeIdx, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

			  g_nodeIdx++;
		  }
		  gcnt++;
	  }
	  fclose(file_restest);

  }
  //parameters.test = NULL; // Restore pointer
}


/******************************************************************************
 Description: Main training routine for a SOM, this also works for standard
 SOM, and for SOM-SD.

 Return value:  0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainSOM_GPU(struct Parameters *parameters) {
  UNSIGNED i, nnum;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node;
  int g_nodeIdx, gcnt;
  struct Graph *gptr;
  FILE *logfile;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;
  time_t starttime;
  char file_res_exp[20];
  char command[200];

  starttime = time(NULL);
  CopyData_H2D(parameters);
  printf("Time Used for transfering data: %s\n", PrintTime(time(NULL) - starttime));

  //if (map_ydim<1024)
  //	  Gpu_Init_Variables<<<map_ydim,map_xdim>>>(dev_coor_x,dev_coor_y);
  //else{
  dim3 grids(map_xdim,map_ydim);
  Gpu_Init_Variables_2<<<grids,1>>>(dev_coor_x,dev_coor_y);
  //}

  //PrintData_Kernel<<<1,1>>>(dev_sample, dev_codebook, noc, data_dim, num_samples, dev_coor_x,dev_coor_y);

  starttime = time(NULL);
  
  //InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file   */

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  
  printf("Training a SOM map in GPU mode......\n"); /* Print what is being done      */
  map = &parameters->map;

  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
    tlen += gptr->numnodes;
  }
  tlen = tlen * (parameters->rlen - map->iter);

  t = 0;
  //starttime = time(NULL);
  for (i = map->iter; i < parameters->rlen; i++) {
    //getTimeElapseStart();
    //printf("EPOCH %d\n\n", i);
    //if (parameters->graphorder == 1)
    //	  parameters->train = RandomizeGraphOrder(parameters->train);

    counter = 0;
    terror = 0.0;
    gcnt = 0;
    g_nodeIdx=0;

    //FILE *file_res = fopen(fileresult,"w");
    //if (file_res==NULL) fprintf(stderr, "Error open file result\n");

    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      gcnt++;
      //OPTIM: kernelize this to store all in device
      alpha_t =  parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
      radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;
      float radius_tt = radius_t*radius_t;

      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	node = gptr->nodes[nnum];

	t++;

	FindWinnerEucledian_GPU(g_nodeIdx, &winner);/* Find best matching codebook*/

	//FindWinnerEucledian_GPU1(g_nodeIdx, node, &winner);
	//fprintf(file_res,"%d %d %d %d %s\n", gcnt, g_nodeIdx, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

	GaussianAdapt_GPU(g_nodeIdx, radius_tt, alpha_t);/* update codebook*/
	//Limit: did not store the winning coordinate to node->x and node->y
	//printf("EPOCH %d\n\n", i);


	terror += winner.diff;
	counter++;

	g_nodeIdx++;
      }
    }

    //fclose(file_res);
    //fflush(file_res);

    //sprintf(file_res_exp, "%s.%d", fileresult, i);
    //sprintf(command, "cp %s %s", fileresult, file_res_exp);
    //system(command);
    //printf(command);

    //ComputeMutagPerformance(*parameters, 0);

    map->iter++;

    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    PrintProgress(map->iter); /* Print Progress */
    Hausekeeping(parameters);
    //fprintf(stderr, "Time Used on CPU: %s\n", PrintTime(time(NULL) - starttime));
    //getTimeElapseEnd();
  }
  StopProgressMeter();
  if (logfile != stdout)  MyFclose(logfile);

  //PrintData_Kernel<<<1,1>>>(dev_sample, dev_codebook, noc, data_dim, num_samples, dev_coor_x,dev_coor_y);

  printf("Time Used for training: %s\n", PrintTime(time(NULL) - starttime));
   
  CopyData_D2H(parameters,&winner);

  ComputeHRSOMPerformance(parameters);


  return 0;
}


__global__ void UpdateWinningCoor_kernel(int ssdim, int nodeIdx, int *dev_nodesample, int* dev_coor_x, int* dev_coor_y)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0) {
    dev_nodesample[nodeIdx*ssdim] = dev_coor_x[winner_codeno];
    dev_nodesample[nodeIdx*ssdim+1] = dev_coor_y[winner_codeno];

    /*for (i=0;i<8;i++){
      printf("%d ", dev_nodesample[nodeIdx*ssdim+i]);
      }
      printf("\n");*/
  }
}

__global__ void UpdateChildrensLocation_kernel(int data_dim, int ssdim, int fan_out, int nodeIdx, int offset, int *dev_nodesample, float *dev_sample)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < fan_out) {

    int g_nodechild = dev_nodesample[nodeIdx*ssdim+2+tid];
    if (g_nodechild > -1){
      dev_sample[nodeIdx*data_dim+offset+tid*2] = (float) dev_nodesample[g_nodechild*ssdim];
      dev_sample[nodeIdx*data_dim+offset+tid*2+1] = (float) dev_nodesample[g_nodechild*ssdim+1];
    }

  }
  __syncthreads();
  /*
  if (tid==0){
    int i;
    for (i=0;i<data_dim;i++)
      printf("%f ", dev_sample[nodeIdx*data_dim+i]);
    printf("\n");
    }*/
}

void UpdateChildrensLocation_GPU(struct Graph *gptr, int g_nodeIdx)
{

  int offset = gptr->ldim;

  UpdateChildrensLocation_kernel<<<1,fan_out>>>(data_dim, ssdim, fan_out, g_nodeIdx, offset, dev_nodesample, dev_sample);


  /*for (i = 0; i < gptr->FanOut; i++){
    if (node->children[i] != NULL){
    node->points[offset+i*2] =  (FLOAT)node->children[i]->x;
    node->points[offset+i*2+1]= (FLOAT)node->children[i]->y;
    }
    }
  */
}

/******************************************************************************
 Description: Main training routine for a SOM, this also works for standard
 SOM, and for SOM-SD.

 Return value:  0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainSOMSD_GPU(struct Parameters *parameters) {
  UNSIGNED i, nnum;
  FLOAT alpha_t, radius_t, radius_tt;
  struct Map *map;
  struct Node *node;
  int g_nodeIdx, gcnt;
  struct Graph *gptr;
  FILE *logfile;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;
  time_t starttime;

  starttime = time(NULL);
  CopyData_H2D(parameters);
  fprintf(stderr, "Time Used for transfering data: %s\n", PrintTime(time(NULL) - starttime));

  //if (map_ydim<1024)
  //	  Gpu_Init_Variables<<<map_ydim,map_xdim>>>(dev_coor_x,dev_coor_y);
  //else{
  dim3 grids(map_xdim,map_ydim);
  Gpu_Init_Variables_2<<<grids,1>>>(dev_coor_x,dev_coor_y);
  //}

  //PrintData_Kernel<<<1,1>>>(dev_sample, dev_codebook, noc, data_dim, num_samples, dev_coor_x,dev_coor_y);

  starttime = time(NULL);

  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  //InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file   */

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */

  printf("Training a SOMSD map in GPU mode......\n"); /* Print what is being done      */
  map = &parameters->map;

  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
    tlen += gptr->numnodes;
  }
  tlen = tlen * (parameters->rlen - map->iter);

  t = 0;
  starttime = time(NULL);
  for (i = map->iter; i < parameters->rlen; i++) {
    //getTimeElapseStart();
    //printf("EPOCH %d\n\n", i);
    //if (parameters->graphorder == 1)
    //	  parameters->train = RandomizeGraphOrder(parameters->train);

    counter = 0;
    terror = 0.0;
    gcnt = 0;
    g_nodeIdx = 0;/*the global node index*/
    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      //OPTIM: kernelize this to store all in device
      alpha_t =  parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
      radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;
      radius_tt = radius_t*radius_t;

      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	node = gptr->nodes[nnum];

	t++;

	UpdateChildrensLocation_GPU(gptr, g_nodeIdx); /* Update child-state-vector   */

	if (map->iter==parameters->rlen-1)
	  FindWinnerEucledian_GPU1(g_nodeIdx, node, &winner);
	else
	  FindWinnerEucledian_GPU(g_nodeIdx, &winner);
	
	UpdateWinningCoor_kernel<<<1,1>>>(ssdim, g_nodeIdx, dev_nodesample, dev_coor_x,dev_coor_y);

	GaussianAdapt_GPU(g_nodeIdx, radius_tt, alpha_t);/* update codebook*/

	if (map->iter==parameters->rlen-1)
	  fprintf(file_res,"%d %d %d %d %s\n", gcnt, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
	//printf("EPOCH %d\n\n", i);
	terror += winner.diff;
	counter++;

	g_nodeIdx++;
      }
      gcnt++;
    }

    map->iter++;

    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    PrintProgress(map->iter); /* Print Progress */
    Hausekeeping(parameters);
    //fprintf(stderr, "Time Used on CPU: %s\n", PrintTime(time(NULL) - starttime));
    //getTimeElapseEnd();
  }
  StopProgressMeter();
  if (logfile != stdout)  MyFclose(logfile);

  //PrintData_Kernel<<<1,1>>>(dev_sample, dev_codebook, noc, data_dim, num_samples, dev_coor_x,dev_coor_y);

  printf("Time Used for training: %s\n", PrintTime(time(NULL) - starttime));

  CopyData_D2H(parameters,&winner);

  fclose(file_res);
  //End Tuc special

  return 0;
}

__global__ void Gpu_Init_glb_ivec(int g_nodeIdx, int data_dim, int offset, int dimm, int graph_dim, int noc,
				  float* glb_ivec, float* nodetmp, float* dev_sample) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < dimm) {
    if (tid<offset) glb_ivec[tid] = dev_sample[data_dim*g_nodeIdx+tid];
    else if (tid<graph_dim) glb_ivec[tid] = 0;

    if (tid<noc) nodetmp[tid] = 0;

    tid += blockDim.x * gridDim.x;
  }

}
__global__ void PMGSOM_First_kernel(int nodeIdx, int ssdim, int offset, int numneighbor,
				    int goptx, int gopty, int gopt_numxblocks, float* glb_ivec, int* dev_nodesample) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numneighbor) {
    int g_nodeneighbor = dev_nodesample[nodeIdx*ssdim+2+tid];
    if (g_nodeneighbor > -1){
      int x = dev_nodesample[g_nodeneighbor*ssdim];
      int y = dev_nodesample[g_nodeneighbor*ssdim+1];

      glb_ivec[offset + x/goptx+ y/gopty * gopt_numxblocks] += 1.0;
    }
  }
}
__global__ void PMGSOM_Second_kernel(int nodeIdx, int ssdim, int map_xdim, int numneighbor,
				     float* glb_ivec, float* nodetmp, int* xset, int* yset, int* dev_nodesample) {

  noc_now =0;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numneighbor) {
    int g_nodeneighbor = dev_nodesample[nodeIdx*ssdim+2+tid];
    if (g_nodeneighbor > -1){
      int x = dev_nodesample[g_nodeneighbor*ssdim];
      int y = dev_nodesample[g_nodeneighbor*ssdim+1];

      nodetmp[x+y*map_xdim] += 1.0;

      int count = atomicAdd(&noc_now,1);
      xset[count] = x;
      yset[count] = y;
    }

  }
}

__device__ float ComputeHexaDistance_GPU(int bx, int by, int tx, int ty)
{
  float ret, dy;
  int dx;

  dx = bx - tx;
  dy = by - ty;

  if (dx & 1) {
    if (tx & 1)
      dy += 0.5;
    else
      dy -= 0.5;
  }

  ret = dy * dy;
  ret += 0.75 * dx * dx;

  return ret;

}

__global__ void PMGSOM_Third_kernel(int noc, int neighc, int map_xdim, float argue1,float argue2,float argue3,
				    float* glb_ivec, float* nodetmp, int* xset, int* yset) {

  noc_now =0;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i;

  while (tid < noc) {

    float maximp = 0;
    int xd = tid / map_xdim;
    int yd = tid % map_xdim;

#pragma unroll 32
    for (i=0;i<neighc;i++){
      float d = ComputeHexaDistance_GPU(xd, yd, xset[i], yset[i]);
      float winval = __expf(-(d*d)/argue1)/argue2;
      float tmp = __expf(d/argue3)*nodetmp[xset[i]+yset[i]*map_xdim]*winval;
      if (maximp<tmp)
	maximp = tmp;
    }
    if (nodetmp[xd+yd*map_xdim] < maximp)
      nodetmp[xd+yd*map_xdim] = maximp;


    tid += blockDim.x * gridDim.x;
  }

}
__global__ void PMGSOM_Fourth_kernel(int offset, int noc, int map_xdim, int goptx, int gopty, int gopt_numxblocks,
				     float* glb_ivec, float* nodetmp) {

  noc_now =0;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < noc) {

    int xd = tid / map_xdim;
    int yd = tid % map_xdim;

    glb_ivec[offset + xd/goptx+ yd/gopty * gopt_numxblocks] += nodetmp[xd+yd*map_xdim];

    tid += blockDim.x * gridDim.x;
  }

}
__global__ void Gpu_Transfer_glb_ivec_to_sample(int g_nodeIdx, int offset, int graph_dim, int data_dim,
						float* glb_ivec, float* dev_sample) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < graph_dim - offset) {
    dev_sample[data_dim*g_nodeIdx+offset+tid] = glb_ivec[offset+tid];

    tid += blockDim.x * gridDim.x;
  }

}

int TrainPMGraphSOM_GPU(struct Parameters *parameters) {
  UNSIGNED i, offset, nnum, goptx, gopty, gopt_numxblocks;
  FLOAT alpha_t, radius_t, r_fuzzy, sigma;
  int g_nodeIdx, gcnt;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  double PI = 3.1415926535897932;
  FLOAT converge = (float)1/sqrtf(2*PI);
  float* glb_ivec;

  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;
  time_t starttime;
  starttime = time(NULL);

  CopyData_H2D(parameters);
  fprintf(stderr, "Time Used for transferring data: %s\n", PrintTime(time(NULL) - starttime));

  dim3 grids(map_xdim,map_ydim);
  Gpu_Init_Variables_2<<<grids,1>>>(dev_coor_x,dev_coor_y);

  starttime = time(NULL);

  //InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file       */

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  printf("Training PMGraphSOM in GPU ......"); /* Print what is being done  */
  map = &parameters->map; /* This makes de-referencing easier         */
  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    tlen += gptr->numnodes;
  }
  tlen = tlen * (parameters->rlen - map->iter);

  goptx = map->goptx;
  gopty = map->gopty;
  gopt_numxblocks = map->xdim / goptx;

  /* Now lets get started (to train map) */
  gptr = parameters->train;
  t = 0;
  int wsize = max_pnum + fan_out;
  int neighc;
  int prefuzzy = 3;

  float *nodetmp;
  int *xset, *yset;
  HANDLE_ERROR(cudaMalloc((void**)&xset, wsize * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&yset, wsize * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&nodetmp, noc * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&glb_ivec, gptr->dimension * sizeof(float)));

  printf("Data_dim (map->dim) %d graph->dimension %d noc %d\n", data_dim, gptr->dimension, noc);
  int den = 2;
  fprintf(stderr, "\no PM for first %d iterations; PM radius den = %d\n", (prefuzzy+1), den);

  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (i = map->iter; i < parameters->rlen; i++) {
    counter = 0;
    int gctn = 0;
    terror = 0.0; /* Init train error */

    g_nodeIdx = 0;
    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      gctn++;
      alpha_t = parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
      radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
      float radius_tt = radius_t*radius_t;
      r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
      sigma = converge+(float)r_fuzzy;
      float argue1 = (2*sigma*sigma);
      float argue2 = (sigma*sqrtf(2*PI));
      float argue3 = (-2.0 * r_fuzzy * r_fuzzy);

      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	node = gptr->nodes[nnum]; /* This makes de-referencing easier */
	t++;

	//Assigning kernel here
	offset = gptr->ldim;
	int dimm = max(gptr->dimension, noc);

	Gpu_Init_glb_ivec<<<GRID_FIX,BLOCK_FIX>>>(g_nodeIdx, data_dim, offset,dimm,gptr->dimension,noc,glb_ivec,
						  nodetmp,dev_sample);

	if (i>prefuzzy) {
	  PMGSOM_Second_kernel<<<GRID_FIX,BLOCK_FIX>>>(g_nodeIdx, ssdim, offset, fan_in+fan_out,
						       glb_ivec, nodetmp, xset, yset, dev_nodesample);

	  neighc = node->numchildren+ node->numparents;
	  PMGSOM_Third_kernel<<<GRID_FIX,BLOCK_FIX>>>(noc, neighc, map_xdim, argue1,argue2, argue3,
						      glb_ivec, nodetmp, xset, yset);

	  PMGSOM_Fourth_kernel<<<GRID_FIX,BLOCK_FIX>>>(offset, noc, map_xdim, goptx, gopty, gopt_numxblocks,
						       glb_ivec, nodetmp);
	} else
	  PMGSOM_First_kernel<<<GRID_FIX,BLOCK_FIX>>>(g_nodeIdx, ssdim, offset, fan_in+fan_out,
						      goptx, gopty, gopt_numxblocks, glb_ivec, dev_nodesample);

	Gpu_Transfer_glb_ivec_to_sample<<<GRID_FIX,BLOCK_FIX>>>(g_nodeIdx, offset, gptr->dimension, data_dim,
								glb_ivec, dev_sample);

	if (map->iter==parameters->rlen-1)
	  FindWinnerEucledian_GPU1(g_nodeIdx, node, &winner);
	else
	  FindWinnerEucledian_GPU(g_nodeIdx, &winner);
	
	UpdateWinningCoor_kernel<<<1,1>>>(ssdim, g_nodeIdx, dev_nodesample, dev_coor_x,dev_coor_y);

	GaussianAdapt_GPU(g_nodeIdx, radius_tt, alpha_t);/* update codebook*/

	if (map->iter==parameters->rlen-1)
	  fprintf(file_res,"%d %d %d %d %s\n", gcnt, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

	terror += winner.diff;
	counter++;

	g_nodeIdx++;

      }
    }

    map->iter++;

    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    PrintProgress(map->iter); /* Print Progress */
    Hausekeeping(parameters);
    //fprintf(stderr, "Time Used on CPU: %s\n", PrintTime(time(NULL) - starttime));
    //getTimeElapseEnd();
  }
  StopProgressMeter();
  if (logfile != stdout)  MyFclose(logfile);

  printf("Time Used for training: %s\n", PrintTime(time(NULL) - starttime));

  CopyData_D2H(parameters,&winner);

  fclose(file_res);

  cudaFree(glb_ivec);
  cudaFree(nodetmp);
  cudaFree(xset);
  cudaFree(yset);

  return 1;
}


int TrainMap_GPU(struct Parameters *parameters) {

  //getTimeElapseStart();

  if (parameters->map.type == TYPE_GRAPHSOM)
    TrainPMGraphSOM_GPU(parameters);
  else
    if (parameters->map.type == TYPE_SOMSD)
      TrainSOMSD_GPU(parameters);
    else
      if (parameters->map.type == TYPE_SOM)
	TrainSOM_GPU(parameters);

  printf("END PROGRAMME\n");
  //CopyData_D2H(parameters);

  FreeCuda();

  return 0;
}
//#endif //__USE_CUDA__
/* END OF FILE */

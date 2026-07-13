/*
*/

/************/
/* Includes */
/************/
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"
#include "data.h"
#include "fileio.h"
#include "utils.h"
#include "train_cpu.h"
#include "testsom.h"

#define GAUSS_ALPHA_CONSTANT 4.0
extern char *fileresult;
extern FLOAT* glb_ivec;

/*****************************************************************************
Description: Create a new map for the SOM, and initialize all codebook
             vectors with random values chosen from within a range of values
             observed in a given dataset.
             Initialization can occur in two different modes:
             mode=INIT_LINEAR: Initialize codebooks such that those codebooks
                               which are located closest to the origin of the
                               map receive the smallest values from within
                               a computed range of valid values, and codebooks
                               located furthest from the origin receive the
                               largest values. The increase of values between
                               the two corners is linear.
             mode=INIT_RANDOM: Randomly initialize all codebooks (this is the
                               default behaviour).

Return value: 1 on success, 0 otherwise (an error is set which can be checked
              using PrintErrors() ).
*****************************************************************************/
UNSIGNED InitCodes_CPU(struct Map *map, struct Graph *data, UNSIGNED mode)
{
  UNSIGNED dim, sdim, noc;
  UNSIGNED i, x, y;
  FLOAT *maval, *mival;
  struct Graph *gptr;
  struct Node *node;

  noc = map->xdim * map->ydim;  /* Total number of codebooks */
  if (noc == 0u){
    AddError("Network dimension is zero!");
    return 0u;
  }
  map->codes = (struct Codebook*)MyCalloc(noc, sizeof(struct Codebook));

  /* Find maximum dimension of codebook entries */
  dim = 0u;
  for (gptr = data; gptr != NULL; gptr = gptr->next){
    if (gptr->dimension > dim)
      dim = data->dimension;
  }
  if (dim == 0u){
    AddError("Dimension of all node vectors in the dataset is zero!");
    return 0u;
  }

  map->dim = dim;

  /* allocate codebook vectors */
  i = 0u;
  for (y = 0u; y < map->ydim; y++){
    for (x = 0u; x < map->xdim; x++){
      map->codes[i].points = (FLOAT*)MyMalloc(map->dim * sizeof(FLOAT));
      map->codes[i].x = x;
      map->codes[i].y = y;
      map->codes[i].label = 0;
      i++;
    }
  }

  /* Find the maximum and minimum values of data */
  maval = (FLOAT*)MyMalloc(dim * sizeof(FLOAT));
  mival = (FLOAT*)MyMalloc(dim * sizeof(FLOAT));

  for (i = 0; i < dim; i++) {
    maval[i] = MIN_FLOAT;
    mival[i] = MAX_FLOAT;
  }

   for (gptr = data; gptr != NULL; gptr = gptr->next){
      for (i = 0; i < gptr->numnodes; i++){
	node = gptr->nodes[i];
	for (x = 0; x < dim; x++){
	  maval[x] = max(node->points[x], maval[x]);
	  mival[x] = min(node->points[x], mival[x]);
	}
      }
    }


  /* Check if info about the mapping of data was available */
  sdim = data->ldim + (data->FanOut + data->FanIn) * 2;
  for (x = data->ldim; x < sdim; x++){
    if (maval[x] != -1 || mival[x] != -1)
      break;
  }
  if (x == sdim && sdim != data->ldim){
    /* No mapping info available (all struct info is (-1 -1)).
       Will encourage the generation of random mapping info... */
    for (x = data->ldim; x < sdim; x+=2){
      maval[x]   = map->xdim -1;
      maval[x+1] = map->ydim -1;
      mival[x]   = 0;
      mival[x+1] = 0;
    }
  }

    /* Randomize the vector values */
    for (i = 0; i < noc; i++){
      for (x = 0; x < dim; x++){
	map->codes[i].points[x] = mival[x] + (maval[x] - mival[x]) * drand48();
      }
    }

  free(mival);
  free(maval);

  return 1;
}

/*****************************************************************************
Description: Create a new map for the GRAPHSOM, and initialize all codebook
             vectors with random values chosen from within a range of values
             observed in a given dataset.
             Initialization can occur in two different modes:
             mode=INIT_LINEAR: Initialize codebooks such that those codebooks
                               which are located closest to the origin of the
                               map receive the smallest values from within
                               a computed range of valid values, and codebooks
                               located furthest from the origin receive the
                               largest values. The increase of values between
                               the two corners is linear.
             mode=INIT_RANDOM: Randomly initialize all codebooks (this is the
                               default behaviour).

Return value: 1 on success, 0 otherwise (an error is set which can be checked
              using PrintErrors() ).
*****************************************************************************/
UNSIGNED InitGsomCodes_CPU(struct Map *map, struct Graph *data, UNSIGNED mode)
{
  UNSIGNED dim, noc;
  int  totalnodes;
  int maxout, maxin, maxneighbor;
  UNSIGNED i, x, y;
  UNSIGNED goptx, gopty;
  FLOAT *maval, *mival;
  struct Graph *gptr;
  struct Node *node;

  noc = map->xdim * map->ydim;  /* Total number of codebooks */
  if (noc == 0u){
    AddError("Network dimension is zero!");
    return 0u;
  }
  map->codes = (struct Codebook*)MyCalloc(noc, sizeof(struct Codebook));

  goptx = map->goptx;
  gopty = map->gopty;

  maxout = 0; maxin = 0; maxneighbor = 0;

  /* Find maximum dimension of codebook entries */
  dim = 0u;
  for (gptr = data; gptr != NULL; gptr = gptr->next){
    if (gptr->ldim > dim)
      dim = data->ldim;
    if (gptr->FanOut > maxout)
      maxout = gptr->FanOut;
    if (gptr->FanIn > maxin)
      maxin = gptr->FanIn;
    if (gptr->FanIn + gptr->FanOut > maxneighbor)
      maxneighbor = gptr->FanIn + gptr->FanOut;
  }

  if (map->xdim % goptx){
    AddError("Horizontal matrix dimension must be a multiple of gopt!");
    return 0u;
  }
  else if (map->ydim % gopty){
    AddError("Vertical matrix dimension must be a multiple of gopt!");
    return 0u;
  }
  dim += noc / (goptx * gopty);

  if (dim == 0u){
    AddError("Strange, the dimension of all code vectors is zero!");
    return 0u;
  }
  if (map->topology == TOPOL_VQ && map->type == TYPE_GRAPHSOM){
    AddError("Vector quatization not implemented for network type GraphSOM\n");
    return 0u;
  }

  map->dim = dim;

  /* allocate codebook vectors */
  i = 0u;
  for (y = 0u; y < map->ydim; y++){
    for (x = 0u; x < map->xdim; x++){
      map->codes[i].points = (FLOAT*)MyMalloc(map->dim * sizeof(FLOAT));
      map->codes[i].x = x;
      map->codes[i].y = y;
      map->codes[i].label = 0;
      i++;
    }
  }

  /* Find the maximum and minimum values of data labels */
  maval = (FLOAT*)MyMalloc(dim * sizeof(FLOAT));
  mival = (FLOAT*)MyMalloc(dim * sizeof(FLOAT));

  for (i = 0; i < data->ldim; i++) {
    maval[i] = MIN_FLOAT;
    mival[i] = MAX_FLOAT;
  }

  totalnodes = 0;
  if (map->topology != TOPOL_VQ){
    for (gptr = data; gptr != NULL; gptr = gptr->next){
      for (i = 0; i < gptr->numnodes; i++){
	totalnodes += gptr->numnodes;
	node = gptr->nodes[i];
	for (x = 0; x < data->ldim; x++){
	  maval[x] = max(node->points[x], maval[x]);
	  mival[x] = min(node->points[x], mival[x]);
	}
      }
    }
  }

  if (map->type == TYPE_GRAPHSOM){
    for (x = data->ldim; x < dim; x++){
      maval[x]   = (float)(maxneighbor*2) / (dim - data->ldim);
      mival[x]   = 0;
    }
  }
  else{
    for (x = data->ldim; x < data->ldim + data->FanOut + data->FanIn; x+=2){
      maval[x]   = map->xdim;
      mival[x]   = 0;
      maval[x+1] = map->ydim;
      mival[x+1] = 0;
    }
  }

    //    fprintf(stderr, "\nrandom mode\n");
    /* Randomize the vector values */
    for (i = 0; i < noc; i++){
      for (x = 0; x < dim; x++){
	map->codes[i].points[x] = mival[x] + (maval[x] - mival[x]) * drand48();
      }
    }


  free(mival);
  free(maval);

  return 1;
}

/******************************************************************************
Description: Create a newly initialized map

Return value:
******************************************************************************/
void InitMap_CPU(struct Parameters *params, int argc, char **argv)
{
  UNSIGNED i, mode;
  struct Map *map;

  mode = INIT_DEFAULT;
  map = &params->map;
  /* Check for relevant command line paramaters */
  for (i = 1; i < argc; i++){
    if (!strncmp(argv[i], "-neigh", 2))
      map->neighborhood = GetNeighborhoodID(argv[++i], NULL);
    else if (!strncmp(argv[i], "-topol", 2))
      map->topology = GetTopologyID(argv[++i], NULL);
    else if (!strcmp(argv[i], "-linear"))
      mode = INIT_LINEAR;
    else if (!strcmp(argv[i], "-maptype"))
      params->map.type = GetMapType(argv[++i]);
    else if (!strncmp(argv[i], "-xdim", 2))
      GetArg(argv, i++, TYPE_UNSIGNED, &map->xdim);
    else if (!strncmp(argv[i], "-ydim", 2))
      GetArg(argv, i++, TYPE_UNSIGNED, &map->ydim);
    //else if (!strcmp(argv[i], "-help") || !strcmp(argv[i], "-h") || !strcmp(argv[i], "-?"))//{
    //    Usage();
    //}

    if (CheckErrors() != 0)
      break;
  }

  /* Set default values for gopt if required */
  if (params->map.type == TYPE_GRAPHSOM){
    if (params->map.goptx == 0)
      params->map.goptx = 1;
    if (params->map.gopty == 0)
      params->map.gopty = params->map.goptx;
  }

  if (CheckErrors() == 0){        /* If there weren't and errors so far ... */
    /* Check that we have useful initial data */
    if (map->topology == TOPOL_VQ)
      map->neighborhood = NEIGH_NONE;
    else if (map->neighborhood == UNKNOWN){
      AddMessage("NOTE: Neighborhood defaulted to 'gaussian'");
      map->neighborhood = NEIGH_GAUSSIAN;
    }
    if (map->topology == UNKNOWN){
      AddMessage("NOTE: Topology defaulted to 'hexagonal'");
      map->topology = TOPOL_HEXA;
    }
    if (map->xdim * map->ydim == 0)
      AddError("Network dimension not specified, or is zero.");
  }

  if (params->train == NULL)
    AddError("Can't initialize map without training data.");

  if (CheckErrors() == 0){     /* If there were no errors so far then      */
    fprintf(stderr, "Initializing network...."); /* Print what's happening */
    fprintf(stderr, "%d", params->seed);
    srand48(params->seed);     /* Initialize random number generator       */

    if (map->type == TYPE_GRAPHSOM){ /*     Allocate and initialize a map   */
	InitGsomCodes_CPU(map, params->train, mode); /* Special Init for GRAPHSOM*/
    }
    else
      InitCodes_CPU(map, params->train, mode);     /* Init any other SOM       */

    if (CheckErrors() == 0)
      fprintf(stderr, "\033[01;32m%50s\033[00m\n", "[OK]");    /* print confirmation          */
    else
      fprintf(stderr, "\033[01;31m%50s\033[00m\n", "[FAILED]");/*Network initialization failed*/
  }

  return;
}

void UpdateChildrensLocation_CPU(struct Graph *gptr, struct Node *node)
{
    UNSIGNED i, offset;

    offset = gptr->ldim;
    for (i = 0; i < gptr->FanOut; i++){
      if (node->children[i] != NULL){
        node->points[offset+i*2] =  (FLOAT)node->children[i]->x;
        node->points[offset+i*2+1]= (FLOAT)node->children[i]->y;
                }
          }
}
/******************************************************************************
 Description: Compute the geographical distance between two codebook coordinates
 given a hexagonal neighborhood relationship.

 Return value: The squared!!  eucledian distance between two points.
******************************************************************************/
FLOAT ComputeHexaDistance_CPU(int bx, int by, int tx, int ty)
{
  FLOAT ret, dy;
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
  //  return (FLOAT)sqrtf(ret);
}


/******************************************************************************
 Description: Find best matching codebook using the Eucledian distance meassure.

 Return value: The best matching codebook is returned to parameter "winner".
******************************************************************************/
void FindWinnerEucledian_CPU(struct Map *map, struct Node *node,
			 struct Graph *gptr, struct Winner *winner) {
  FLOAT *mu;
  UNSIGNED vdim;
  UNSIGNED noc;
  FLOAT *sample;
  UNSIGNED n;
  FLOAT diffsf;
  UNSIGNED bestn;

  mu = node->mu;
  diffsf = FLT_MAX;
  sample = node->points;
  vdim = gptr->dimension;
  noc = map->xdim * map->ydim;
  bestn = 0;

  /* Parallelized over codebooks: each thread keeps a private running
     minimum (so the early-exit distance bound still prunes work), then
     the per-thread minima are combined. Essential for high-resolution
     maps where 'noc' (map->xdim * map->ydim) can be in the millions. */
#pragma omp parallel
  {
    FLOAT local_best = FLT_MAX;
    UNSIGNED local_bestn = 0;
    FLOAT *codebook;
    FLOAT diff, difference;
    UNSIGNED i;

#pragma omp for schedule(static) nowait
    for (n = 0; n < noc; n++) { /* For every codebook of the map */
      codebook = map->codes[n].points;
      difference = 0.0;

      /* Compute the difference between codebook and input entry */
      for (i = 0; i < vdim; i++) {
        diff = codebook[i] - sample[i];

        difference += diff * diff * mu[i];
        if (difference > local_best) { //break loop if difference exceeds minimum
          break;
        }
      }

      /* If distance is smaller than previous distances (for this thread) */
      if (difference < local_best) {
        local_bestn = n;
        local_best = difference;
      }
    }

#pragma omp critical
    {
      if (local_best < diffsf) {
        diffsf = local_best;
        bestn = local_bestn;
      }
    }
  }

  winner->codeno = bestn;
  winner->diff = diffsf;
  return;
}
/******************************************************************************
 Description: Adapt all codebook vectors assuming a gaussian neighborhood
 relationship between the codebooks.

 Return value: none
******************************************************************************/
void GaussianAdapt_CPU(struct Graph *gptr, struct Map *map, struct Node *node,
		   struct Winner *winner, FLOAT radius, FLOAT alpha) {
  UNSIGNED n, noc;

  noc = map->xdim * map->ydim;
  node->x = map->codes[winner->codeno].x;
  node->y = map->codes[winner->codeno].y;
  //fprintf(stderr, "\n%d:%d\n", node->x, node->y);
  radius *= radius;

  /* Every codebook is updated independently of every other one, so this
     loop is embarrassingly parallel - important for high-resolution maps
     where 'noc' can be in the millions. */
#pragma omp parallel for schedule(static)
  for (n = 0; n < noc; n++) { /* For every codebook of the map */
    FLOAT dist;
    int i;
    float alpha1;

    /* Compute distance to winner */
    dist = ComputeHexaDistance_CPU(node->x, node->y, map->codes[n].x, map->codes[n].y);

    alpha1 = alpha* expf((dist/(-2.0 * radius)));
    for (i = 0; i < map->dim; i++) {
    	map->codes[n].points[i] += alpha1 * (node->points[i] - map->codes[n].points[i]);
    }

  }
}

void ComputeHRSOMPerformance_cpu(struct Parameters *parameters) {
  int C, n, nnum = 0;
  struct Graph *gptr;
  struct Node *node;
  struct Winner winner = { 0 };
  struct Map *map;

  map = &parameters->map;

  ComputeNetPerformance(*parameters);

  //Begin Tuc special
  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
    	node = gptr->nodes[nnum];
    	UpdateChildrensLocation(gptr, node); /* Update child-state-vector   */
    	FindWinnerEucledian_CPU(map, node, gptr, &winner);/* Find best matching codebook*/
    	fprintf(file_res,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
    }
  }
  fclose(file_res);
  //End Tuc special

  //printf("I am here\n");
  if (parameters->valid!=NULL) {
  	 char fileval[200];
  	 sprintf(fileval, "%s_val", fileresult);
  	  FILE *file_resval = fopen(fileval,"w");
  	  if (file_resval==NULL) fprintf(stderr, "Error open file result\n");

  	  for (gptr = parameters->valid; gptr != NULL; gptr = gptr->next) {
  	    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
  	    	//printf("%d ", gptr->numnodes);
  	    	node = gptr->nodes[nnum];
  	    	UpdateChildrensLocation(gptr, node); /* Update child-state-vector   */
  	    	FindWinnerEucledian_CPU(map, node, gptr, &winner);/* Find best matching codebook*/
  	    	fprintf(file_resval,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

  	    	//printf("%d %d\n",node->label, parameters->winnerclass_train[map->codes[winner.codeno].x][map->codes[winner.codeno].y]);

  	    }
  	  }
  	  fclose(file_resval);
  }

  //printf("I am there\n");
  if(parameters->test!=NULL){
  	 char filetest[200];
  	 sprintf(filetest, "%s_test", fileresult);
  	  FILE *file_restest = fopen(filetest,"w");
  	  if (file_restest==NULL) fprintf(stderr, "Error open file result\n");

  	  for (gptr = parameters->test; gptr != NULL; gptr = gptr->next) {
  	    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
  	    	node = gptr->nodes[nnum];
  	    	UpdateChildrensLocation(gptr, node); /* Update child-state-vector   */
  	    	FindWinnerEucledian_CPU(map, node, gptr, &winner);/* Find best matching codebook*/
  	    	fprintf(file_restest,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));

  	    }
  	  }
  	  fclose(file_restest);
  }

}
/******************************************************************************
 Description: Main training routine for a SOM, this also works for standard
 SOM, and for SOM-SD.

 Return value:  0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainSOM_CPU(struct Parameters *parameters) {
  UNSIGNED i, nnum;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node = NULL;
  struct Graph *gptr;
  FILE *logfile;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;
  time_t starttime;

  //InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file   */

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */  

  fprintf(stderr, "Training SOM in CPU mode......"); /* Print what is being done      */
  map = &parameters->map;

  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
		  tlen += gptr->numnodes;
  }
  tlen = tlen * (parameters->rlen - map->iter);

  t = 0;
  for (i = map->iter; i < parameters->rlen; i++) {
	  starttime = time(NULL);
 	  //getTimeElapseStart();

	  if (parameters->graphorder == 1)
		  parameters->train = RandomizeGraphOrder(parameters->train);

	  counter = 0;
	  terror = 0.0;
	  int gcnt = 0;
	  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
		  gcnt++;
		  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
			  node = gptr->nodes[nnum];
			  alpha_t = //GetAlpha(t, tlen, parameters->alpha);
					  parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
			  radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

			  t++;
			  UpdateChildrensLocation(gptr, node); /* Update child-state-vector   */

			  FindWinnerEucledian_CPU(map, node, gptr, &winner);/* Find best matching codebook*/
			  GaussianAdapt_CPU(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook*/
			  terror += winner.diff;
			  counter++;
		  }
	  }

	  map->iter++;

	  fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
	  fflush(logfile);

	  /*if (_save_then_exit_) {
		  char fname[32];
		  sprintf(fname, "interrupted%d.net", (int)getpid());
		  free(parameters->onetfile);
		  parameters->onetfile = strdup(fname);
		  fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
		  break;
	  }*/

	  /* Create a snapshot if required */
	  if (parameters->snap.interval>0 &&!(map->iter%parameters->snap.interval)) {
		  if (parameters->snap.file != NULL) SaveSnapShot(parameters);
		  if (parameters->snap.command != NULL)
			  system(parameters->snap.command);
	  }

	  PrintProgress(map->iter); /* Print Progress */
	  Hausekeeping(parameters);
	  //fprintf(stderr, "Time Used so far: %s\n", PrintTime(time(NULL) - starttime));
	  //getTimeElapseEnd();

  }
  StopProgressMeter();
  //if (!_save_then_exit_)
	//  fprintf(stderr, "\033[01;32m%56s\033[00m\n", "[OK]");

  if (logfile != stdout)
	  MyFclose(logfile);

  ComputeHRSOMPerformance_cpu(parameters);
  return 0;
}
/******************************************************************************
 Description: Main training routine for a CSOM-SD, this also works for standard
 SOM, and for SOM-SD.

 Return value:  0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainSOMSD_CPU(struct Parameters *parameters) {
  UNSIGNED i, nnum;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;

  //InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file   */

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  //fprint(stderr, "Tuc training map......"); /* Print what is being done      */
  map = &parameters->map;

  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
	 // if (parameters->kfoldtime>=0){
		 // if (gptr->gnum % 10 != parameters->kfoldtime)
		  tlen += gptr->numnodes;
	 // }
  }
  tlen = tlen * (parameters->rlen - map->iter);

  t = 0;
  for (i = map->iter; i < parameters->rlen; i++) {

    if (parameters->graphorder == 1)
      parameters->train = RandomizeGraphOrder(parameters->train);

    counter = 0;
    terror = 0.0;
    int gcnt = 0;
    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      //if (parameters->kfoldtime>=0)
      //  if (gptr->gnum % 10 == parameters->kfoldtime) continue;
      //			fprintf(stderr, "%f\r", (float)(gcnt+1500*i)/(float)(1500*parameters->rlen));
      gcnt++;
      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
 	node = gptr->nodes[nnum];
	alpha_t = parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
	radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;
	
	t++;
	UpdateChildrensLocation(gptr, node); /* Update child-state-vector   */
	
	FindWinnerEucledian_CPU(map, node, gptr, &winner);/* Find best matching codebook*/
	GaussianAdapt_CPU(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook*/
	terror += winner.diff;
	counter++;
      }
    }

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    /* Create a snapshot if required */
    if (parameters->snap.interval>0 &&!(map->iter
					%parameters->snap.interval)) {
      if (parameters->snap.file != NULL)
	SaveSnapShot(parameters);
      if (parameters->snap.command != NULL)
	system(parameters->snap.command);
    }

    //if (parameters->nice)
    //  SleepOnHiLoad(); /* Sleep when system load is high */

    PrintProgress(map->iter); /* Print Progress */
	  Hausekeeping(parameters);
  }
  StopProgressMeter();

  if (logfile != stdout)
    MyFclose(logfile);

  //Begin Tuc special
  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      if (parameters->kfoldtime >= 0)//Can this ever be negative??
	node = gptr->nodes[nnum];
      else{
	fprintf(stderr, PRT_ERROR "Fix bug in function TrainSOMSD_CPU\n");
	exit(0);
      }

      alpha_t =  parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
      radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

      UpdateChildrensLocation(gptr, node); /* Update child-state-vector   */

      FindWinnerEucledian_CPU(map, node, gptr, &winner);/* Find best matching codebook*/

      //printf("%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
      fprintf(file_res,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
    }
  }
  fclose(file_res);
  //int ktime;
  //for (ktime=0;ktime<10;ktime++){
  /*if (parameters->kfoldtime>=0){
	  char filename[100];
	  //sprintf(filename,"%s_val%d",fileresult,ktime);
	  sprintf(filename,"%s_val%d",fileresult,parameters->kfoldtime);
	  printf("%s\n",filename);
	  file_res = fopen(filename,"w");
	  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

	  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
		  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
			  //if (nnum < 12*parameters->kfoldtime || nnum >= 12*(parameters->kfoldtime+1)) continue;//12 activities per person
			  //if (nnum < 60*parameters->kfoldtime || nnum >= 60*(parameters->kfoldtime+1)) continue;//6 activities 10 times on different sensors per person
    		  //if (nnum < 180*parameters->kfoldtime || nnum >= 180*(parameters->kfoldtime+1)) continue;// HASC data
    	  	  if (nnum < 400*parameters->kfoldtime || nnum >= 400*(parameters->kfoldtime+1)) continue;// TROST data

			  node = gptr->nodes[nnum];
			  //alpha_t = GetAlpha(t, tlen, parameters->alpha);
			  //radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

			  if (!parameters->contextual)
				  UpdateOffspringStates(gptr, node);

			  FindWinner(map, node, gptr, &winner);
			  //printf("%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
			  fprintf(file_res,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
		  }
	  }
	  fclose(file_res);
  }*/
  //End Tuc special

  return 0;
}


/******************************************************************************
 Description: Main training routine for training a Fuzzy GraphSOM

 Return value: 0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainPMGraphSOM_CPU(struct Parameters *parameters) {
  UNSIGNED i, k, offset, nnum, goptx, gopty, gopt_numxblocks;
  FLOAT alpha_t, radius_t, r_fuzzy, sigma;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  double PI = 3.1415926535897932; 
  FLOAT converge = (float)1/sqrtf(2*PI);

  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;

  //InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file       */

   /* Set the appropriate function for computing the winner codebook   */
  //FindWinner = FindWinnerEucledianMemorySave; /* Use Eucledian distance        */


  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  fprint(stderr, "Training PMGraphSOM in memory-save mode......"); /* Print what is being done  */
  map = &parameters->map; /* This makes de-referencing easier         */
  tlen = 0; /* Compute the total number of update steps */
  int pnum = 0, maxpnum = 0;
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    tlen += gptr->numnodes;
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      node = gptr->nodes[nnum];
      pnum = node->numparents;
      if (pnum>maxpnum)
	maxpnum = pnum;
    }
  }
  tlen = tlen * (parameters->rlen - map->iter);

  goptx = map->goptx;
  gopty = map->gopty;
  gopt_numxblocks = map->xdim / goptx;

  /* Now lets get started (to train map) */
  t = 0;
  gptr = parameters->train;
  int wsize = maxpnum + gptr->FanOut;
  int xset[wsize];
  int yset[wsize];
  int neighc = 0, idx = 0, xd, yd;
  int prefuzzy = 3;
  FLOAT nodetmp[map->xdim*map->ydim];
  glb_ivec = (float*) MyMalloc(gptr->dimension * sizeof(FLOAT));
  int compute = 0;
  int den = 2;
  fprintf(stderr, "\nno PM for first %d iterations; PM radius den = %d\n", (prefuzzy+1), den);

  //double average_nonzero_codebookelements=0.0;
  //double averagesub_nonzero_codebookelements;

  for (i = map->iter; i < parameters->rlen; i++) {
    counter = 0;
    int gctn = 0;
    terror = 0.0; /* Init train error */

    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      gctn++;
      //averagesub_nonzero_codebookelements=0.0;
      //fprintf(stderr, "%f\r", ((float)i*3750+gctn)/(((float)parameters->rlen)*3750));							
      //fprintf(stderr, "graph %s: ", gptr->gname);
      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	//fputc('\b', stderr);
	compute=0;
	//fprintf(stderr, "%f\r", (float)nnum/(float)gptr->numnodes);
	FLOAT winval = 1;
	node = gptr->nodes[nnum]; /* This makes de-referencing easier */
	alpha_t = parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
	//radius_t = 1.0 + (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;		
	radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
	r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
	sigma = converge+(float)r_fuzzy;				
                                                   			
	t++;
	if (!parameters->contextual) {
	  offset = gptr->ldim;
	  for(k=0;k<offset;k++)
	    glb_ivec[k] = node->points[k];
	  //for(k=offset;k<gptr->dimension;k++){
	  //glb_ivec[k] = 0;
	  //}	
						   
	  memset(&glb_ivec[offset], 0, (gptr->dimension-offset)*sizeof(FLOAT));
	  memset(&nodetmp[0], 0, (map->xdim*map->ydim)*sizeof(FLOAT));
	  //fprintf(stderr, "children: %d\n", node->numchildren);
	  for (k = 0; k < node->numchildren; k++) {
	    //if (node->children[k] != NULL) {
	    if (offset + node->children[k]->x/goptx + node->children[k]->y/gopty*gopt_numxblocks>= gptr->dimension)
	      fprintf(stderr, "sffgfdg\n");
	    if (i>prefuzzy) {
	      nodetmp[node->children[k]->x+node->children[k]->y*map->xdim]++;
	      xset[neighc] = node->children[k]->x;
	      yset[neighc] = node->children[k]->y;
	      neighc++;
	      compute=1;
	    } 
	    else
	      glb_ivec[offset + node->children[k]->x/goptx+ node->children[k]->y/gopty * gopt_numxblocks]++;
	    //} 
	    //	else
	    //break; /* No more offsprings after a NULL-pointer expected */
	  }
	  //fprintf(stderr, "parents: %d\n", node->numparents);
	  for (k = 0u; k < node->numparents; k++) {
	    if (i>prefuzzy) {
	      nodetmp[node->parents[k]->x+node->parents[k]->y*map->xdim]++;
	      xset[neighc]= node->parents[k]->x;
	      yset[neighc]= node->parents[k]->y;
	      neighc++;
	      compute=1;
	    } 
	    else
	      glb_ivec[offset + node->parents[k]->x/goptx+ node->parents[k]->y/gopty * gopt_numxblocks]++;
	  }

	  if (i>prefuzzy && compute) {
	    for (xd = 0; xd < map->xdim; xd++) {
	      for (yd = 0; yd < map->ydim; yd++) {
		FLOAT maximp = 0;
		for (idx = 0; idx < neighc; idx++) {
		  FLOAT d = ComputeHexaDistance_CPU(xd, yd, xset[idx], yset[idx]);
		  if (d > radius_t*8)
		    continue;
		  winval = expf(-(d*d)/(2*sigma*sigma))/(sigma*sqrtf(2*PI));
		  //fprintf(stderr, "%1.9f, %1.9f, %1.9f\n", d, sigma, winval);
		  FLOAT tmp = expf((d/(-2.0 * r_fuzzy * r_fuzzy)))*nodetmp[xset[idx]+yset[idx]*map->xdim]*winval;
		  if (maximp<tmp)
		    maximp = tmp;
		}
		if (nodetmp[xd+yd*map->xdim] < maximp)
		  nodetmp[xd+yd*map->xdim] = maximp;
	      }
	    }

	    neighc = 0;
	    for (xd = 0; xd < map->xdim; xd++)
	      for (yd = 0; yd < map->ydim; yd++)
		glb_ivec[offset+xd/goptx+yd/gopty*gopt_numxblocks]+= nodetmp[xd+yd*map->xdim];
	  }
	}
				
	//for(k=0;k<gptr->dimension;k++)
	//fprintf(stderr, "%f ", glb_ivec[k]);
	//fprintf(stderr, "\n");
	FindWinnerEucledian_CPU(map, node, gptr, &winner); /* Find best matching codebook */
	//if (parameters->removed > 0 && gctn != parameters->removed){
	  GaussianAdapt_CPU(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook */
	  terror += winner.diff;
	  counter++;

      }
    }

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    /* Create a snapshot if required */
    if (parameters->snap.interval>0 &&!(map->iter%parameters->snap.interval)) {
      if (parameters->snap.file != NULL){
	SaveSnapShot(parameters);
	//ClusterINEX2008(*parameters, 0);
      }
      if (parameters->snap.command != NULL)
	system(parameters->snap.command);
    }

    //if (parameters->nice)
     // SleepOnHiLoad(); /* Sleep when system load is high */

    PrintProgress(map->iter); /* Print Progress */
	  Hausekeeping(parameters);
  }
  StopProgressMeter();

  if (logfile != stdout)
    MyFclose(logfile);


  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
	  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
		  compute=0;
		  FLOAT winval = 1;
		  node = gptr->nodes[nnum];
		  alpha_t = parameters->alpha*exp(-((FLOAT)t*t*GAUSS_ALPHA_CONSTANT)/((FLOAT)tlen*tlen));
		  radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
		  r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
		  sigma = converge+(float)r_fuzzy;

		  t++;
		  if (!parameters->contextual) {
			  offset = gptr->ldim;
			  for(k=0;k<offset;k++)
				  glb_ivec[k] = node->points[k];
						   
			  memset(&glb_ivec[offset], 0, (gptr->dimension-offset)*sizeof(FLOAT));
			  memset(&nodetmp[0], 0, (map->xdim*map->ydim)*sizeof(FLOAT));
			  for (k = 0; k < node->numchildren; k++) {
				  if (offset + node->children[k]->x/goptx + node->children[k]->y/gopty*gopt_numxblocks>= gptr->dimension)
					  fprintf(stderr, "sffgfdg\n");
				  if (i>prefuzzy) {
					  nodetmp[node->children[k]->x+node->children[k]->y*map->xdim]++;
					  xset[neighc] = node->children[k]->x;
					  yset[neighc] = node->children[k]->y;
					  neighc++;
					  compute=1;
				  }
				  else
					  glb_ivec[offset + node->children[k]->x/goptx+ node->children[k]->y/gopty * gopt_numxblocks]++;
			  }
			  for (k = 0u; k < node->numparents; k++) {
				  if (i>prefuzzy) {
					  nodetmp[node->parents[k]->x+node->parents[k]->y*map->xdim]++;
					  xset[neighc]= node->parents[k]->x;
					  yset[neighc]= node->parents[k]->y;
					  neighc++;
					  compute=1;
				  }
				  else
					  glb_ivec[offset + node->parents[k]->x/goptx+ node->parents[k]->y/gopty * gopt_numxblocks]++;
			  }
	
			  if (i>prefuzzy && compute) {
				  for (xd = 0; xd < map->xdim; xd++) {
					  for (yd = 0; yd < map->ydim; yd++) {
						  FLOAT maximp = 0;
						  for (idx = 0; idx < neighc; idx++) {
							  FLOAT d = ComputeHexaDistance_CPU(xd, yd, xset[idx], yset[idx]);
							  if (d > radius_t*8)
								  continue;
							  winval = expf(-(d*d)/(2*sigma*sigma))/(sigma*sqrtf(2*PI));
							  FLOAT tmp = expf((d/(-2.0 * r_fuzzy * r_fuzzy)))*nodetmp[xset[idx]+yset[idx]*map->xdim]*winval;
							  if (maximp<tmp)
								  maximp = tmp;
						  }
						  if (nodetmp[xd+yd*map->xdim] < maximp)
							  nodetmp[xd+yd*map->xdim] = maximp;
					  }
				  }
	  
				  neighc = 0;
				  for (xd = 0; xd < map->xdim; xd++)
					  for (yd = 0; yd < map->ydim; yd++)
						  glb_ivec[offset+xd/goptx+yd/gopty*gopt_numxblocks]+= nodetmp[xd+yd*map->xdim];
			  }
		  }
				
		  FindWinnerEucledian_CPU(map, node, gptr, &winner);
		  //printf("%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
		  fprintf(file_res,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
	  }
  }
  fclose(file_res);
  //End Tuc special


	
  free(glb_ivec);
  return 1;
}


/******************************************************************************
 Description: Chooses the right function to trains the various types of SOM
 that are supported by this software.

 Return value: The return value produced by the corresponding training function.
 This would normally be 0 if training could not be initiated, or
 1 otherwise.
******************************************************************************/
int TrainMap_CPU(struct Parameters *parameters)
{
#ifdef _OPENMP
  if (parameters->ncpu > 0)
    omp_set_num_threads(parameters->ncpu);
  fprintf(stderr, "OpenMP CPU mode: using up to %d thread(s) (override with -cpu <N>)\n",
	  omp_get_max_threads());
#endif
  if (parameters->map.type == TYPE_GRAPHSOM) {
    if (parameters->fuzzy){
      if(parameters->memorysave)
    	  return TrainPMGraphSOM_CPU(parameters);
    }
  }else if (parameters->map.type == TYPE_SOMSD)
    return TrainSOMSD_CPU(parameters);
  else TrainSOM_CPU(parameters);
  return 1;
}

/* END OF FILE */

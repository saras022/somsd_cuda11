/*
  Contents: Functions used by all somsd modules (sominit, somsd, somtest).

  Author: Markus Hagenbuchner

  Comments and questions concerning this program package may be sent
  to 'markus@artificial-neural.net'
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
#include <unistd.h>
#include "common.h"
#include "train.h"
#include "utils.h"
#include "data.h"

/* Begin functions... */
/****************************************************************************
 Function: 

 Parameter:

 Result: 
****************************************************************************/
void Hausekeeping(struct Parameters *param)
{
  if (param->flags & PAUSE){
    fprintf(stderr, "\nPaused. Press 'p' to continue or 'X' to quit\n");
    while(param->flags & PAUSE)
      sleep(1);
  }
}

/******************************************************************************
Description: Convert network topology from a string pointed to by cptr to an
             ID number. Store ID number in uval. uval remains unchanged if
             cptr does not hold data.             

Return value: 1 if ID number of a recognized topology was assigned, or
              0 otherwise.
******************************************************************************/
UNSIGNED GetTopologyID(char *cptr, UNSIGNED *uval)
{
  if (cptr == NULL)                       /* No data given, return UNKNOWN */
    return (UNSIGNED)UNKNOWN;

  if (!strncasecmp(cptr, "hex", 3)){      /* Hexagonal topology */
    if (uval)
      *uval = (UNSIGNED)TOPOL_HEXA;
    return TOPOL_HEXA;
  }
  else if (!strncasecmp(cptr, "rect", 4)){ /* Rectangular topology */
    if (uval)
      *uval = (UNSIGNED)TOPOL_RECT;
    return TOPOL_RECT;
  }
  else if (!strncasecmp(cptr, "oct", 3)){ /* Octagonal topology */
    if (uval)
      *uval = (UNSIGNED)TOPOL_OCT;
    return TOPOL_OCT;
  }
  else if (!strncasecmp(cptr, "vq", 2)){  /*Vector quantization (no topology)*/
    if (uval)
      *uval = (UNSIGNED)TOPOL_VQ;
    return TOPOL_VQ;
  }
  else if (!strncasecmp(cptr, "none", 2)){/*Vector quantization (no topology)*/
    if (uval)
      *uval = (UNSIGNED)TOPOL_VQ;
    return TOPOL_VQ;
  }
  else if (uval)
    *uval = (UNSIGNED)UNKNOWN;

  return (UNSIGNED)UNKNOWN;                /* Topology not recognized */
}

/******************************************************************************
Description: Convert network neighborhood from a string pointed to by cptr to
             an ID number. If uval is not NULL, then store the ID number in
             uval. uval remains unchanged if cptr does not hold data.            

Return value: The value UNKNOWN is returned if ID number of a recognized
              topology was not found, otherwise the ID number is returned.
******************************************************************************/
UNSIGNED GetNeighborhoodID(char *cptr, UNSIGNED *uval)
{
  if (cptr == NULL)                        /* No data given, return UNKNOWN */
    return UNKNOWN;

  if (!strncasecmp(cptr, "bubble", 6)){    /* Bubble neighborhood           */
    if (uval)
      *uval = NEIGH_BUBBLE;
    return NEIGH_BUBBLE;
  }
  else if (!strncasecmp(cptr, "gauss", 5)){ /* Gaussian neighborhood        */
    if (uval)
      *uval = NEIGH_GAUSSIAN;
    return NEIGH_GAUSSIAN;
  }
  else if (!strncasecmp(cptr, "none", 4)){  /* No neighborhood (VQ mode)    */
    if (uval)
      *uval = NEIGH_NONE;
    return NEIGH_NONE;
  }
  else if (uval)
    *uval = UNKNOWN;

  return UNKNOWN;                          /* Neighborhood not recognized   */
}

/******************************************************************************
Description: Converts a string which is assumed to specify the type of alpha
             value decrease to a numerical identifier.

Return value: A numerical identifier which indicates the alpha type, or
              UNKNOWN if the string did not hold a recognized keyword.
******************************************************************************/
UNSIGNED GetAlphaType(char *atype)
{
  if (atype == NULL)
    AddError("No parameter for option alpha_type specified\n");
  else if (!strncasecmp(atype, "sigmoid", 7))  /* Sigmoidal decreasing alpha */
    return ALPHA_SIGMOIDAL;
  else if (!strncasecmp(atype, "linear", 6))   /* Linearly decreasing alpha  */
    return ALPHA_LINEAR;
  else if (!strncasecmp(atype, "exponent", 8)) /* Exponential decrease alpha */
    return ALPHA_EXPONENTIAL;
  else if (!strncasecmp(atype, "const", 5))    /* Constant alpha value       */
    return ALPHA_CONSTANT;
  else
    AddError("Unknown type for option alpha_type specified\n");

  return UNKNOWN;
}

/******************************************************************************
Description: Converts a string which is assumed to specify the type of map
             to a unique numerical identifier. If val is not NULL, the ID is
             stored in a location pointed to by val before the same value is
             returned by this function.

Return value: A numerical identifier which indicates the map type, or
              UNKNOWN if the string did not hold a recognized keyword.
******************************************************************************/
UNSIGNED GetMapTypeID(char *mtype, UNSIGNED *val)
{
  if (mtype == NULL)                       /* No data given, return UNKNOWN */
    return (UNSIGNED)TYPE_UNKNOWN;

  if (!strcasecmp(mtype, "som")){               /* Standard SOM      */
    if (val)
      *val = TYPE_SOM;
    return TYPE_SOM;
  }
  else if (!strcasecmp(mtype, "somsd")){        /* SOM-SD            */
    if (val)
      *val = TYPE_SOMSD;
    return TYPE_SOMSD;
  }
   else if (!strncasecmp(mtype, "pmgraphsom", 4)){ /* GraphSOM          */
    if (val)
      *val = TYPE_GRAPHSOM;
    return TYPE_GRAPHSOM;
  }
  else{
    if (val)
      *val = TYPE_UNKNOWN;
    return TYPE_UNKNOWN;
  }
}
/******************************************************************************
Description: Converts a string which is assumed to specify the type of map
             to a unique numerical identifier.

Return value: A numerical identifier which indicates the map type, or
              UNKNOWN if the string did not hold a recognized keyword.
******************************************************************************/
UNSIGNED GetMapType(char *mtype)
{
  UNSIGNED retval;

  if (mtype == NULL)
    AddError("No parameter for option maptype specified\n");

  if ((retval = GetMapTypeID(mtype, NULL)) == TYPE_UNKNOWN)
    AddError("Unknown type for option maptype specified\n");

  return retval;
}

/******************************************************************************
Description: Convert a network topology ID to a corresponding string (i.e.
             an ID value of TOPOL_HEXA will be converted to "hexagonal").

Return value: Pointer to the corresponding topology name, or NULL if the
              ID could not be resolved to a known name.
******************************************************************************/
const char* GetTopologyName(UNSIGNED ID)
{
  if (ID == TOPOL_HEXA)          /* Hexagonal topology    */
    return "hexagonal";
  else if (ID == TOPOL_RECT)     /* Rectangular topology  */
    return "rectagonal";
  else if (ID == TOPOL_OCT)      /* Octangular topology   */
    return "octagonal";
  else if (ID == TOPOL_VQ)       /* VQ mode (no topology) */
    return "vq";
  else
    return NULL;                 /* ID unknown */
}

/******************************************************************************
Description: Convert a network neighborhood ID to a corresponding string (i.e.
             an ID value of NEIGH_BUBBLE will be converted to "bubble").

Return value: Pointer to the corresponding neighborhood name, or NULL if the
              ID could not be resolved to a known name.
******************************************************************************/
const char* GetNeighborhoodName(UNSIGNED ID)
{
  if (ID == NEIGH_BUBBLE)        /* Bubble neighborhood      */
    return "bubble";
  else if (ID == NEIGH_GAUSSIAN) /* Gaussian neighborhood    */
    return "gaussian";
  else if (ID == NEIGH_NONE)     /* No neighborhood (VQ mode)*/
    return "none";
  else
    return NULL;                 /* ID unknown */
}

/******************************************************************************
Description: Converts a maptype-ID into a corresponding string (i.e. the 
             map-ID TYPE_GRAPHSOM is converted to string "GraphSOM".

Return value: Pointer to the corresponding map name, or NULL if the
              ID could not be resolved to a known name.
******************************************************************************/
const char *GetMapTypeName(UNSIGNED ID)
{
  if (ID == TYPE_SOM)             /* Standard SOM       */
    return "som";
  else if (ID == TYPE_SOMSD)      /* SOM-SD             */  
    return "somsd";
  else if (ID == TYPE_CSOMSD)     /* Contextual SOM-SD  */
    return "csomsd";
  else if (ID == TYPE_GRAPHSOM)   /* GraphSOM           */
    return "pmgraphsom";
  else
    return NULL;
}

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
UNSIGNED InitCodes(struct Map *map, struct Graph *data, UNSIGNED mode)
{
  UNSIGNED dim, sdim, noc;
  UNSIGNED offset, offset2;
  int nc;
  UNSIGNED i, x, y;
  FLOAT *maval, *mival;
  FLOAT dist, maxdist;
  struct Graph *gptr;
  struct Node *node;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

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
  if (map->topology == TOPOL_VQ)
    dim = data->ldim + (data->FanOut + data->FanIn) * noc + data->tdim;

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

  if (map->topology == TOPOL_VQ){
    for (gptr = data; gptr != NULL; gptr = gptr->next){
      for (i = 0; i < gptr->numnodes; i++){
	node = gptr->nodes[i];
	for (x = 0; x < gptr->ldim; x++){
	  maval[x] = max(node->points[x], maval[x]);
	  mival[x] = min(node->points[x], mival[x]);
	}
	for (x = 0; x < data->FanOut + data->FanIn; x++){
	  offset = gptr->ldim + x*noc;
	  offset2 = gptr->ldim + x*2;
	  for (nc = 0; nc < noc; nc++){
	    if (nc == (int)node->points[offset2]){
	      maval[offset+nc] = max(1.0, maval[offset+nc]);
	      mival[offset+nc] = min(1.0, mival[offset+nc]);
	    }
	    else{
	      maval[offset+nc] = max(0.0, maval[offset+nc]);
	      mival[offset+nc] = min(0.0, mival[offset+nc]);
	    }
	  }
	}
	for (x = 0; x < gptr->tdim; x++){
	  offset = gptr->ldim + (data->FanOut + data->FanIn) * noc;
	  offset2 = gptr->ldim + (data->FanOut + data->FanIn) * 2;
	  maval[offset+x] = max(node->points[offset2+x], maval[offset+x]);
	  mival[offset+x] = min(node->points[offset2+x], mival[offset+x]);
	}
      }
    }
  }
  else{  /* Normal SOM-SD mode (not in VQ mode) */
    for (gptr = data; gptr != NULL; gptr = gptr->next){
      for (i = 0; i < gptr->numnodes; i++){
	node = gptr->nodes[i];
	for (x = 0; x < dim; x++){
	  maval[x] = max(node->points[x], maval[x]);
	  mival[x] = min(node->points[x], mival[x]);
	}
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

  switch (mode){
  case INIT_LINEAR:
    fprintf(stderr, "\nlinear mode\n");
    /* Linear initialization of vector values */
    ComputeDistance = ComputeHexaDistance;

    nc = 0;
    maxdist = ComputeDistance(map->xdim-1, map->ydim-1, 0, 0);
    for (y = 0u; y < map->ydim; y++){
      for (x = 0u; x < map->xdim; x++){
	dist = ComputeDistance(map->codes[nc].x, map->codes[nc].y, 0, 0);
	dist = sqrtf((FLOAT)dist);
	for (i = 0; i < dim; i++){
	  map->codes[nc].points[i] = mival[i] + ((maval[i] - mival[i]) * dist / maxdist);
	}
	for (; i < map->dim; i++){ /* In VQ mode, map->dim > dim */
	  if (i-dim == nc)
	    map->codes[nc].points[i] = 1.0;
	  else
	    map->codes[nc].points[i] = 0.0;
	}
	nc++;
      }
    }
    break;
  case INIT_RANDOM:
  default:
    //    fprintf(stderr, "\nrandom mode\n");
    /* Randomize the vector values */
    for (i = 0; i < noc; i++){
      for (x = 0; x < dim; x++){
	map->codes[i].points[x] = mival[x] + (maval[x] - mival[x]) * drand48();
      }
      for (; x < map->dim; x++){ /* In VQ mode, map->dim > dim */
	if (x-dim == i)
	  map->codes[i].points[x] = 1.0;
	else
	  map->codes[i].points[x] = 0.0;
      }
    }
    break;
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
UNSIGNED InitGsomCodes(struct Map *map, struct Graph *data, UNSIGNED mode)
{
  UNSIGNED dim, noc;
  int nc, totalnodes;
  int maxout, maxin, maxneighbor;
  UNSIGNED i, x, y;
  UNSIGNED goptx, gopty;
  FLOAT *maval, *mival;
  FLOAT dist, maxdist;
  struct Graph *gptr;
  struct Node *node;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

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

  switch (mode){
  case INIT_LINEAR:
    fprintf(stderr, "\nlinear mode\n");
    /* Linear initialization of vector values */
    ComputeDistance = ComputeHexaDistance;

    nc = 0;
    maxdist = ComputeDistance(map->xdim-1, map->ydim-1, 0, 0);
    for (y = 0u; y < map->ydim; y++){
      for (x = 0u; x < map->xdim; x++){
	dist = ComputeDistance(map->codes[nc].x, map->codes[nc].y, 0, 0);
	dist = sqrtf((FLOAT)dist);
	for (i = 0; i < dim; i++){
	  map->codes[nc].points[i] = mival[i] + ((maval[i] - mival[i]) * dist / maxdist);
	}
	nc++;
      }
    }
    break;
  case INIT_RANDOM:
  default:
    //    fprintf(stderr, "\nrandom mode\n");
    /* Randomize the vector values */
    for (i = 0; i < noc; i++){
      for (x = 0; x < dim; x++){
	map->codes[i].points[x] = mival[x] + (maval[x] - mival[x]) * drand48();
      }
    }
    break;
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
UNSIGNED InitGsomCodesCompact(struct Map *map, struct Graph *data, UNSIGNED mode)
{
  UNSIGNED dim, noc;
  int nc, totalnodes;
  int n,c,k;
  int maxout, maxin, maxneighbor;
  UNSIGNED i, x, y;
  FLOAT *maval, *mival;
  FLOAT dist, maxdist;
  struct Graph *gptr;
  struct Node *node;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

  noc = map->xdim * map->ydim;  /* Total number of codebooks */
  if (noc == 0u){
    AddError("Network dimension is zero!");
    return 0u;
  }
  map->codes = (struct Codebook*)MyCalloc(noc, sizeof(struct Codebook));

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

    for (n = 0; n < gptr->numnodes; n++){
      node = gptr->nodes[n];
      c = 0;

      for (k = 0; k < gptr->FanOut; k++) {
	if (node->children[k] != NULL)
	  c++;
      }
      if (maxneighbor < (c + node->numparents))
	maxneighbor = c+node->numparents;
    }
  }
  dim += 2*maxneighbor;

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

  switch (mode){
  case INIT_LINEAR:
    fprintf(stderr, "\nlinear mode\n");
    /* Linear initialization of vector values */
    ComputeDistance = ComputeHexaDistance;

    nc = 0;
    maxdist = ComputeDistance(map->xdim-1, map->ydim-1, 0, 0);
    for (y = 0u; y < map->ydim; y++){
      for (x = 0u; x < map->xdim; x++){
	dist = ComputeDistance(map->codes[nc].x, map->codes[nc].y, 0, 0);
	dist = sqrtf((FLOAT)dist);
	for (i = 0; i < dim; i++){
	  map->codes[nc].points[i] = mival[i] + ((maval[i] - mival[i]) * dist / maxdist);
	}
	nc++;
      }
    }
    break;
  case INIT_RANDOM:
  default:
    //    fprintf(stderr, "\nrandom mode\n");
    /* Randomize the vector values */
    for (i = 0; i < noc; i++){
      for (x = 0; x < dim; x++){
	map->codes[i].points[x] = mival[x] + (maval[x] - mival[x]) * drand48();
      }
    }
    break;
  }

  free(mival);
  free(maval);

  return 1;
}

/******************************************************************************
Description: This function is called in VQ mode, and will compute the auxillary
             variables a and b.

Return value: This function does not return a value.
******************************************************************************/
void VQSet_ab(struct Parameters *parameters)
{
  UNSIGNED n, i, noc;
  struct Map *map;
  struct Graph *graph;
  FLOAT tmp;

  map = &parameters->map;
  if ((graph = parameters->train) == NULL)
    if ((graph = parameters->valid) == NULL)
      if ((graph = parameters->test) == NULL){
	AddError("Init error in VQSet_ab: No data available.");
	return;
      }

  noc = map->ydim * map->xdim;
  for (n = 0; n < noc; n++){
    tmp = 0.0;
    for (i = 0; i < graph->FanOut * noc; i++)
      tmp += SQR(map->codes[n].points[graph->ldim + i]);
    map->codes[n].a = tmp;
    tmp = 0.0;
    for (i = 0; i < graph->FanIn * noc; i++)
      tmp += SQR(map->codes[n].points[graph->FanOut * noc + graph->ldim + i]);
    map->codes[n].b = tmp;
  }
}

/*****************************************************************************
Description: Build integral on fuzzy neighbor function

Return value: This function return a float value.
*****************************************************************************/
FLOAT BuildIntegral(struct Parameters *params){
    int r = params->radius;
	  FLOAT c = 1/(-2.0 * r * r);
	  FLOAT n=20000;
	  FLOAT v, h,s1, s2;  
	  FLOAT upper = ComputeHexaDistance(params->map.xdim/4, params->map.ydim/4, 0, 0);
	  FLOAT lower = 0;
      int idx;
	  h=(upper-lower)/n;  
	  v=lower;  
	  s1=0;  
	  s2=0;
	   
	  for(idx=1;idx<=n;idx++)  
	     { 
	        v += h;
	        s1 += expf(c*v)*h;  
	     } 
	   
	   upper = ComputeHexaDistance(params->map.xdim/4, params->map.ydim/4, params->map.xdim-1, params->map.ydim-1);
	   lower = 0;

	   h=(upper-lower)/n;  
	   v=lower;  
	   for(idx=1;idx<=n;idx++)  
	       { 
	   	    v += h;
	   	    s2 += expf(c*v)*h;  
	       }  
	   
	   return s1 + s2;
}

/*****************************************************************************
Description: Compute optimal mu-weights

Return value: This function does not return a value.
*****************************************************************************/
void GetMuValues(struct Parameters *params, FLOAT *mu1, FLOAT *mu2, FLOAT *mu3, FLOAT *mu4, FLOAT *initmu1, FLOAT *initmu2)
{
  UNSIGNED ldim, cend, tend, pend;
  UNSIGNED dim, num, connections;
  UNSIGNED i, j, x;
  FLOAT *avg, *sigma;
  FLOAT d1, d2, d3, d4;
  FLOAT x1, x2, x3, x4, k;
  struct Graph *gptr;
  struct Node *node;

  if (params->map.codes == NULL || params->map.dim == 0 || params->train==NULL)
    return;

  if(params->map.topology == TOPOL_VQ){
    fprintf(stderr, "\nWarning: Suggested mu values in VQ mode are incorrect.\n");
    fprintf(stderr, "         Function GetMuValues() not yet adapted to VQ mode.\n");
  }

  dim = params->map.dim;
  avg    = (FLOAT*)MyCalloc(dim, sizeof(FLOAT));
  sigma  = (FLOAT*)MyCalloc(dim, sizeof(FLOAT));

  /* Compute average values */
  ldim = params->train->ldim;
  if (params->map.type == TYPE_GRAPHSOM){
    cend = params->train->dimension - params->train->tdim;
    pend = cend;
     }
  else{
    cend = ldim + 2*params->train->FanOut;
    pend = cend + 2*params->train->FanIn;
      }
  tend = pend + params->train->tdim;

  if(dim != tend){
    AddError("Dimension of codebooks != dimension of train-set vectors");
    *mu1 = 0;
    *mu2 = 0;
    *mu3 = 0;
    *mu4 = 0;
    return;
  }

  num = 0;
  for (gptr = params->train; gptr != NULL; gptr = gptr->next){
    for (i = 0; i < gptr->numnodes; i++){
      node = gptr->nodes[i];
      for (x = 0; x < ldim; x++)    /* Sum all values in node label   */
	avg[x] += node->points[x];
      for (x = pend; x < tend; x++) /* Sum all values in target vector */
	avg[x] += node->points[x];
      num++;
    }
  }
  for (i = 0; i < dim; i++)        /* Compute the average */
    avg[i] = avg[i]/num;
  
  /* Computation on State vector */
  if (params->map.type == TYPE_GRAPHSOM){
    connections = 0; /* Total connectivity */
    for (gptr = params->train; gptr != NULL; gptr = gptr->next){
      for (i = 0; i < gptr->numnodes; i++){
	node = gptr->nodes[i];
	if (node->children != NULL){
	  for (j = 0; node->children[j] != NULL; j++);
	  connections += j;
	}
	connections += node->numparents;
      }
    }
    if (connections > 0){
      for (x = ldim; x < pend; x++)  /* Average of GraphSOM states */
	avg[x] = (float)connections/(float)(num*(pend-ldim));
    }
    else
      for (x = ldim; x < pend; x++)
	avg[x] = 0;
  }
  else{  //If not a GraphSOM then...
    for (x = ldim; x < pend; x+=2){  /* Average of parent and child states */
      avg[x]   = (params->map.xdim-1)/2.0;
      avg[x+1] = (params->map.ydim-1)/2.0;
    }
  }

  /* Compute std.deviation */
  num = 0;
  for (gptr = params->train; gptr != NULL; gptr = gptr->next){
    for (i = 0; i < gptr->numnodes; i++){
      node = gptr->nodes[i];
      for (x = 0; x < ldim; x++)
	sigma[x] += SQR(node->points[x] - avg[x]);
      for (x = pend; x < tend; x++)
	sigma[x] += SQR(node->points[x] - avg[x]);
      num++;
    }
  }
  for (i = 0; i < dim; i++)
    sigma[i] = sigma[i]/num;
  for (x = ldim; x < pend; x++)
    sigma[x] = SQR(avg[x]/2);

  for (d1 = 0.0, x = 0; x < ldim; x++)
    d1 += sigma[x];
  for (d2 = 0.0; x < cend; x++)
    d2 += sigma[x];
  for (d3 = 0.0; x < pend; x++)
    d3 += sigma[x];
  for (d4 = 0.0; x < tend; x++)
    d4 += sigma[x];

  if (ldim > 0){              /* If we have a node label */
    x1 = 1.0;
    x2 = (d2 > 0.0) ? d1/d2 : 0.0;
    x3 = (d3 > 0.0) ? d1/d3 : 0.0;
    x4 = (d4 > 0.0) ? d1/d4 : 0.0;
    k = 1/(x1+x2+x3+x4); /* Normalizing factor */
  }
  else if (cend - ldim > 0){  /* No node label but child states */
    x1 = 0.0;
    x2 = 1.0;
    x3 = (d3 > 0.0) ? d2/d3 : 0.0;
    x4 = (d4 > 0.0) ? d2/d4 : 0.0;
    k = 1/(x2+x3+x4);    /* Normalizing factor */
  }
  else if (pend - cend > 0){  /* No label or child states but parent states */
    x1 = 0.0;
    x2 = 0.0;
    x3 = 1.0;
    x4 = (d4 > 0.0) ? d3/d4 : 0.0;
    k = 1/(x3+x4);       /* Normalizing factor */
  }
  else if (tend - pend > 0){  /* Only have target vector */
    x1 = 0.0;
    x2 = 0.0;
    x3 = 0.0;
    x4 = 1.0;
    k = 1.0;
  }
  else{                       /* Node dimension is empty */
    x1 = 0.0;
    x2 = 0.0;
    x3 = 0.0;
    x4 = 0.0;
    k = 0.0;
  }

  *mu1 = k*x1;
  *mu2 = k*x2;
  *mu3 = k*x3;
  *mu4 = k*x4;
  *initmu1 = *mu1;
  *initmu2 = *mu2;
  
  free(avg);
  free(sigma);
}

/*****************************************************************************
Description: Get number of unique graph structures of the input data

Return value: This function does not return a value.
*****************************************************************************/
void GetUniqueStruct(struct Parameters *params){
  struct Graph *gptr;
  struct Node *node;
  int i;
	  
  if (params->map.codes == NULL || params->map.dim == 0 || params->train==NULL)
    return;
	     
  for (gptr = params->train; gptr != NULL; gptr = gptr->next){
    fprintf(stderr, "%s\n", gptr->gname);
    for (i = 0; i < gptr->numnodes; i++){
      node = gptr->nodes[i];	   
      fprintf(stderr, "%d\n", i);
      fprintf(stderr, "%s\n--------\n", GetStructure(node, node, gptr->FanOut, 0));
    }
  }
}

char* GetStructure(struct Node *node, struct Node *ndtmp, int fout, int flag)
{
  int k=0;
  const char *initstr = "0";
  char *graphstr;
  graphstr = (char *)calloc(100, sizeof(char));
  strcat(graphstr, initstr);  
  //fprintf(stderr, "%d\n", node->numparents);
  //fprintf(stderr, "%s - begin\n", graphstr);
  struct Node **nodelist;
  int idx = 0;

  fprintf(stderr, "ERROR: In function GetStructure()\n");
  fprintf(stderr, "This function is incomplete (it will segfault)\n");
  fprintf(stderr, "ABORTING before segfaulting\n");
  exit(0);

  for (k = 0; k < node->numparents; k++){
    //fprintf(stderr, "\n%d\n", k);
    if(node->parents[k]!=ndtmp && !flag){
      strcat(graphstr, "(");
      strcat(graphstr, GetStructure(node->parents[k], node, fout, 0)); 
      strcat(graphstr, ")");  
      //fprintf(stderr, "%s - parents\n", graphstr);
    }
    else if(node->parents[k]!=ndtmp && flag){
      strcat(graphstr, "0");
      nodelist[idx] = node->parents[k];
      idx++;
    }
  }
 
  //fprintf(stderr, "%s - parents\n", graphstr);
     
  for (k = 0; k < fout; k++){
    if (node->children[k] != NULL){
      if(node->children[k]!=ndtmp){
	strcat(graphstr, "(");
	strcat(graphstr, GetStructure(node->children[k], node, fout, 1)); 
	strcat(graphstr, ")"); 
      }
    }
    else
      break;
  }
  //fprintf(stderr, "%s - child\n", graphstr);
  //fprintf(stderr, "%s\n", graphstr);
  return graphstr;		         
}

/*****************************************************************************
Description: Suggest optimal mu-weights. Suggestions are printed to stderr.

Return value: This function does not return a value.
*****************************************************************************/
void SuggestMu(struct Parameters *params)
{
  FLOAT mu1, mu2, mu3, mu4, initmu1, initmu2;
  char buffer[128];

  AddMessage(PRT_WARNING "Incomplete implementation of mu vectors.");
  AddMessage("         You should set mu values manually!");
  //  return;

  if (params->nmu1 == 0){
    params->mu1 = (float*) calloc(1, sizeof(FLOAT));
    params->nmu1 = 1;
  }
  if (params->nmu2 == 0){
    params->mu2 = (float*) calloc(1, sizeof(FLOAT));
    params->nmu2 = 1;
  }
  if (params->nmu3 == 0){
    params->mu3 = (float*) calloc(1, sizeof(FLOAT));
    params->nmu3 = 1;
  }
  if (params->nmu4 == 0){
    params->mu4 = (float*) calloc(1, sizeof(FLOAT));
    params->nmu4 = 1;
  }

  //  mu1 = fsum(params.nmu1, params.mu1);
  //  mu2 = fsum(params.nmu2, params.mu2);
  //  mu3 = fsum(params.nmu3, params.mu3);
  //  mu4 = fsum(params.nmu4, params.mu4);

  GetMuValues(params, &mu1, &mu2, &mu3, &mu4, &initmu1, &initmu2);
     
  if (similar(*params->mu1 + *params->mu2 + *params->mu3 + *params->mu4, 0.0, 4.0*EPSILON)){
    //ToDo: below, make sure memory is (re-)allocated
    *(params->mu1) = mu1;
    *params->mu2 = mu2;
    *params->mu3 = mu3;
    *params->mu4 = mu4;
    params->initmu1 = initmu1;
    params->initmu2 = initmu2;
    strcpy(buffer, "         Will use:");

    params->mu1[0] = mu1;
    params->mu2[0] = mu2;
    params->mu3[0] = mu3;
    params->mu4[0] = mu4;
    if (mu1 > 0.00001)
      sprintf(&buffer[strlen(buffer)], " -mu1 %.9f", mu1);
    else if (mu1 > 0.0)
      sprintf(&buffer[strlen(buffer)], " -mu1 %.9e", mu1);
    if (mu2 > 0.00001)
      sprintf(&buffer[strlen(buffer)], " -mu2 %.9f", mu2);
    else if (mu4 > 0.0)
      sprintf(&buffer[strlen(buffer)], " -mu2 %.9e", mu2);
    if (mu3 > 0.00001)
      sprintf(&buffer[strlen(buffer)], " -mu3 %.9f", mu3);
    else if (mu4 > 0.0)
      sprintf(&buffer[strlen(buffer)], " -mu3 %.9e", mu3);
    if (mu4 > 0.00001)
      sprintf(&buffer[strlen(buffer)], " -mu4 %.9f", mu4);
    else if (mu4 > 0.0)
      sprintf(&buffer[strlen(buffer)], " -mu4 %.9e", mu4);
    if (mu1+mu2+mu3+mu4 <= 0.00001)
      AddError("All \\mu weights are zero!");
    else
      AddMessage(buffer);
  }
  else{
    if (!approx(*params->mu1, mu1, 0.01) || !approx(*params->mu2, mu2, 0.01) || !approx(*params->mu3, mu3, 0.01) || !approx(*params->mu4, mu4, 0.01)){
      AddMessage("Caution: The mu-values are not optimal.");
      if (mu1+mu2+mu3+mu4 >= 0.00001)
	     strcpy(buffer, "Suggesting use of:");

      if (initmu1 > 0.00001)
	     sprintf(&buffer[strlen(buffer)], " -initmu1 %1.9f", initmu1);
      else if (initmu1 > 0.0)
	     sprintf(&buffer[strlen(buffer)], " -initmu1 %.9e", initmu1);
      if (initmu2 > 0.00001)
	     sprintf(&buffer[strlen(buffer)], " -initmu2 %1.9f", initmu2);
      else if (initmu2 > 0.0)
	     sprintf(&buffer[strlen(buffer)], " -initmu2 %.9e", initmu2);
      if (mu1 > 0.00001)
	     sprintf(&buffer[strlen(buffer)], " -mu1 %1.9f", mu1);
      else if (mu1 > 0.0)
	     sprintf(&buffer[strlen(buffer)], " -mu1 %.9e", mu1);
      if (mu2 > 0.00001)
	     sprintf(&buffer[strlen(buffer)], " -mu2 %1.9f", mu2);
      else if (mu2 > 0.0)
	     sprintf(&buffer[strlen(buffer)], " -mu2 %.9e", mu2);
      if (mu3 > 0.00001)
	     sprintf(&buffer[strlen(buffer)], " -mu3 %1.9f", mu3);
      else if (mu3 > 0.0)
	     sprintf(&buffer[strlen(buffer)], " -mu3 %.9e", mu3);
      if (mu4 > 0.00001)
	     sprintf(&buffer[strlen(buffer)], " -mu4 %1.9f", mu4);
      else if (mu4 > 0.0)
	     sprintf(&buffer[strlen(buffer)], " -mu4 %.9e", mu4);
      AddMessage(buffer);
    }
  }
  //GetUniqueStruct(params);
}

/*****************************************************************************
Description: Print an overview of essential network information in a compact
             and formatted fashion.

Return value: This function does not return a value.
*****************************************************************************/
/*void PrintNetworkInfo(struct Parameters *params, FILE *ofile)
{
  fprintf(stderr, "Network configurations:\n");
  fprintf(stderr, "\tType: %s\n", GetMapTypeName(params->map.type));
  fprintf(stderr, "\tSize: %ux%u\n", params->map.xdim, params->map.ydim);
  fprintf(stderr, "\tNeighborhood: %s\n", GetNeighborhoodName(params->map.neighborhood));
  fprintf(stderr, "\tTopology: %s\n", GetTopologyName(params->map.topology));
  fprintf(stderr, "\tCodebook dimension: %u\n", params->map.dim);

  if (params->map.type == TYPE_GRAPHSOM)
    fprintf(stderr, "\tGopt: %ux%u\n", params->map.goptx, params->map.gopty);
  if(params->fuzzy)
	   fprintf(stderr, "Using fuzzy map: Ratio of mu1/mu2 started from %1.9f to %1.9f\n", params->initmu1/params->initmu2, params->mu1/params->mu2); 
  else
    fprintf(stderr, "\t(mu1, mu2, mu3, mu4) = (%e, %e, %e, %e)\n", params->mu1, params->mu2, params->mu3, params->mu4);
  fprintf(stderr, "\tRadius(t=0): %d\n", params->radius);
  fprintf(stderr, "\tAlpha(t=0): %lf\n", params->alpha);
  fprintf(stderr, "\tSeed: %d\n", params->seed);

  if (params->rlen-params->map.iter == 0)
    fprintf(stderr, "\tMap has been fully trained for %u iterations.\n", params->map.iter);
  else if (params->map.iter == 0){
    fprintf(stderr, "\tUntrained map.");
    if (params->rlen)
      fprintf(stderr, " %u iterations remaining.", params->rlen);
    fprintf(stderr, "\n");
  }
  else
    fprintf(stderr, "\tMap is partially trained. Remaining iterations: %u\n", params->rlen-params->map.iter);
  if (params->datafile)
    fprintf(stderr, "\tTraining set: %s\n", params->datafile);
  if (params->testfile)
    fprintf(stderr, "\tTest set: %s\n", params->testfile);
  if (params->validfile)
    fprintf(stderr, "\tValidation set: %s\n", params->validfile);
  //if (params->logfile)
    //fprintf(stderr, "\tLog file: %s\n", params->logfile);
}*/

void PrintNetworkInfo(struct Parameters *params, FILE *ofile)
{
  fprintf(ofile, "Network configurations:\n");
  fprintf(ofile, "\tType: %s\n", GetMapTypeName(params->map.type));
  fprintf(ofile, "\tSize: %ux%u\n", params->map.xdim, params->map.ydim);
  fprintf(ofile, "\tNeighborhood: %s\n", GetNeighborhoodName(params->map.neighborhood));
  fprintf(ofile, "\tTopology: %s\n", GetTopologyName(params->map.topology));
  fprintf(ofile, "\tCodebook dimension: %u\n", params->map.dim);

  if (params->map.type == TYPE_GRAPHSOM)
    fprintf(ofile, "\tGopt: %ux%u\n", params->map.goptx, params->map.gopty);
  if(params->fuzzy){
    //ToDo: below, Tread mu1 as vector
    fprintf(ofile, "Using fuzzy map: Ratio of mu1/mu2 started from %1.9f to %1.9f\n", params->initmu1/params->initmu2, (*params->mu1)/(*params->mu2));
  }
  else{
    //ToDo: below, Print vector of mu?
    fprintf(ofile, "\t(mu1, mu2, mu3, mu4) = (%e, %e, %e, %e)\n", *params->mu1, *params->mu2, *params->mu3, *params->mu4);
  }
  fprintf(ofile, "\tRadius(t=0): %d\n", params->radius);
  fprintf(ofile, "\tAlpha(t=0): %lf\n", params->alpha);
  fprintf(ofile, "\tSeed: %d\n", params->seed);

  if (params->rlen-params->map.iter == 0)
    fprintf(ofile, "\tMap has been fully trained for %u iterations.\n", params->map.iter);
  else if (params->map.iter == 0){
    fprintf(ofile, "\tUntrained map.");
    if (params->rlen)
      fprintf(ofile, " %u iterations remaining.", params->rlen);
    fprintf(ofile, "\n");
  }
  else
    fprintf(ofile, "\tMap is partially trained. Remaining iterations: %u\n", params->rlen-params->map.iter);
  if (params->datafile)
    fprintf(ofile, "\tTraining set: %s\n", params->datafile);
  if (params->testfile)
    fprintf(ofile, "\tTest set: %s\n", params->testfile);
  if (params->validfile)
    fprintf(ofile, "\tValidation set: %s\n", params->validfile);
  //if (params->logfile)
    //fprintf(stderr, "\tLog file: %s\n", params->logfile);
}

/*
  Contents: Routines used to train the som-sd.

  Author: Markus Hagenbuchner

  Comments and questions concerning this program package may be sent
  to 'markus@artificial-neural.net'


  ChangeLog:
  11/12/2008:
  -Added training roution for PMGraphSOM and PMGraphSOM_Efficiency
  02/06/2008:
  - Added a training routine for GraphSOM.
  22/08/2007:
  - Speed improvement: Approximate computation of winner neuron using the
  Eucledian distance meassure by probing local neurons, then scanning the
  region which had the smallest error. Speed gain is about 5%. This
  function is invoced *only* if the APPROXIMATION flag is set at compile
  time.
  - Speed improvement: Restrict update of neurons to within 10 times the
  neighborhood radius. Speed gain is approx 80%. This approximation is
  activated *only* if the APPROXIMATION flag is set at compile time.
  - Minor speed improvement: Computation of tradius in TrainMap() improved.
  03/10/2006:
  - BugFix: File name for interrupted training runs was not unique if several
  processes terminated within one second.
  30/03/2006:
  - Bug removed: ComputeHexaDistance(.) now computes the correct distance.
  - Speed improvement: The Distance(.) functions now return a squared value
  which results in a speed improvement of about 4-8% depending on which
  Distance(.) function is used.
  - Speed improvement: Forced inlining of function AdaptVector(.) resulted in
  a speed gain of approx. 8%.
  20/05/2005:
  - Bug removed: Confusion of topology and neighborhood in TrainMap(.)

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
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include "common.h"
#include "data.h"
#include "fileio.h"
//#include "system.h"
#include "train.h"
#include "utils.h"
//#include "testsom.h"


//#define ONE_CLASS_ONLY "-1"

/********************/
/* Global variables */
/********************/
int _save_then_exit_ = 0; /* Indicator for interrupt caught */
FLOAT* glb_ivec;
FLOAT  glb_beta = 0.0;

char *fileresult;
/* Begin functions... */

/******************************************************************************
 Description: Function is used to compute the current alpha value (the
 learning rate) for the current training iteration, given the
 maximum number of training iterations, and given initial value
 of alpha at iter=0, and assuming a linearly decrease of alpha.

 Return value: The alpha value for the given itaration assuming a linearly
 decreasing learning rate.
******************************************************************************/
FLOAT LinearDecrease(UNSIGNED iter, UNSIGNED length, FLOAT alphastart)
{
  return (alphastart * (FLOAT) (length - iter) / (FLOAT) length);
}

/******************************************************************************
 Description: Function the current alpha value. Other parameters are ignored.
 It is a dummy function which is required to maintain compatibility
 with other alpha functions.

 Return value: The alpha value for the given itaration which is assumed to
 remain unchanged.
******************************************************************************/
FLOAT ConstantAlpha(UNSIGNED iter, UNSIGNED length, FLOAT alphastart)
{
  return alphastart;
}

/******************************************************************************
 Description: Function is used to compute the current alpha value (the
 learning rate) for the current training iteration, given the
 maximum number of training iterations, and given initial value
 of alpha at iter=0, and assuming an exponential decrease of alpha.
 The Contant INV_ALPHA_CONSTANT specifies the "steepness" of the
 decrease, and the lower limit of alpha.

 Return value: The alpha value for the given itaration assuming a exponential
 decreasing learning rate.
******************************************************************************/
#define INV_ALPHA_CONSTANT 100.0
FLOAT ExponentialAlpha(UNSIGNED iter, UNSIGNED length, FLOAT alphastart) {
  FLOAT c;

  c = length / INV_ALPHA_CONSTANT;

  return (alphastart * c / (c + iter));
}

/******************************************************************************
 Description: Function is used to compute the current alpha value (the
 learning rate) for the current training iteration, given the
 maximum number of training iterations, and given initial value
 of alpha at iter=0, and assuming an sigmoidal decrease of alpha.
 The constant GAUSS_ALPHA_CONSTANT specifies the "steepness" of the
 decrease, and influences the lower limit of alpha.

 Return value: The alpha value for the given itaration assuming a sigmoidal
 decreasing learning rate.
******************************************************************************/
#define GAUSS_ALPHA_CONSTANT 4.0
float SigmoidalAlpha(UNSIGNED iter, UNSIGNED length, FLOAT alphastart) {
  return alphastart*exp(-((FLOAT)iter*iter*GAUSS_ALPHA_CONSTANT)
			/((FLOAT)length*length));
}

/******************************************************************************
 Description: Compute the geographical distance between two codebook coordinates
 given a rectagonal neighborhood relationship.

 Return value: The squared!! Eucledian distance between two points.
******************************************************************************/
FLOAT ComputeRectDistance(int bx, int by, int tx, int ty)
{
  FLOAT ret, diff;

  diff = bx - tx;
  ret = diff * diff;
  diff = by - ty;
  ret += diff * diff;

  return ret;
  //  return (FLOAT)sqrtf(ret);
}

/******************************************************************************
 Description: Compute the geographical distance between two codebook coordinates
 given a hexagonal neighborhood relationship.

 Return value: The squared!!  eucledian distance between two points.
******************************************************************************/
FLOAT ComputeHexaDistance(int bx, int by, int tx, int ty)
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
 Description: Compute the geographical distance between two codebook coordinates
 given a octagonal neighborhood relationship.

 Return value: The squared distance between two points.
******************************************************************************/
FLOAT ComputeOctDistance(int bx, int by, int tx, int ty)
{
  FLOAT ret;

  ret = (FLOAT)max(abs(bx-tx), abs(by-ty));

  return ret*ret;
}

/******************************************************************************
 Description: move a codebook vector towards another vector

 Return value: 
******************************************************************************/
void AdaptVector(FLOAT *codebook, FLOAT *sample, UNSIGNED dim,
			FLOAT alpha) {
  UNSIGNED i;

  for (i = 0; i < dim; i++) {
    //fprintf(stderr, "%f : %f\n", sample[i], codebook[i]);
    codebook[i] += alpha * (sample[i] - codebook[i]);
  }
}

/******************************************************************************
 Description: Find best matching codebook using the Eucledian distance meassure.

 Return value: The best matching codebook is returned to parameter "winner".
******************************************************************************/
void FindWinnerEucledian(struct Map *map, struct Node *node,
			 struct Graph *gptr, struct Winner *winner) {
  FLOAT *mu;
  UNSIGNED vdim;
  UNSIGNED noc;
  /* Number of codebooks in the map */
  FLOAT *codebook, *sample;
  UNSIGNED n, i;
  FLOAT diffsf, diff, difference;
  int cnt = 0;

  mu = node->mu;
  diffsf = FLT_MAX;
  sample = node->points;
  vdim = gptr->dimension;
  noc = map->xdim * map->ydim;
 
  for (n = 0; n < noc; n++) { /* For every codebook of the map */
    codebook = map->codes[n].points;
    difference = 0.0;

#ifdef EXCLUSIVEMAPPING
    if (map->activation[n] > 0.0){
      /* Compute the difference between codebook and input entry */
      for (i = 0; i < vdim; i++) {
	diff = codebook[i] - sample[i];

	difference += diff * diff * mu[i];
	if (difference >= map->activation[n]) {
	  break;
	}
      }

      /* If distance is smaller than previous distances */
      if (difference < map->activation[n]) {
	cnt++;
	//	fprintf(stderr, "A: %d %d %d\n", cnt, node->nnum, n);
	winner->codeno = n;
	diffsf = difference;
      }
    }else{
#endif
    /* Compute the difference between codebook and input entry */
    for (i = 0; i < vdim; i++) {
      diff = codebook[i] - sample[i];

      difference += diff * diff * mu[i];
      if (difference > diffsf) {
	break;
      }
    }

    /* If distance is smaller than previous distances */
    if (difference < diffsf) {
      cnt++;
      //      fprintf(stderr, "N: %d %d %d\n", cnt, node->nnum, n);
      winner->codeno = n;
      diffsf = difference;
    }
#ifdef EXCLUSIVEMAPPING
    }
#endif
  }

#ifdef EXCLUSIVEMAPPING
  if (cnt == 0){
    for (n = 0; n < noc; n++) { /* For every codebook of the map */
      if (map->activation[n] > 0.0)
	cnt++;
    }
    fprintf(stderr, "DEBUG: %d %d %f %f %d\n", cnt,noc,diffsf, difference, node->nnum);
    fprintf(stderr, "ERROR: Trained in EXCLUSIVE mode on a map containing fewer codebooks than number of nodes in the dataset\n");
    exit(0);
  }
  map->activation[winner->codeno] = diffsf;
#endif


  //fprintf(stderr, "%d - %d\n", cnt, winner->codeno);
  winner->diff = diffsf;
  return;
}

/******************************************************************************
 Description: Find best matching codebook using the Eucledian distance meassure.

 Return value: The best matching codebook is returned to parameter "winner".
******************************************************************************/
void FindWinnerEucledianMemorySave(struct Map *map, struct Node *node,struct Graph *gptr, struct Winner *winner) {
  FLOAT *mu;
  UNSIGNED vdim;
  UNSIGNED noc;
  /* Number of codebooks in the map */
  FLOAT *codebook, *sample;
  UNSIGNED n, i;
  FLOAT diffsf, diff, difference;

  mu = node->mu;
  diffsf = FLT_MAX;
  sample = glb_ivec;
  vdim = gptr->dimension;
  noc = map->xdim * map->ydim;
  int cnt = 0;
  //fprintf(stderr, "node = %d\n", node->nnum);

  /*
    printf("\nInput vector of dimension %d or node %d:\n",vdim,gptr->ldim);
  for (i = 0; i < vdim; i++) {
	  printf("%f ",glb_ivec[i]);
  }
  for (n = 0; n < noc; n++){
	  printf("\nCodebook vector %d:\n",n);
	  for (i = 0; i < vdim; i++) {
		  printf("%f ",map->codes[n].points[i]);
	  }
  }*/
  //fprintf(stderr, "\n%1.9f\n", mu[0]);
  for (n = 0; n < noc; n++) { /* For every codebook of the map */
    codebook = map->codes[n].points;
    difference = 0.0;
		
    /* Compute the difference between codebook and input entry */
    for (i = 0; i < vdim; i++) {
      if(mu[i]==0)
	continue;
      diff = codebook[i] - sample[i];

      difference += diff * diff * mu[i];
      //if(i==0)
      //fprintf(stderr, "mu = %1.9f\n", mu[i]);
      if (difference > diffsf) {
	break;
      }
    }

    /* If distance is smaller than previous distances */
    if (difference < diffsf) {
      cnt++;
      winner->codeno = n;
      diffsf = difference;
    }
  }
  //fprintf(stderr, "%d - %d\n", cnt, winner->codeno);
  winner->diff = diffsf;
  return;
}


/******************************************************************************
 Description: Find best matching codebook using the Eucledian distance meassure.
 This function approximates the search by computing the best match
 on a 2-grid, then to probe around the best grid point. This
 approximation results in about 5% speed improvement. The quality
 of the approximation improves with the training of the map, and
 hence, it is best invoked after some initial training.

 Return value: The best matching codebook is returned to parameter "winner".
******************************************************************************/
void FindWinnerEucledianApprox(struct Map *map, struct Node *node,
			       struct Graph *gptr, struct Winner *winner) {
  FLOAT *mu;
  UNSIGNED vdim;
  FLOAT *codebook, *sample;
  UNSIGNED n, noff, x, y, i;
  UNSIGNED bx, by;
  FLOAT diffsf, diff, difference;
  int ssx, sex, ssy, sey;

  vdim = gptr->dimension;
  mu = node->mu;
  diffsf = FLT_MAX;
  sample = node->points;
  bx = by = n = 0;
  for (y = 0; y < map->ydim; y+=2) { //check every second neuron
    noff = y * map->xdim;
    for (x = 0; x < map->xdim; x+=2) {
      n = noff + x;
      codebook = map->codes[n].points;
      difference = 0.0;

      /* Compute the difference between codebook and input entry */
      for (i = 0; i < vdim; i++) {
	diff = codebook[i] - sample[i];
	difference += diff * diff * mu[i];
	if (difference > diffsf)
	  break;
      }
      /* If distance is smaller than previous distances */
      if (difference < diffsf) {
	bx = x;
	by = y;
	winner->codeno = n;
	diffsf = difference;
      }
    }
  }

  if (bx > 0)
    ssx = -1;
  else
    ssx = 0;
  if (bx < map->xdim-1)
    sex = 1;
  else
    sex = 0;

  if (by > 0)
    ssy = -1;
  else
    ssy = 0;
  if (by < map->ydim-1)
    sey = 1;
  else
    sey = 0;

  for (y = by+ssy; y <= by+sey; y++) { //Probe around the winner
    noff = y * map->xdim;
    for (x = bx+ssx; x <= bx+sex; x++) {
      n = noff + x;
      codebook = map->codes[n].points;
      difference = 0.0;

      /* Compute the difference between codebook and input entry */
      for (i = 0; i < vdim; i++) {
	diff = codebook[i] - sample[i];
	difference += diff * diff * mu[i];
	if (difference > diffsf)
	  break;
      }
      /* If distance is smaller than previous distances */
      if (difference < diffsf) {
	winner->codeno = n;
	diffsf = difference;
      }
    }
  }

  winner->diff = diffsf;

  return;
}

/******************************************************************************
 Description: 

 Return value: 
******************************************************************************/
void VQFindWinnerEucledian(struct Map *map, struct Node *node,
			   struct Graph *gptr, struct Winner *winner) {
  FLOAT *mu;
  UNSIGNED ldim, fanout, fanin, tend;
  UNSIGNED noc;
  /* Number of codebooks in the map */
  FLOAT *codebook, *sample;
  UNSIGNED n, i;
  int id;

  FLOAT diffsf, diff, difference;

  ldim = gptr->ldim; /* Offset for label component         */
  fanout = gptr->FanOut; /* Offset for child state component   */
  fanin = gptr->FanIn; /* Offset for parent state component  */
  tend = ldim+2*(fanin+fanout)+gptr->tdim; /* End of target vector component */

  mu = node->mu;

  noc = map->xdim * map->ydim;
  diffsf = FLT_MAX;
  sample = node->points;
  for (n = 0; n < noc; n++) { /* For every codebook of the map */
    codebook = map->codes[n].points;
    difference = 0.0;

    /* Compute the difference between codebook and input entry label */
    for (i = 0; i < ldim; i++) {
      diff = codebook[i] - sample[i];
      difference += diff * diff * mu[i];
      if (difference >= diffsf)
	goto big_difference;
    }

    /* Consider children coordinate vector */
    diff = map->codes[n].a;
    for (i = 0; i < fanout; i++) {
      id = (int)sample[ldim + i*2];
      if (id >= 0)
	diff += (1.0 - 2 * codebook[ldim+noc*i+id]);
    }
    difference += diff * mu[i-1];
    if (difference >= diffsf)
      goto big_difference;

    /* Difference to parent coordinate vector */
    diff = map->codes[n].b;
    for (i = 0; i < fanin; i++) {
      id = (int)sample[ldim + 2+ fanout + i*2];
      if (id >= 0)
	diff += (1.0 - 2 * codebook[ldim+noc*fanout+noc*i+id]);
    }
    difference += diff * mu[i-1];
    if (difference >= diffsf)
      goto big_difference;

    /* Difference to target vector component */
    for (i = ldim + 2*fanin + 2*fanout; i < tend; i++) {
      diff = codebook[i] - sample[i];
      difference += diff * diff * mu[i];
      if (difference >= diffsf)
	goto big_difference;
    }

    /* Distance is smaller than previous distances */
    winner->codeno = n;
    diffsf = difference;
  big_difference: continue;
  }
  winner->diff = diffsf;

  return;
}

/******************************************************************************
 Description: Adapt all codebook vectors which are located within a fixed
 radius around the winning codebook.

 Return value: none
******************************************************************************/
void BubbleAdapt(struct Graph *gptr, struct Map *map, struct Node *node,
		 struct Winner *winner, FLOAT radius, FLOAT alpha) {
  UNSIGNED n, noc;
  FLOAT dist;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

  ComputeDistance = ComputeHexaDistance;

  noc = map->xdim * map->ydim;
  node->x = map->codes[winner->codeno].x;
  node->y = map->codes[winner->codeno].y;
  radius *= radius; /* Distance computation is squared, thus square radius */
  for (n = 0; n < noc; n++) { /* For every codebook of the map */

    /* Compute distance to winner */
    dist = ComputeDistance(node->x, node->y, map->codes[n].x,
			   map->codes[n].y);
    if (dist <= radius)
      AdaptVector(map->codes[n].points, node->points, map->dim, alpha);/*Update step*/
  }
}

/******************************************************************************
 Description: Adapt all codebook vectors assuming a gaussian neighborhood
 relationship between the codebooks.

 Return value: none
******************************************************************************/
void GaussianAdapt(struct Graph *gptr, struct Map *map, struct Node *node,
		   struct Winner *winner, FLOAT radius, FLOAT alpha) {
  UNSIGNED n, noc;
  //UNSIGNED off;
  FLOAT dist;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

  ComputeDistance = ComputeHexaDistance;

  noc = map->xdim * map->ydim;
  node->x = map->codes[winner->codeno].x;
  node->y = map->codes[winner->codeno].y;
  //fprintf(stderr, "\n%d:%d\n", node->x, node->y);
  radius *= radius;

  for (n = 0; n < noc; n++) { /* For every codebook of the map */

    /* Compute distance to winner */
    dist = ComputeDistance(node->x, node->y, map->codes[n].x, map->codes[n].y);

#ifdef APPROXIMATION
    if (dist > radius*8)
      continue;
#endif

    /*
      adapt(map->codes[n].points, node->points, gptr->ldim, node->mu1*alpha * expf((dist/(-2.0 * radius))));
      off = gptr->ldim;
      adapt(&map->codes[n].points[off], &node->points[off], 2*gptr->FanOut, node->mu2*alpha * expf((dist/(-2.0 * radius))));
      off += 2*gptr->FanOut;
      adapt(&map->codes[n].points[off], &node->points[off], 2*gptr->FanIn, node->mu3*alpha * expf((dist/(-2.0 * radius))));
      off += 2*gptr->FanIn;
      adapt(&map->codes[n].points[off], &node->points[off], gptr->tdim, node->mu4*alpha * expf((dist/(-2.0 * radius))));
    */
    /* Update the codebook */

    AdaptVector(map->codes[n].points, node->points, map->dim, alpha
		* expf((dist/(-2.0 * radius))));
  }
}

/******************************************************************************
 Description: Adapt all codebook vectors assuming a gaussian neighborhood
 relationship between the codebooks.

 Return value: none
******************************************************************************/
void GaussianAdaptSupervised(struct Graph *gptr, struct Map *map, struct Node *node, struct Winner *winner, FLOAT radius, FLOAT alpha)
{
  UNSIGNED n, noc;
  //UNSIGNED off;
  FLOAT dist;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

  ComputeDistance = ComputeHexaDistance;

  noc = map->xdim * map->ydim;
  node->x = map->codes[winner->codeno].x;
  node->y = map->codes[winner->codeno].y;
  //fprintf(stderr, "\n%d:%d\n", node->x, node->y);
  radius *= radius;

  for (n = 0; n < noc; n++) { /* For every codebook of the map */

    /* Compute distance to winner */
    dist = ComputeDistance(node->x, node->y, map->codes[n].x, map->codes[n].y);

#ifdef APPROXIMATION
    if (dist > radius*8)
      continue;
#endif

    /*
      adapt(map->codes[n].points, node->points, gptr->ldim, node->mu1*alpha * expf((dist/(-2.0 * radius))));
      off = gptr->ldim;
      adapt(&map->codes[n].points[off], &node->points[off], 2*gptr->FanOut, node->mu2*alpha * expf((dist/(-2.0 * radius))));
      off += 2*gptr->FanOut;
      adapt(&map->codes[n].points[off], &node->points[off], 2*gptr->FanIn, node->mu3*alpha * expf((dist/(-2.0 * radius))));
      off += 2*gptr->FanIn;
      adapt(&map->codes[n].points[off], &node->points[off], gptr->tdim, node->mu4*alpha * expf((dist/(-2.0 * radius))));
    */
    /* Update the codebook */

    if (map->codes[n].label == 0 || node->label == map->codes[n].label || dist > radius)
      AdaptVector(map->codes[n].points, node->points, map->dim, alpha * expf((dist/(-2.0 * radius))));
    else
      AdaptVector(map->codes[n].points, node->points, map->dim, -alpha * glb_beta * expf(((dist)/(-2.0 * radius))));
  }
}

/******************************************************************************
 Description: Adapt all codebook vectors assuming a gaussian neighborhood
 relationship between the codebooks.

 Return value: none
******************************************************************************/
void GaussianAdaptMemSave(struct Graph *gptr, struct Map *map, struct Node *node,
			  struct Winner *winner, FLOAT radius, FLOAT alpha) {
  UNSIGNED n, noc;
  //UNSIGNED off;
  FLOAT dist;
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

  ComputeDistance = ComputeHexaDistance;

  noc = map->xdim * map->ydim;
  node->x = map->codes[winner->codeno].x;
  node->y = map->codes[winner->codeno].y;
  //fprintf(stderr, "\n%d:%d\n", node->x, node->y);
  radius *= radius;

  for (n = 0; n < noc; n++) { /* For every codebook of the map */

    /* Compute distance to winner */
    dist = ComputeDistance(node->x, node->y, map->codes[n].x,
			   map->codes[n].y);

#ifdef APPROXIMATION
    if (dist > radius*8)
      continue;
#endif

    /*
      adapt(map->codes[n].points, node->points, gptr->ldim, node->mu1*alpha * expf((dist/(-2.0 * radius))));
      off = gptr->ldim;
      adapt(&map->codes[n].points[off], &node->points[off], 2*gptr->FanOut, node->mu2*alpha * expf((dist/(-2.0 * radius))));
      off += 2*gptr->FanOut;
      adapt(&map->codes[n].points[off], &node->points[off], 2*gptr->FanIn, node->mu3*alpha * expf((dist/(-2.0 * radius))));
      off += 2*gptr->FanIn;
      adapt(&map->codes[n].points[off], &node->points[off], gptr->tdim, node->mu4*alpha * expf((dist/(-2.0 * radius))));
    */
    /* Update the codebook */

    AdaptVector(map->codes[n].points, glb_ivec, map->dim, alpha
		* expf((dist/(-2.0 * radius))));
  }
}

/******************************************************************************
 Description: 

 Return value: 
******************************************************************************/
void VQAdapt(struct Graph *gptr, struct Map *map, struct Node *node,
	     struct Winner *winner, FLOAT radius, FLOAT alpha) {
  int i, n, id;
  FLOAT *codebook, a, b;
  UNSIGNED noc, ldim, offset;

  node->winner = winner->codeno;
  ldim = gptr->ldim;
  noc = map->xdim * map->ydim;

  /* Update the codebook */
  codebook = map->codes[winner->codeno].points;

  for (i = 0; i < ldim; i++)
    /* update label component */
    codebook[i] += alpha * (node->points[i] - codebook[i]);

  /* update child coord component */
  a = 0.0;
  for (i = 0; i < gptr->FanOut; i++) {
    id = node->points[ldim + 2*i];
    for (n = 0; n < noc; n++) {
      if (n == id)
	codebook[ldim+i*noc+n] += alpha * (1 - codebook[ldim+i*noc+n]);
      else
	codebook[ldim+i*noc+n] -= alpha * codebook[ldim+i*noc+n];
      a += SQR(codebook[ldim+i*noc+n]);
    }
  }
  map->codes[winner->codeno].a = a;

  /* update parent coord component */
  offset = ldim+gptr->FanOut*noc;
  b = 0.0;
  for (i = 0; i < gptr->FanIn; i++) {
    id = (int)node->points[gptr->ldim + 2*noc + 2*i];
    for (n = 0; n < noc; n++) {
      if (n == id)
	codebook[offset+i*noc+n] += alpha * (1
					     - codebook[offset+i*noc+n]);
      else
	codebook[offset+i*noc+n] -= alpha * codebook[offset+i*noc+n];
      b += SQR(codebook[offset+i*noc+n]);
    }
  }
  map->codes[winner->codeno].b = b;

  /* update target component */
  offset = ldim + noc * (gptr->FanOut + gptr->FanIn);
  for (i = 0; i < gptr->tdim; i++)
    codebook[offset+i] += alpha * (node->points[ldim+2*(gptr->FanOut+gptr->FanIn)+i] - codebook[offset+i]);
}

/****************************************************************************
 Description: Signal handler. This function specifies what to do when a ctrl-c
 is caught.

 Return value: This function does not return a value.
****************************************************************************/
void SigHandler(int arg) {
  static int flag = 0;
  time_t mytime;

  _save_then_exit_ = 1; /* Indicate that we wish to save before exit */
  mytime = time(NULL);
  if (flag == 0) { /* If ctrl-c cought the first time... */
    fprintf(stderr, "\nFirst interrupt signal detected on %s", ctime(&mytime));
    SlideIn(stderr, 1, "Interrupting training...");
    fputc('\n', stderr);
    SlideIn(stderr, 0, "Wait for current iteration to stop for a safe exit or press ctrl-c again to");
    fputc('\n', stderr);
    SlideIn(stderr, 0, "force an immediate stop but then all trained network data will be lost!");
    fputc('\n', stderr);
  }

  else { /* If ctrl-c is cought more than once */
    fprintf(stderr, "\nSecond interrupt signal detected on %s", ctime(&mytime));
    fprintf(stderr, "Forced exit. Stopping now!\n");
    exit(0); /* Stop here and now */
  }
  flag++;
}

/****************************************************************************
 Description: Installs a signal handler which will catch interrupt signals such
 as those initiated by ctrl-c of a kill command.

 Return value: This function does not return a value.
****************************************************************************/
void InstallHandlers() {
  struct sigaction act;

  memset(&act, 0, sizeof(struct sigaction));
  act.sa_handler = SigHandler;
  sigaction(SIGINT, &act, NULL);
}

/******************************************************************************
 Description: Set the appropriate function for computing the alpha value

 Return value: A function pointer to the appropriate function for computing
 the alpha value.
******************************************************************************/
FLOAT (*SetAlpha(int alphatype))(UNSIGNED, UNSIGNED, FLOAT)
{
  if (alphatype == ALPHA_LINEAR)
    return LinearDecrease; /* Linearly decreasing alpha      */
  else if (alphatype == ALPHA_EXPONENTIAL)
    return ExponentialAlpha;/* Exponentially decreasing alpha */
  else if (alphatype == ALPHA_CONSTANT)
    return ConstantAlpha; /* Constant alpha value           */
  else
    return SigmoidalAlpha; /* Default alpha type is sigmoidal*/
}

/******************************************************************************
 Description: Set the appropriate function for adapting the network parameters

 Return value: A function pointer to the appropriate function for adapting the
 network parameters.
******************************************************************************/
void (*SetAdapt(UNSIGNED neighborhood))(struct Graph*, struct Map*,
					struct Node*, struct Winner*, FLOAT, FLOAT) {
  if (neighborhood == NEIGH_GAUSSIAN)
    return GaussianAdapt; /* Gaussian neighborhood */
  else if (neighborhood == NEIGH_BUBBLE)
    return BubbleAdapt; /* Strict neighborhood   */
  else
    return GaussianAdapt; /* Default neighborhood  */
}

/******************************************************************************
 Description: Set the appropriate function for adapting the network parameters

 Return value: A function pointer to the appropriate function for adapting the
 network parameters.
******************************************************************************/
void (*ReSetAdapt(UNSIGNED neighborhood, struct Parameters *parameters))(struct Graph*, struct Map*,
									 struct Node*, struct Winner*, FLOAT, FLOAT) {
  if (neighborhood == NEIGH_GAUSSIAN){
    //if(parameters->memorysave)
      //return GaussianAdaptMemSave;
    return GaussianAdapt; /* Gaussian neighborhood */
  }
  else if (neighborhood == NEIGH_BUBBLE)
    return BubbleAdapt; /* Strict neighborhood   */
  else
    return GaussianAdapt; /* Default neighborhood  */
}

/******************************************************************************
 Description: Main training routine for a CSOM-SD, this also works for standard
 SOM, and for SOM-SD.

 Return value:  0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainSOMSD(struct Parameters *parameters) {
  UNSIGNED i, nnum;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node = NULL;
  struct Graph *gptr;
  FILE *logfile;
  FLOAT (*GetAlpha)(UNSIGNED, UNSIGNED, FLOAT);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,
		     struct Winner *winner);
  void (*Adapt)(struct Graph *gptr, struct Map *map, struct Node *node,
		struct Winner *winner, FLOAT radius, FLOAT alpha_t);
  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);
  int kstepmode = 1;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;

  InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file   */

  /* Set the appropriate function for computing the learning rate */
  GetAlpha = SetAlpha(parameters->alphatype);

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledian; /* Use Eucledian distance        */

  /* Set the appropriate function for adapting the network parameters */
  Adapt = SetAdapt(parameters->map.neighborhood);

  /* Set the appropriate function for updating childens location in nodes */
  if (parameters->map.topology == TOPOL_VQ) { /* In VQ mode...   */
    UpdateOffspringStates = UpdateChildrensLocationVQ; /* use ID value    */
    FindWinner = VQFindWinnerEucledian; /* Use Eucledian distance VQ mode */
    Adapt = VQAdapt; /* No topology = no neighborhood = VQ adapt mode   */
    VQSet_ab(parameters); /* Initialize auxillary variables a and b      */
  }
  if (parameters->contextual) {
    if (parameters->undirected)
      kstepmode = 0;
    else if (parameters->train->FanIn == 0) {
      fprintf(stderr, "Warning: No inlink available for contextual mode. Will fall back to normal mode.\n");
      UpdateOffspringStates = UpdateChildrensLocation; /* Use coordinates */
      parameters->contextual = NO;

    }
    if (parameters->contextual) {
      fprintf(stderr, "Contextual mode: Training on single map is assumed\n");
      fprintf(stderr, "Will recompute states at every iteration!!\n");
      UpdateOffspringStates = UpdateChildrenAndParentLocation;/*p & c coords*/
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);
    }
  } else
    UpdateOffspringStates = UpdateChildrensLocation; /* Use coordinates */

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
#ifdef EXCLUSIVEMAPPING
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif

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
    	  //if (parameters->kfoldtime>=0)
   			 //if (nnum >= 12*parameters->kfoldtime && nnum < 12*(parameters->kfoldtime+1)) continue;//Skip the validation set, 12 acts/person of PA project data
    		 //if (nnum >= 60*parameters->kfoldtime && nnum < 60*(parameters->kfoldtime+1)) continue;//Skip the validation set, 60 acts/persom of HASC data
    		  //if (nnum >= 180*parameters->kfoldtime && nnum < 180*(parameters->kfoldtime+1)) continue;// HASC data
    	  	 //if (nnum >= 400*parameters->kfoldtime && nnum < 400*(parameters->kfoldtime+1)) continue;// TROST data
	node = gptr->nodes[nnum];
	alpha_t = GetAlpha(t, tlen, parameters->alpha);
	radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

	t++;
	if (!parameters->contextual)
	  UpdateOffspringStates(gptr, node); /* Update child-state-vector   */

#ifdef APPROXIMATION
	if (i != 0)
	  FindWinnerEucledianApprox(map, node, gptr, &winner);
	else
#endif
	  FindWinner(map, node, gptr, &winner);/* Find best matching codebook*/
	Adapt(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook*/
	terror += winner.diff;
	counter++;
      }
    }

    if (parameters->contextual)
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    if (_save_then_exit_) { /* Save and exit if a interrupt signal was caught */
      char fname[32];
      sprintf(fname, "interrupted%d.net", (int)getpid());
      free(parameters->onetfile);
      parameters->onetfile = strdup(fname); /* Assign alternate filename */
      fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
      break; /* Break the training cycle  */
    }

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
  if (!_save_then_exit_)
    fprintf(stderr, "\033[01;32m%56s\033[00m\n", "[OK]");

  if (logfile != stdout)
    MyFclose(logfile);

  //Begin Tuc special
  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
	  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
    	  if (parameters->kfoldtime>=0)
   			 node = gptr->nodes[nnum];
		  alpha_t = GetAlpha(t, tlen, parameters->alpha);
		  radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

		  if (!parameters->contextual)
			  UpdateOffspringStates(gptr, node); /* Update child-state-vector   */

		  FindWinner(map, node, gptr, &winner);/* Find best matching codebook*/
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
 Description: Main training routine for a CSOM-SD, this also works for standard
 SOM, and for SOM-SD.

 Return value:  0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainSOM(struct Parameters *parameters) {
  UNSIGNED i, nnum;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  FLOAT (*GetAlpha)(UNSIGNED, UNSIGNED, FLOAT);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,
		     struct Winner *winner);
  void (*Adapt)(struct Graph *gptr, struct Map *map, struct Node *node,
		struct Winner *winner, FLOAT radius, FLOAT alpha_t);
  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);
  //  int kstepmode = 1;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;

  InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file   */

  /* Set the appropriate function for computing the learning rate */
  GetAlpha = SetAlpha(parameters->alphatype);

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledian; /* Use Eucledian distance        */

  /* Set the appropriate function for adapting the network parameters */
  Adapt = SetAdapt(parameters->map.neighborhood);

  UpdateOffspringStates = UpdateChildrensLocation; /* Use coordinates */

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  //fprint(stderr, "Tuc training map......"); /* Print what is being done      */
  map = &parameters->map;

  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next){
		  tlen += gptr->numnodes;
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
		  gcnt++;
		  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
			  node = gptr->nodes[nnum];
			  alpha_t = GetAlpha(t, tlen, parameters->alpha);
			  radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

			  t++;
			  UpdateOffspringStates(gptr, node); /* Update child-state-vector   */

			  FindWinner(map, node, gptr, &winner);/* Find best matching codebook*/
			  Adapt(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook*/
			  terror += winner.diff;
			  counter++;
		  }
	  }

	  map->iter++;
	  fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
	  fflush(logfile);

	  if (_save_then_exit_) { /* Save and exit if a interrupt signal was caught */
		  char fname[32];
		  sprintf(fname, "interrupted%d.net", (int)getpid());
		  free(parameters->onetfile);
		  parameters->onetfile = strdup(fname); /* Assign alternate filename */
		  fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
		  break; /* Break the training cycle  */
	  }

	  /* Create a snapshot if required */
	  if (parameters->snap.interval>0 &&!(map->iter%parameters->snap.interval)) {
		  if (parameters->snap.file != NULL) SaveSnapShot(parameters);
		  if (parameters->snap.command != NULL)
			  system(parameters->snap.command);
	  }

	  PrintProgress(map->iter); /* Print Progress */
    Hausekeeping(parameters);
  }
  StopProgressMeter();
  if (!_save_then_exit_)
	  fprintf(stderr, "\033[01;32m%56s\033[00m\n", "[OK]");

  if (logfile != stdout)
	  MyFclose(logfile);

  //Begin Tuc special
  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      if (parameters->kfoldtime>=0)//??How can this be negative??
	node = gptr->nodes[nnum];
      else{
	fprintf(stderr, PRT_ERROR "Fix bug in function TrainSOM (node undefined)\n");
	exit(0);
      }
		  alpha_t = GetAlpha(t, tlen, parameters->alpha);
		  radius_t = parameters->radius - (parameters->radius - 1.0)	* (float)t/(float)tlen;

		  UpdateOffspringStates(gptr, node); /* Update child-state-vector   */

		  FindWinner(map, node, gptr, &winner);/* Find best matching codebook*/
		  //printf("%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
		  fprintf(file_res,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
	  }
  }
  fclose(file_res);
  //End Tuc special

  return 0;
}
/******************************************************************************
 Description: Computes the stable mapping point

 Return value: void
******************************************************************************/
void ComputeStablePoint(struct Parameters *parameters, struct Graph *gptr, UNSIGNED t, UNSIGNED tlen, int wsize) {
  UNSIGNED i, k, offset, nnum, goptx, gopty, gopt_numxblocks;
  FLOAT radius_t, r_fuzzy, sigma, sqrtfpi;
  struct Map *map;
  struct Node *node;
  double PI = 3.1415926535897932; 
  FLOAT converge = (float)1/sqrtf(2*PI);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,
		     struct Winner *winner);
  //  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);
  int *xset;
  int *yset;
  int neighc = 0, idx = 0, xd, yd;
  int prefuzzy = 3;
  FLOAT *nodetmp;
  struct Winner winner;
  UNSIGNED max;
  int den = 5, numlabels = 0;
  UNSIGNED ***mapclass;  /* Supervised mode */
  int x, y, j, l;

  ComputeDistance = ComputeHexaDistance;

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledian; /* Use Eucledian distance        */

  /* Set the appropriate function for updating childens location in nodes */
  if (parameters->map.topology == TOPOL_VQ) { /* In VQ mode...   */
    // UpdateOffspringStates = UpdateChildrensLocationVQ; /* use ID value    */
    FindWinner = VQFindWinnerEucledian; /* Use Eucledian distance VQ mode */
    VQSet_ab(parameters); /* Initialize auxillary variables a and b      */
  }

  mapclass = NULL;
  if (parameters->super == REJECT){  //Supervised mode
    if (parameters->beta == 0.0){
      fprintf(stderr, "WARNING: Rejection rate 'beta' is zero.\n");
      fprintf(stderr, "         Disabeling supervised mode!\n");
      parameters->super = 0;
    }
    else{
      glb_beta = parameters->beta; //Dirty hack to pass rejection rate
      if (parameters->map.neighborhood == NEIGH_GAUSSIAN){
	//	Adapt = GaussianAdaptSupervised;
      }
      else{
	fprintf(stderr, "ERROR: Supervised mode using a neighbourhood other than Gaussian is not yet\n");
	fprintf(stderr, "         supported! Exiting now.\n");
	exit(0);
      }
      numlabels = GetNumLabels()+1;
      mapclass = (UNSIGNED***)malloc(parameters->map.xdim* sizeof(UNSIGNED**));
      for (x = 0; x < parameters->map.xdim; x++){
	mapclass[x]=(UNSIGNED**)malloc(parameters->map.ydim*sizeof(UNSIGNED*));
	for (y = 0; y < parameters->map.ydim; y++)
	  mapclass[x][y] = (UNSIGNED*)calloc(numlabels, sizeof(UNSIGNED));
      }
    }
  }

  map = &parameters->map; /* This makes de-referencing easier         */
  goptx = map->goptx;
  gopty = map->gopty;
  gopt_numxblocks = map->xdim / goptx;

  /* Now lets get started */
  sqrtfpi = sqrtf(2*PI);
  xset = (int*) malloc(wsize * sizeof(int));
  yset = (int*) malloc(wsize * sizeof(int));
  nodetmp = (float*) malloc(map->xdim*map->ydim * sizeof(FLOAT));

  /* Init state vector to zero */
  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
    node = gptr->nodes[nnum]; /* This makes de-referencing easier */
    offset = gptr->ldim;
    memset(&node->points[offset], 0, (gptr->dimension-offset) * sizeof(FLOAT));
    node->x = 0;
    node->y = 0;
  }

  for (i = 0; i < 3*parameters->precision; i++) {
#ifdef EXCLUSIVEMAPPING
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif

    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      FLOAT winval = 1;
      node = gptr->nodes[nnum]; /* This makes de-referencing easier */
      radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
      r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
      sigma = converge+(float)r_fuzzy;				
                                                   			
      if (!parameters->contextual) {
	offset = gptr->ldim;
	memset(&node->points[offset], 0, (gptr->dimension-offset) * sizeof(FLOAT));
	memset(&nodetmp[0], 0, (map->xdim*map->ydim)	* sizeof(FLOAT));

	for (k = 0; k < gptr->FanOut; k++) {
	  if (node->children[k] != NULL) {
	    if (offset + node->children[k]->x/goptx + node->children[k]->y/gopty*gopt_numxblocks>= gptr->dimension)
	      fprintf(stderr, "Totally unexpected coordinate encountered.\n");
	    if (i>prefuzzy) {
	      nodetmp[node->children[k]->x+node->children[k]->y*map->xdim]++;
	      xset[neighc] = node->children[k]->x;
	      yset[neighc] = node->children[k]->y;
	      neighc++;
	    } 
	    else
	      node->points[offset + node->children[k]->x/goptx+ node->children[k]->y/gopty * gopt_numxblocks]++;
	  } 
	  else
	    break; /* No more offsprings after a NULL-pointer expected */
	}

	for (k = 0u; k < node->numparents; k++) {
	  if (i>prefuzzy) {
	    nodetmp[node->parents[k]->x+node->parents[k]->y*map->xdim]++;
	    xset[neighc]= node->parents[k]->x;
	    yset[neighc]= node->parents[k]->y;
	    neighc++;
	  } 
	  else
	    node->points[offset + node->parents[k]->x/goptx+ node->parents[k]->y/gopty * gopt_numxblocks]++;
	}

	if (i>prefuzzy) {
	  for (xd = 0; xd < map->xdim; xd++) {
	    for (yd = 0; yd < map->ydim; yd++) {
	      FLOAT maximp = 0;
	      for (idx = 0; idx < neighc; idx++) {
		FLOAT d = ComputeDistance(xd, yd, xset[idx], yset[idx]);
		if (d > radius_t*8)
		  continue;
		winval = expf(-(d*d)/(2*sigma*sigma))/(sigma*sqrtfpi);
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
	      node->points[offset+xd/goptx+yd/gopty*gopt_numxblocks]+= nodetmp[xd+yd*map->xdim];
	}
      }
      FindWinner(map, node, gptr, &winner); /* Find best matching codebook */

      if (mapclass){ // in supervised mode
	mapclass[map->codes[winner.codeno].x][map->codes[winner.codeno].y][node->label]++;
      }
    }

    if (mapclass){ // In supervised mode, update class label of codebooks
      for (j = 0; j < map->xdim * map->ydim; j++){
	map->codes[j].label = 0;
	max = mapclass[map->codes[j].x][map->codes[j].y][0];
	for (l = 1; l < numlabels; l++){
	  if (mapclass[map->codes[j].x][map->codes[j].y][l] > max){
	    map->codes[j].label = l;
	    max = mapclass[map->codes[j].x][map->codes[j].y][l];
	  }
	}
      }
      //Now reset 'mapclass'
      for (x = 0; x < map->xdim; x++){
	for (y = 0; y < map->ydim; y++)
	  memset(mapclass[x][y], 0, numlabels * sizeof(UNSIGNED));
      }
    }
    
    //if (parameters->nice)
    //  SleepOnHiLoad(); /* Sleep when system load is high */
  }
  free(xset);
  free(yset);
  free(nodetmp);


#ifdef EXCLUSIVEMAPPING
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif
}


/******************************************************************************
 Description: Main training routine for training a Fuzzy GraphSOM

 Return value: 0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainPMGraphSOM(struct Parameters *parameters) {
  UNSIGNED i, k, offset, nnum, goptx, gopty, gopt_numxblocks;
  FLOAT alpha_t, radius_t, r_fuzzy, sigma, sqrtfpi;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  double PI = 3.1415926535897932; 
  FLOAT converge = (float)1/sqrtf(2*PI);
  FLOAT (*GetAlpha)(UNSIGNED, UNSIGNED, FLOAT);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,
		     struct Winner *winner);
  void (*Adapt)(struct Graph *gptr, struct Map *map, struct Node *node,
		struct Winner *winner, FLOAT radius, FLOAT alpha_t);
  //  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);
  //  FILE *rofile=NULL;
  char rfname[256];
  int wsize;
  int *xset;
  int *yset;
  int neighc = 0, idx = 0, xd, yd;
  int prefuzzy = 3;
  int gctn = 0;
  FLOAT *nodetmp;
  int kstepmode = 1;
  struct Winner winner;
  UNSIGNED tlen, t, max;
  int counter, gcount, ginclude;
  FLOAT terror;
  int den = 5;
  UNSIGNED ***mapclass;  /* Supervised mode */
  int x, y, j, l;
  int numlabels = 0;

  ComputeDistance = ComputeHexaDistance;
  InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file       */

  /* Set the appropriate function for computing the learning rate     */
  GetAlpha = SetAlpha(parameters->alphatype);

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledian; /* Use Eucledian distance        */

  /* Set the appropriate function for adapting the network parameters */
  Adapt = SetAdapt(parameters->map.neighborhood);

  /* Set the appropriate function for updating childens location in nodes */
  if (parameters->map.topology == TOPOL_VQ) { /* In VQ mode...   */
    // UpdateOffspringStates = UpdateChildrensLocationVQ; /* use ID value    */
    FindWinner = VQFindWinnerEucledian; /* Use Eucledian distance VQ mode */
    Adapt = VQAdapt; /* No topology = no neighborhood = VQ adapt mode   */
    VQSet_ab(parameters); /* Initialize auxillary variables a and b      */
  }

  mapclass = NULL;
  if (parameters->super == REJECT){  //Supervised mode
    if (parameters->beta == 0.0){
      fprintf(stderr, "WARNING: Rejection rate 'beta' is zero.\n");
      fprintf(stderr, "         Disabeling supervised mode!\n");
      parameters->super = 0;
    }
    else{
      glb_beta = parameters->beta; //Dirty hack to pass rejection rate
      if (parameters->map.neighborhood == NEIGH_GAUSSIAN)
	Adapt = GaussianAdaptSupervised;
      else{
	fprintf(stderr, "ERROR: Supervised mode using a neighbourhood other than Gaussian is not yet\n");
	fprintf(stderr, "         supported! Exiting now.\n");
	exit(0);
      }
      numlabels = GetNumLabels()+1;
      mapclass = (UNSIGNED***)malloc(parameters->map.xdim* sizeof(UNSIGNED**));
      for (x = 0; x < parameters->map.xdim; x++){
	mapclass[x]=(UNSIGNED**)malloc(parameters->map.ydim*sizeof(UNSIGNED*));
	for (y = 0; y < parameters->map.ydim; y++)
	  mapclass[x][y] = (UNSIGNED*)calloc(numlabels, sizeof(UNSIGNED));
      }
    }
  }

  //TMP: disable contextual mode
  if (parameters->contextual){
    fprintf(stderr, "Note: Contextual mode disabled!!\n");
    fprintf(stderr, "      Code change required if stable point needs to be computed\n");
    parameters->contextual = 0;
  }

  if (parameters->contextual) {
    if (parameters->undirected)
      kstepmode = 0;
    else if (parameters->train->FanIn == 0) {
      fprintf(stderr, "Warning: No inlink available for contextual mode. Will fall back to normal mode.\n");
      //      UpdateOffspringStates = UpdateChildrensLocation;
      parameters->contextual = NO;
    }
    if (parameters->contextual) {
      fprintf(stderr, "Contextual mode: Training on single map is assumed\n");
      fprintf(stderr, "Will recompute states at every iteration!!\n");
      //      UpdateOffspringStates = UpdateChildrenAndParentLocation;
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);
    }
  }
  /*else
    UpdateOffspringStates = UpdateChildrensLocation;*/

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  fprint(stderr, "Training PMGraphSOM......"); /* Print what is being done  */
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
  sqrtfpi = sqrtf(2*PI);
  gptr = parameters->train;
  wsize = maxpnum + gptr->FanOut;
  xset = (int*) malloc(wsize * sizeof(int));
  yset = (int*) malloc(wsize * sizeof(int));
  nodetmp = (float*) malloc(map->xdim*map->ydim * sizeof(FLOAT));

  fprintf(stderr, "\nno PM for first %d iterations; PM radius den = %d\n", (prefuzzy+1), den);

  if (parameters->percentremoved > 0){
    for (gcount = 0, gptr = parameters->train; gptr != NULL; gcount++, gptr = gptr->next);
    fprintf(stderr, "# Removing the last %d graphs from training set\n", parameters->percentremoved*gcount/100);
    ginclude = gcount-parameters->percentremoved*gcount/100;
  }
  else{
    for (gcount = 0, gptr = parameters->train; gptr != NULL; gcount++, gptr = gptr->next);
    ginclude = gcount;
  }
  fprintf(stderr, "# -> SOM updated on %d graphs.\n", ginclude);

  if (parameters->removed > 0){
    for (gptr = parameters->train; (gptr->gnum+1) != parameters->removed && gptr != NULL; gptr = gptr->next);
    if (gptr == NULL){
      fprintf(stderr, "Error: Cannot remove non-existing graph %d from dataset\n", parameters->removed);
      exit(0);
    }
    fprintf(stderr, "# Removed %d-th graph (%s) from training set\n", parameters->removed, gptr->gname);
#ifdef ONE_CLASS_ONLY
    fprintf(stderr, "# Training one class only! Class trained: '%s'\n", ONE_CLASS_ONLY);
    if (strcmp(ONE_CLASS_ONLY, GetLabel(gptr->nodes[0]->label))){
      fprintf(stderr, "# WARNING: Graph removed is never trained anyway (is from a different class '%s')\n", GetLabel(gptr->nodes[0]->label));
      fprintf(stdout, "WARNING: Graph removed (%s,%d) is never trained anyway (is from a different class '%s')\n", gptr->gname, parameters->removed, GetLabel(gptr->nodes[0]->label));
      fflush(stdout);
      exit(0);
    }
#endif

  }
  else{
#ifdef ONE_CLASS_ONLY
    fprintf(stderr, "# Training one class only! Class trained: '%s'\n", ONE_CLASS_ONLY);
#endif
  }
  for (i = map->iter; i < parameters->rlen; i++) {
#ifdef EXCLUSIVEMAPPING
    int ex = 0, ti;
    for (ti = 0; ti < parameters->map.xdim * parameters->map.ydim;ti++)
      if (parameters->map.activation[ti] > 0.0) ex++;
    fprintf(stderr, "Activations: %d\n", ex);
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif
    //    if (rofile)
    //      fclose(rofile);
    sprintf(rfname, "iter%d.dat", i);
    //    rofile = fopen(rfname, "w");
    counter = 0;
    gctn = 0;
    terror = 0.0; /* Init train error */

    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      gctn++;
      //fprintf(stderr, "graph %s: ", gptr->gname);
      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	//fprintf(stderr, "node %d\n", nnum);
	FLOAT winval = 1;
	node = gptr->nodes[nnum]; /* This makes de-referencing easier */
	alpha_t = GetAlpha(t, tlen, parameters->alpha);
	//radius_t = 1.0 + (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;		
	radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
	r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
	sigma = converge+(float)r_fuzzy;				
                                                   			
	t++;
	if (parameters->precision > 0){     //If higher precision is requested
	  ComputeStablePoint(parameters, gptr, t, tlen, wsize);  //re-compute mappings until stable
	}

	if (!parameters->contextual) {
	  offset = gptr->ldim;
	  memset(&node->points[offset], 0, (gptr->dimension-offset)	* sizeof(FLOAT));
	  memset(&nodetmp[0], 0, (map->xdim*map->ydim)	* sizeof(FLOAT));

	  for (k = 0; k < gptr->FanOut; k++) {
	    if (node->children[k] != NULL) {
	      if (offset + node->children[k]->x/goptx + node->children[k]->y/gopty*gopt_numxblocks>= gptr->dimension)
		fprintf(stderr, "Totally unexpected coordinate encountered.\n");
	      if (i>prefuzzy) {
		nodetmp[node->children[k]->x+node->children[k]->y*map->xdim]++;
		xset[neighc] = node->children[k]->x;
		yset[neighc] = node->children[k]->y;
		neighc++;
	      } 
	      else
		node->points[offset + node->children[k]->x/goptx+ node->children[k]->y/gopty * gopt_numxblocks]++;
	    } 
	    else
	      break; /* No more offsprings after a NULL-pointer expected */
	  }

	  for (k = 0u; k < node->numparents; k++) {
	    if (i>prefuzzy) {
	      nodetmp[node->parents[k]->x+node->parents[k]->y*map->xdim]++;
	      xset[neighc]= node->parents[k]->x;
	      yset[neighc]= node->parents[k]->y;
	      neighc++;
	    } 
	    else
	      node->points[offset + node->parents[k]->x/goptx+ node->parents[k]->y/gopty * gopt_numxblocks]++;
	  }

	  if (i>prefuzzy) {
	    for (xd = 0; xd < map->xdim; xd++) {
	      for (yd = 0; yd < map->ydim; yd++) {
		FLOAT maximp = 0;
		for (idx = 0; idx < neighc; idx++) {
		  FLOAT d = ComputeDistance(xd, yd, xset[idx], yset[idx]);
		  if (d > radius_t*8)
		    continue;
		  winval = expf(-(d*d)/(2*sigma*sigma))/(sigma*sqrtfpi);
		  //fprintf(stderr, "%1.9f, %1.9f, %1.9f\n", d, sigma, winval);
		  FLOAT tmp = expf((d/(-2.0 * r_fuzzy * r_fuzzy)))*nodetmp[xset[idx]+yset[idx]*map->xdim]*winval;
		  if (maximp<tmp)
		    maximp = tmp;
		}
		// if (i == 32 && nnum < 4 && gctn < 2){//xxx
		/*
		if (maximp >= 0.000001)
		  fprintf(rofile, "%d %d %d %d %f\n", gctn, nnum, xd, yd, maximp);
		*/

		if (nodetmp[xd+yd*map->xdim] < maximp)
		  nodetmp[xd+yd*map->xdim] = maximp;
	      }
	    }

	    neighc = 0;
	    for (xd = 0; xd < map->xdim; xd++)
	      for (yd = 0; yd < map->ydim; yd++)
		node->points[offset+xd/goptx+yd/gopty*gopt_numxblocks]+= nodetmp[xd+yd*map->xdim];
	  }
	}
	FindWinner(map, node, gptr, &winner); /* Find best matching codebook */

	if (mapclass){ // in supervised mode
	  mapclass[map->codes[winner.codeno].x][map->codes[winner.codeno].y][node->label]++;
	}

	if (gctn < ginclude){
	  if (parameters->removed == 0 || (gptr->gnum+1) != parameters->removed){
#ifdef ONE_CLASS_ONLY
	    if (!strcmp(ONE_CLASS_ONLY, GetLabel(node->label))){
#endif
	      Adapt(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook*/
	      terror += winner.diff;
	      counter++;
#ifdef ONE_CLASS_ONLY
	    }
#endif
	  }
	}
      }
    }

    if (mapclass){ // In supervised mode, update class label of codebooks
      for (j = 0; j < map->xdim * map->ydim; j++){
	map->codes[j].label = 0;
	max = mapclass[map->codes[j].x][map->codes[j].y][0];
	for (l = 1; l < numlabels; l++){
	  if (mapclass[map->codes[j].x][map->codes[j].y][l] > max){
	    map->codes[j].label = l;
	    max = mapclass[map->codes[j].x][map->codes[j].y][l];
	  }
	}
      }
      //Now reset 'mapclass'
      for (x = 0; x < map->xdim; x++){
	for (y = 0; y < map->ydim; y++)
	  memset(mapclass[x][y], 0, numlabels * sizeof(UNSIGNED));
      }
    }
    
    if (parameters->contextual)
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    if (_save_then_exit_) { /* Save and exit if a interrupt signal was caught*/
      char fname[32];
      sprintf(fname, "interrupted%d.net", (int)getpid());
      free(parameters->onetfile);
      parameters->onetfile = strdup(fname); /* Assign alternate filename */
      fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
      break; /* Break the training cycle  */
    }

    /* Create a snapshot if required */
    if (parameters->snap.interval>0 &&!(map->iter%parameters->snap.interval)) {
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

  //  fclose(rofile);


#ifdef EXCLUSIVEMAPPING
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif



  t = tlen-1;
  radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
  r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
  sigma = converge+(float)r_fuzzy;				

  gctn = 0;
  /*Mutag special!!! Can be removed */

  /* Init state vector to zero */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      node = gptr->nodes[nnum]; /* This makes de-referencing easier */
      offset = gptr->ldim;
      memset(&node->points[offset], 0, (gptr->dimension-offset) * sizeof(FLOAT));
      node->x = 0;
      node->y = 0;
    }
  }

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      gctn++;

    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      FLOAT winval = 1;
      node = gptr->nodes[nnum]; /* This makes de-referencing easier */
                                                   			
      if (!parameters->contextual) {
	offset = gptr->ldim;
	memset(&node->points[offset], 0, (gptr->dimension-offset) * sizeof(FLOAT));
	memset(&nodetmp[0], 0, (map->xdim*map->ydim) * sizeof(FLOAT));

	for (k = 0; k < gptr->FanOut; k++) {
	  if (node->children[k] != NULL) {
	    nodetmp[node->children[k]->x+node->children[k]->y*map->xdim]++;
	    xset[neighc] = node->children[k]->x;
	    yset[neighc] = node->children[k]->y;
	    neighc++;
	  }
	  else
	    break; /* No more offsprings after a NULL-pointer expected */
	}

	for (k = 0u; k < node->numparents; k++) {
	  nodetmp[node->parents[k]->x+node->parents[k]->y*map->xdim]++;
	  xset[neighc]= node->parents[k]->x;
	  yset[neighc]= node->parents[k]->y;
	  neighc++;
	}

	for (xd = 0; xd < map->xdim; xd++) {
	  for (yd = 0; yd < map->ydim; yd++) {
	    FLOAT maximp = 0;
	    for (idx = 0; idx < neighc; idx++) {
	      FLOAT d = ComputeDistance(xd, yd, xset[idx], yset[idx]);
	      if (d > radius_t*8)
		continue;
	      winval = expf(-(d*d)/(2*sigma*sigma))/(sigma*sqrtfpi);
	      FLOAT tmp = expf((d/(-2.0 * r_fuzzy * r_fuzzy)))*nodetmp[xset[idx]+yset[idx]*map->xdim]*winval;
	      if (maximp<tmp)
		maximp = tmp;
	    }
	    if (nodetmp[xd+yd*map->xdim] < maximp)
	      nodetmp[xd+yd*map->xdim] = maximp;
	  }

	  neighc = 0;
	  for (xd = 0; xd < map->xdim; xd++)
	    for (yd = 0; yd < map->ydim; yd++)
	      node->points[offset+xd/goptx+yd/gopty*gopt_numxblocks]+= nodetmp[xd+yd*map->xdim];
	}
      }

      FindWinner(map, node, gptr, &winner);/* Refresh best matching codebook */

      //      oflag = 0;
      if (gctn <= ginclude){
	if (parameters->removed == 0 || (gptr->gnum+1) != parameters->removed){
#ifdef ONE_CLASS_ONLY
	  if (!strcmp(ONE_CLASS_ONLY, GetLabel(node->label))){
#endif
	    fprintf(stdout, "(%s) %d %d %d %d %s %e trained\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);
	    //	    oflag = 1;

#ifdef ONE_CLASS_ONLY
	  }
	  else{
	    fprintf(stdout, "(%s) %d %d %d %d %s %e\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);
	  }
#endif
	}
	else{
	  fprintf(stdout, "(%s) %d %d %d %d %s %e removed\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);
	}
      }
      //      if (!oflag)
      else
	fprintf(stdout, "(%s) %d %d %d %d %s %e removepercent\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);

    }
  }






  if (!_save_then_exit_)
    fprintf(stderr, "\033[01;32m%51s\033[00m\n", "[OK]");

  if (logfile != stdout)
    MyFclose(logfile);

  return 1;
}

/******************************************************************************
 Description: Main training routine for training a Fuzzy GraphSOM

 Return value: 0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainPMGraphSOMCompact(struct Parameters *parameters) {
  UNSIGNED i, k, offset, nnum;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  //  double PI = 3.1415926535897932; 
  //  FLOAT converge = (float)1/sqrtf(2*PI);
  FLOAT (*GetAlpha)(UNSIGNED, UNSIGNED, FLOAT);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,
		     struct Winner *winner);
  void (*Adapt)(struct Graph *gptr, struct Map *map, struct Node *node,
		struct Winner *winner, FLOAT radius, FLOAT alpha_t);
  //  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);
  //  FILE *rofile=NULL;
  char rfname[256];
  int gctn = 0;
  int kstepmode = 1;
  struct Winner winner;
  UNSIGNED tlen, t, max;
  int counter, gcount, ginclude;
  FLOAT terror;
  //  int den = 5;
  UNSIGNED ***mapclass;  /* Supervised mode */
  int x, y, j, l;
  int numlabels = 0;
  int pnum = 0, maxpnum = 0;
#ifndef NOORDER
  FLOAT t1, t2;
  int s1, s2;
#endif


  InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file       */

  /* Set the appropriate function for computing the learning rate     */
  GetAlpha = SetAlpha(parameters->alphatype);

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledian; /* Use Eucledian distance        */

  /* Set the appropriate function for adapting the network parameters */
  Adapt = SetAdapt(parameters->map.neighborhood);

  /* Set the appropriate function for updating childens location in nodes */
  if (parameters->map.topology == TOPOL_VQ) { /* In VQ mode...   */
    //UpdateOffspringStates = UpdateChildrensLocationVQ; /* use ID value    */
    FindWinner = VQFindWinnerEucledian; /* Use Eucledian distance VQ mode */
    Adapt = VQAdapt; /* No topology = no neighborhood = VQ adapt mode   */
    VQSet_ab(parameters); /* Initialize auxillary variables a and b      */
  }

  mapclass = NULL;
  if (parameters->super == REJECT){  //Supervised mode
    if (parameters->beta == 0.0){
      fprintf(stderr, "WARNING: Rejection rate 'beta' is zero.\n");
      fprintf(stderr, "         Disabeling supervised mode!\n");
      parameters->super = 0;
    }
    else{
      glb_beta = parameters->beta; //Dirty hack to pass rejection rate
      if (parameters->map.neighborhood == NEIGH_GAUSSIAN)
	Adapt = GaussianAdaptSupervised;
      else{
	fprintf(stderr, "ERROR: Supervised mode using a neighbourhood other than Gaussian is not yet\n");
	fprintf(stderr, "         supported! Exiting now.\n");
	exit(0);
      }
      numlabels = GetNumLabels()+1;
      mapclass = (UNSIGNED***)malloc(parameters->map.xdim* sizeof(UNSIGNED**));
      for (x = 0; x < parameters->map.xdim; x++){
	mapclass[x]=(UNSIGNED**)malloc(parameters->map.ydim*sizeof(UNSIGNED*));
	for (y = 0; y < parameters->map.ydim; y++)
	  mapclass[x][y] = (UNSIGNED*)calloc(numlabels, sizeof(UNSIGNED));
      }
    }
  }

  //TMP: disable contextual mode
  if (parameters->contextual){
    fprintf(stderr, "Note: Contextual mode disabled!!\n");
    fprintf(stderr, "      Code change required if stable point needs to be computed\n");
    parameters->contextual = 0;
  }

  if (parameters->contextual) {
    if (parameters->undirected)
      kstepmode = 0;
    else if (parameters->train->FanIn == 0) {
      fprintf(stderr, "Warning: No inlink available for contextual mode. Will fall back to normal mode.\n");
      //      UpdateOffspringStates = UpdateChildrensLocation;
      parameters->contextual = NO;
    }
    if (parameters->contextual) {
      fprintf(stderr, "Contextual mode: Training on single map is assumed\n");
      fprintf(stderr, "Will recompute states at every iteration!!\n");
      //      UpdateOffspringStates = UpdateChildrenAndParentLocation;
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);
    }
  }
  /*else
    UpdateOffspringStates = UpdateChildrensLocation;*/

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  fprint(stderr, "Training PMGraphSOM......"); /* Print what is being done  */
  map = &parameters->map; /* This makes de-referencing easier         */
  tlen = 0; /* Compute the total number of update steps */
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

  /* Now lets get started (to train map) */
  t = 0;
  //  sqrtfpi = sqrtf(2*PI);
  gptr = parameters->train;

  if (parameters->percentremoved > 0){
    for (gcount = 0, gptr = parameters->train; gptr != NULL; gcount++, gptr = gptr->next);
    fprintf(stderr, "# Removing the last %d graphs from training set\n", parameters->percentremoved*gcount/100);
    ginclude = gcount-parameters->percentremoved*gcount/100;
  }
  else{
    for (gcount = 0, gptr = parameters->train; gptr != NULL; gcount++, gptr = gptr->next);
    ginclude = gcount;
  }
  fprintf(stderr, "# -> SOM updated on %d graphs.\n", ginclude);

  if (parameters->removed > 0){
    for (gptr = parameters->train; (gptr->gnum+1) != parameters->removed && gptr != NULL; gptr = gptr->next);
    if (gptr == NULL){
      fprintf(stderr, "Error: Cannot remove non-existing graph %d from dataset\n", parameters->removed);
      exit(0);
    }
    fprintf(stderr, "# Removed %d-th graph (%s) from training set\n", parameters->removed, gptr->gname);
#ifdef ONE_CLASS_ONLY
    fprintf(stderr, "# Training one class only! Class trained: '%s'\n", ONE_CLASS_ONLY);
    if (strcmp(ONE_CLASS_ONLY, GetLabel(gptr->nodes[0]->label))){
      fprintf(stderr, "# WARNING: Graph removed is never trained anyway (is from a different class '%s')\n", GetLabel(gptr->nodes[0]->label));
      fprintf(stdout, "WARNING: Graph removed (%s,%d) is never trained anyway (is from a different class '%s')\n", gptr->gname, parameters->removed, GetLabel(gptr->nodes[0]->label));
      fflush(stdout);
      exit(0);
    }
#endif

  }
  else{
#ifdef ONE_CLASS_ONLY
    fprintf(stderr, "# Training one class only! Class trained: '%s'\n", ONE_CLASS_ONLY);
#endif
  }
  for (i = map->iter; i < parameters->rlen; i++) {
#ifdef EXCLUSIVEMAPPING
    int ex = 0, ti;
    for (ti = 0; ti < parameters->map.xdim * parameters->map.ydim;ti++)
      if (parameters->map.activation[ti] > 0.0) ex++;
    fprintf(stderr, "Activations: %d\n", ex);
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif

    sprintf(rfname, "iter%d.dat", i);
    counter = 0;
    gctn = 0;
    terror = 0.0; /* Init train error */

    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      gctn++;
      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	node = gptr->nodes[nnum]; /* This makes de-referencing easier */
	alpha_t = GetAlpha(t, tlen, parameters->alpha);
	radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
	//	r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
	//	sigma = converge+(float)r_fuzzy;				
	t++;
	if (!parameters->contextual) {
	  offset = gptr->ldim;
	  memset(&node->points[offset], 0, (gptr->dimension-offset)	* sizeof(FLOAT));

	  for (k = 0; k < gptr->FanOut; k++) {
	    if (node->children[k] != NULL) {
	      node->points[offset++] = node->children[k]->x;
	      node->points[offset++] = node->children[k]->y;
	    }
	  }

	  for (k = 0u; k < node->numparents; k++) {
	    node->points[offset++] = node->parents[k]->x;
	    node->points[offset++] = node->parents[k]->y;
	  }
	  for (; offset < parameters->map.dim; offset++)
	    node->points[offset] = -1.0;

#ifndef NOORDER
	  //Inverse bubblesort of state vector
	  for (s1 = gptr->ldim; s1 < offset; s1 += 2){
	    for (s2 = offset-2; s2 > s1; s2 -= 2){
	      if (node->points[s2] != -1){
		if (node->points[s1]*node->points[s1] + node->points[s1+1]*node->points[s1+1] > node->points[s2]*node->points[s2] + node->points[s2+1]*node->points[s2+1]){
		  t1 = node->points[s2];
		  t2 = node->points[s2+1];
		  node->points[s2] = node->points[s1];
		  node->points[s2+1] = node->points[s1+1];
		  node->points[s1] = t1;
		  node->points[s1+1] = t2;
		}
	      }
	    }
	  }
#endif
	}
	FindWinner(map, node, gptr, &winner); /* Find best matching codebook */

	if (mapclass){ // in supervised mode
	  mapclass[map->codes[winner.codeno].x][map->codes[winner.codeno].y][node->label]++;
	}

	if (gctn < ginclude){
	  if (parameters->removed == 0 || (gptr->gnum+1) != parameters->removed){
#ifdef ONE_CLASS_ONLY
	    if (!strcmp(ONE_CLASS_ONLY, GetLabel(node->label))){
#endif
	      Adapt(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook*/
	      terror += winner.diff;
	      counter++;
#ifdef ONE_CLASS_ONLY
	    }
#endif
	  }
	}
      }
    }

    if (mapclass){ // In supervised mode, update class label of codebooks
      for (j = 0; j < map->xdim * map->ydim; j++){
	map->codes[j].label = 0;
	max = mapclass[map->codes[j].x][map->codes[j].y][0];
	for (l = 1; l < numlabels; l++){
	  if (mapclass[map->codes[j].x][map->codes[j].y][l] > max){
	    map->codes[j].label = l;
	    max = mapclass[map->codes[j].x][map->codes[j].y][l];
	  }
	}
      }
      //Now reset 'mapclass'
      for (x = 0; x < map->xdim; x++){
	for (y = 0; y < map->ydim; y++)
	  memset(mapclass[x][y], 0, numlabels * sizeof(UNSIGNED));
      }
    }
    
    if (parameters->contextual)
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    if (_save_then_exit_) { /* Save and exit if a interrupt signal was caught*/
      char fname[32];
      sprintf(fname, "interrupted%d.net", (int)getpid());
      free(parameters->onetfile);
      parameters->onetfile = strdup(fname); /* Assign alternate filename */
      fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
      break; /* Break the training cycle  */
    }

    /* Create a snapshot if required */
    if (parameters->snap.interval>0 &&!(map->iter%parameters->snap.interval)) {
      if (parameters->snap.file != NULL)
	SaveSnapShot(parameters);
      if (parameters->snap.command != NULL)
	system(parameters->snap.command);
    }

    //if (parameters->nice)
     // SleepOnHiLoad(); /* Sleep when system load is high */

    PrintProgress(map->iter); /* Print Progress */
    Hausekeeping(parameters);
  }
  StopProgressMeter();

  //  fclose(rofile);


#ifdef EXCLUSIVEMAPPING
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));
#endif



  t = tlen-1;
  radius_t = (parameters->radius - 1.0) * (float)(tlen - t)/(float)tlen;
  //  r_fuzzy = radius_t*expf(-t/tlen)/(float)den;
  //  sigma = converge+(float)r_fuzzy;				

  gctn = 0;
  /*Mutag special!!! Can be removed */

  /* Init state vector to zero */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      node = gptr->nodes[nnum]; /* This makes de-referencing easier */
      offset = gptr->ldim;
      memset(&node->points[offset], 0, (gptr->dimension-offset) * sizeof(FLOAT));
      node->x = 0;
      node->y = 0;
    }
  }

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
    gctn++;

    for (nnum = 0; nnum < gptr->numnodes; nnum++) {
      node = gptr->nodes[nnum]; /* This makes de-referencing easier */
                                                   			
      if (!parameters->contextual) {
	offset = gptr->ldim;
	memset(&node->points[offset], 0, (gptr->dimension-offset) * sizeof(FLOAT));

	for (k = 0; k < gptr->FanOut; k++) {
	  if (node->children[k] != NULL) {
	    node->points[offset++] = node->children[k]->x;
	    node->points[offset++] = node->children[k]->y;
	  }
	}

	for (k = 0u; k < node->numparents; k++) {
	  node->points[offset++] = node->parents[k]->x;
	  node->points[offset++] = node->parents[k]->y;
	}
	for (; offset < parameters->map.dim; offset++)
	  node->points[offset] = -1.0;

#ifndef NOORDER
	for (s1 = gptr->ldim; s1 < offset; s1 += 2){
	  for (s2 = offset-2; s2 > s1; s2 -= 2){
	    if (node->points[s2] != -1){
	      if (node->points[s1]*node->points[s1] + node->points[s1+1]*node->points[s1+1] > node->points[s2]*node->points[s2] + node->points[s2+1]*node->points[s2+1]){
		t1 = node->points[s2];
		t2 = node->points[s2+1];
		node->points[s2] = node->points[s1];
		node->points[s2+1] = node->points[s1+1];
		node->points[s1] = t1;
		node->points[s1+1] = t2;
	      }
	    }
	  }
	}
#endif
      }

      FindWinner(map, node, gptr, &winner);/* Refresh best matching codebook */

      if (gctn <= ginclude){
	if (parameters->removed == 0 || (gptr->gnum+1) != parameters->removed){
#ifdef ONE_CLASS_ONLY
	  if (!strcmp(ONE_CLASS_ONLY, GetLabel(node->label))){
#endif
	    fprintf(stdout, "(%s) %d %d %d %d %s %e trained\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);
	    //	    oflag = 1;

#ifdef ONE_CLASS_ONLY
	  }
	  else{
	    fprintf(stdout, "(%s) %d %d %d %d %s %e\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);
	  }
#endif
	}
	else{
	  fprintf(stdout, "(%s) %d %d %d %d %s %e removed\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);
	}
      }
      //      if (!oflag)
      else
	fprintf(stdout, "(%s) %d %d %d %d %s %e removepercent\n", gptr->gname, gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label), winner.diff);

    }
  }






  if (!_save_then_exit_)
    fprintf(stderr, "\033[01;32m%51s\033[00m\n", "[OK]");

  if (logfile != stdout)
    MyFclose(logfile);

  return 1;
}

/******************************************************************************
 Description: Main training routine for training a Fuzzy GraphSOM

 Return value: 0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainPMGraphSOM_Efficient(struct Parameters *parameters) {
  UNSIGNED i, k, offset, nnum, goptx, gopty, gopt_numxblocks;
  FLOAT alpha_t, radius_t, r_fuzzy, sigma;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  double PI = 3.1415926535897932; 
  FLOAT converge = (float)1/sqrtf(2*PI);
  FLOAT (*GetAlpha)(UNSIGNED, UNSIGNED, FLOAT);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,struct Winner *winner);
  void (*Adapt)(struct Graph *gptr, struct Map *map, struct Node *node,struct Winner *winner, FLOAT radius, FLOAT alpha_t);
  //  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);
  FLOAT (*ComputeDistance)(int bx, int by, int tx, int ty);

  ComputeDistance = ComputeHexaDistance;
  int kstepmode = 1;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;

  InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file       */

  /* Set the appropriate function for computing the learning rate     */
  GetAlpha = SetAlpha(parameters->alphatype);

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledianMemorySave; /* Use Eucledian distance        */

  /* Set the appropriate function for adapting the network parameters */
  Adapt = ReSetAdapt(parameters->map.neighborhood, parameters);

  /* Set the appropriate function for updating childens location in nodes */
  if (parameters->map.topology == TOPOL_VQ) { /* In VQ mode...   */
    // UpdateOffspringStates = UpdateChildrensLocationVQ; /* use ID value    */
    FindWinner = VQFindWinnerEucledian; /* Use Eucledian distance VQ mode */
    Adapt = VQAdapt; /* No topology = no neighborhood = VQ adapt mode   */
    VQSet_ab(parameters); /* Initialize auxillary variables a and b      */
  }

  if (parameters->contextual) {
    if (parameters->undirected)
      kstepmode = 0;
    else if (parameters->train->FanIn == 0) {
      fprintf(stderr, "Warning: No inlink available for contextual mode. Will fall back to normal mode.\n");
      //      UpdateOffspringStates = UpdateChildrensLocation;
      parameters->contextual = NO;
    }
    if (parameters->contextual) {
      fprintf(stderr, "Contextual mode: Training on single map is assumed\n");
      fprintf(stderr, "Will recompute states at every iteration!!\n");
      //      UpdateOffspringStates = UpdateChildrenAndParentLocation;
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);
    }
  }
  /*else
    UpdateOffspringStates = UpdateChildrensLocation;*/

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
	alpha_t = GetAlpha(t, tlen, parameters->alpha);
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
		  FLOAT d = ComputeDistance(xd, yd, xset[idx], yset[idx]);
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
	FindWinner(map, node, gptr, &winner); /* Find best matching codebook */
	//if (parameters->removed > 0 && gctn != parameters->removed){
	  Adapt(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook */
	  terror += winner.diff;
	  counter++;
	//}

      //Tuc special for calculate the compression ratio of PM-SOM, because this only apply for one graph dataset
      //int vdim = gptr->dimension;
     // int tuci;

      //printf("\nInput vector of dimension %d or node %d:\n",vdim,ldim);
      // for (tuci = 41;tuci < vdim; tuci++) {
      	  //printf("%f ",glb_ivec[tuci]);
		//  if (glb_ivec[tuci]!=0)
		//	  averagesub_nonzero_codebookelements=(double) averagesub_nonzero_codebookelements+1;
        //}

      }
    }
    //averagesub_nonzero_codebookelements/=11402;
    //averagesub_nonzero_codebookelements/=radius_t;
    //average_nonzero_codebookelements+=averagesub_nonzero_codebookelements;
    //fprintf(stderr, "Average at iter %d is K=%f, up to now K=%f\n",map->iter, averagesub_nonzero_codebookelements,average_nonzero_codebookelements/(map->iter+1));
    //End Tuc special

    if (parameters->contextual)
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    if (_save_then_exit_) { /* Save and exit if a interrupt signal was caught */
      char fname[32];
      sprintf(fname, "interrupted%d.net", (int)getpid());
      free(parameters->onetfile);
      parameters->onetfile = strdup(fname); /* Assign alternate filename */
      fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
      break; /* Break the training cycle  */
    }

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
  if (!_save_then_exit_)
    fprintf(stderr, "\033[01;32m%51s\033[00m\n", "[OK]");

  if (logfile != stdout)
    MyFclose(logfile);

  //Begin Tuc special
  //average_nonzero_codebookelements/=map->iter;
  //fprintf(stderr, "Average number use for CR K=%f\n",average_nonzero_codebookelements);

  FILE *file_res = fopen(fileresult,"w");
  if (file_res==NULL) fprintf(stderr, "Error open file result\n");

  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
	  for (nnum = 0; nnum < gptr->numnodes; nnum++) {
		  compute=0;
		  FLOAT winval = 1;
		  node = gptr->nodes[nnum];
		  alpha_t = GetAlpha(t, tlen, parameters->alpha);
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
							  FLOAT d = ComputeDistance(xd, yd, xset[idx], yset[idx]);
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
				
		  FindWinner(map, node, gptr, &winner);
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
 Description: Main training routine for training a GraphSOM

 Return value: 0 if training could not be initiated, 1 otherwise
******************************************************************************/
int TrainGraphSOM(struct Parameters *parameters) {
  UNSIGNED i, k, offset, nnum, goptx, gopty, gopt_numxblocks;
  FLOAT alpha_t, radius_t;
  struct Map *map;
  struct Node *node;
  struct Graph *gptr;
  FILE *logfile;
  FLOAT (*GetAlpha)(UNSIGNED, UNSIGNED, FLOAT);
  void (*FindWinner)(struct Map *map, struct Node *node, struct Graph *gptr,
		     struct Winner *winner);
  void (*Adapt)(struct Graph *gptr, struct Map *map, struct Node *node,
		struct Winner *winner, FLOAT radius, FLOAT alpha_t);
  //  void (*UpdateOffspringStates)(struct Graph *gptr, struct Node *node);

  int kstepmode = 1;
  struct Winner winner;
  UNSIGNED tlen, t;
  int counter;
  FLOAT terror;
  char ofname[256];
  FILE *rofile = NULL;

  InstallHandlers(); /* Install interrupt handler */
  logfile = MyFopen(parameters->logfile, "w"); /* Open log-file       */

  /* Set the appropriate function for computing the learning rate     */
  GetAlpha = SetAlpha(parameters->alphatype);

  /* Set the appropriate function for computing the winner codebook   */
  FindWinner = FindWinnerEucledian; /* Use Eucledian distance        */

  /* Set the appropriate function for adapting the network parameters */
  Adapt = SetAdapt(parameters->map.neighborhood);

  /* Set the appropriate function for updating childens location in nodes */
  if (parameters->map.topology == TOPOL_VQ) { /* In VQ mode...   */
    // UpdateOffspringStates = UpdateChildrensLocationVQ; /* use ID value    */
    FindWinner = VQFindWinnerEucledian; /* Use Eucledian distance VQ mode */
    Adapt = VQAdapt; /* No topology = no neighborhood = VQ adapt mode   */
    VQSet_ab(parameters); /* Initialize auxillary variables a and b      */
  }

  if (parameters->contextual) {
    if (parameters->undirected)
      kstepmode = 0;
    else if (parameters->train->FanIn == 0) {
      fprintf(stderr, "Warning: No inlink available for contextual mode. Will fall back to normal mode.\n");
      //      UpdateOffspringStates = UpdateChildrensLocation;
      parameters->contextual = NO;
    }
    if (parameters->contextual) {
      fprintf(stderr, "Contextual mode: Training on single map is assumed\n");
      fprintf(stderr, "Will recompute states at every iteration!!\n");
      //      UpdateOffspringStates = UpdateChildrenAndParentLocation;
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);
    }
  }
  /*else
    UpdateOffspringStates = UpdateChildrensLocation;*/

  InitProgressMeter(parameters->rlen); /* Initialize the progress meter */
  fprint(stderr, "Training GraphSOM......"); /* Print what is being done  */
  map = &parameters->map; /* This makes de-referencing easier         */
  tlen = 0; /* Compute the total number of update steps */
  for (gptr = parameters->train; gptr != NULL; gptr = gptr->next)
    tlen += gptr->numnodes;
  tlen = tlen * (parameters->rlen - map->iter);

  goptx = map->goptx;
  gopty = map->gopty;
  gopt_numxblocks = map->xdim / goptx;

  /* Now lets get started (to train map) */
  t = 0;
  for (i = map->iter; i < parameters->rlen; i++) {

#ifdef EXCLUSIVEMAPPING
    memset(parameters->map.activation, 0, parameters->map.xdim * parameters->map.ydim *sizeof(FLOAT));

#endif

    if (rofile)
      fclose(rofile);
    sprintf(ofname, "giter%d.dat", i);
    rofile = fopen(ofname, "w");


    if (parameters->graphorder == 1)
      parameters->train = RandomizeGraphOrder(parameters->train);

    /* //This is only good for undirected graphs...not yet handled properly
       if (parameters->nodeorder == 1)
       parameters->train = RandomizeNodeOrder(parameters->train);
    */

    counter = 0;
    terror = 0.0; /* Init train error */

    gptr = parameters->train;

    for (gptr = parameters->train; gptr != NULL; gptr = gptr->next) {
      //      fprintf(stderr, "graph %s: ", gptr->gname);
      for (nnum = 0; nnum < gptr->numnodes; nnum++) {
	//	fprintf(stderr, "node %d\n", nnum);
	node = gptr->nodes[nnum]; /* This makes de-referencing easier */
	alpha_t = GetAlpha(t, tlen, parameters->alpha);
	radius_t = 1.0 + (parameters->radius - 1.0) * (float)(tlen - t)
	  /(float)tlen;
	t++;

	if (!parameters->contextual) {
	  offset = gptr->ldim;
	  memset(&node->points[offset], 0, (gptr->dimension-offset)
		 * sizeof(FLOAT));

	  for (k = 0; k < gptr->FanOut; k++) {
	    if (node->children[k] != NULL) {
	      if (offset + node->children[k]->x/goptx
		  + node->children[k]->y/gopty
		  * gopt_numxblocks
		  >= gptr->dimension)
		fprintf(stderr, "sffgfdg\n");
	      node->points[offset + node->children[k]->x/goptx + node->children[k]->y/gopty * gopt_numxblocks]++;
	    } else
	      break; /* No more offsprings after a NULL-pointer expected */
	  }

	  for (k = 0u; k < node->numparents; k++) {
	    node->points[offset + node->parents[k]->x/goptx + node->parents[k]->y/gopty * gopt_numxblocks]++;
	  }
	}
	//for(k=0;k<gptr->dimension;k++)
	// fprintf(stderr, "%f ", node->points[k]);
	//fprintf(stderr, "\n");
	FindWinner(map, node, gptr, &winner); /* Find best matching codebook */

	//	if (i == parameters->rlen - 1)
	  fprintf(rofile, "%s %d %d %d %s\n", gptr->gname, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label));
	  fprintf(stderr, "%s %d %d %d %s\n", gptr->gname, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y, GetLabel(node->label));

	Adapt(gptr, map, node, &winner, radius_t, alpha_t);/* update codebook */
	terror += winner.diff;
	counter++;
      }
    }
    if (parameters->contextual)
      K_Step_Approximation(&parameters->map, parameters->train, kstepmode);

    map->iter++;
    fprintf(logfile, "%f\n", terror/counter); /* Print normalized q-error */
    fflush(logfile);

    if (_save_then_exit_) { /* Save and exit if a interrupt signal was caught */
      char fname[32];
      sprintf(fname, "interrupted%d.net", (int)getpid());
      free(parameters->onetfile);
      parameters->onetfile = strdup(fname); /* Assign alternate filename */
      fprintf(stderr, "\nSaving net to '%s'\n", parameters->onetfile);
      break; /* Break the training cycle  */
    }

    /* Create a snapshot if required */
    if (parameters->snap.interval>0 &&!(map->iter
					%parameters->snap.interval)) {
      if (parameters->snap.file != NULL){
	SaveSnapShot(parameters);
      }
      if (parameters->snap.command != NULL)
	system(parameters->snap.command);
    }

    //if (parameters->nice)
    //  SleepOnHiLoad(); /* Sleep when system load is high */

    PrintProgress(map->iter); /* Print Progress */
    Hausekeeping(parameters);
  }
  StopProgressMeter();
  fclose(rofile);
  if (!_save_then_exit_)
    fprintf(stderr, "\033[01;32m%51s\033[00m\n", "[OK]");

  if (logfile != stdout)
    MyFclose(logfile);

  return 1;
}

/******************************************************************************
 Description: Chooses the right function to trains the various types of SOM
 that are supported by this software.

 Return value: The return value produced by the corresponding training function.
 This would normally be 0 if training could not be initiated, or
 1 otherwise.
******************************************************************************/
int TrainMap(struct Parameters *parameters) {

#ifdef EXCLUSIVEMAPPING
  if (parameters->memorysave){
    fprintf(stderr, "\nWARNING: Can not use exclusive mapping in memorysave mode!!\n");
  }
  else {
    fprintf(stderr, "\nWARNING: Using exclusive mapping!!\n");
    parameters->map.activation = (float*) calloc(parameters->map.xdim * parameters->map.ydim,sizeof(FLOAT));
  }
#endif

  if (parameters->map.type == TYPE_GRAPHSOM) {
    if (parameters->fuzzy){
      if(parameters->memorysave)
    	  return TrainPMGraphSOM_Efficient(parameters);
      if (parameters->compact)
    	  return TrainPMGraphSOMCompact(parameters);
      return TrainPMGraphSOM(parameters);
    }
    return TrainGraphSOM(parameters);
  }
  else
    return TrainSOMSD(parameters);
}

/* END OF FILE */

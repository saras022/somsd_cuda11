

/************/
/* Includes */
/************/
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"
#include "data.h"
#include "fileio.h"
#include "train_cpu.h"
#include "utils.h"
#include "testsom.h"

/*************/
/* Constants */
/*************/
#define CONTEXT        0x00000001
#define CONTEXTUAL     0x00000002
#define MAPNODETYPE    0x00000004
#define MAPNODELABEL   0x00000008
#define MAPGRAPH       0x00000010
#define MAPNODE        0x00000020
#define SHOWGRAPH      0x00000040
#define SHOWCLUSTERING 0x00000080
#define SHOWSUBGRAPHS  0x00000100
#define PRECISION      0x00000200
#define RETRIEVALPERF  0x00000400
#define CLASSIFY       0x00000800
#define CLASSIFYNODES  0x00001000
#define ANALYSE        0x00002000
#define BALANCE        0x00004000
#define DISTANCES      0x00008000
#define WEBSOM         0x00010000
#define TRUNCATE       0x00020000
#define INEX2007       0x00040000
#define INEX2008       0x00080000
#define QUIET          0x01000000
#define REPORTONLY     0x02000000
#define OTHER          0x04000000
#define LABELED        0x08000000
#define CLASSINEX2008  0x06000000

struct RGB{
  int red;
  int green;
  int blue;
};

struct VMap{
  UNSIGNED max;
  UNSIGNED **activation;
  UNSIGNED numclasses;
  UNSIGNED ***classes;
  UNSIGNED **winnerclass;
};

struct AllHits{
  struct Graph *graph;
  struct Node *node;
  char *structID;
  char *substructID;
  struct AllHits *next;
};

int KstepEnabled = 0;
int winnerclass_train[5000][5000]={0};


int **ComputeConfusionMatrix(int xdim, int ydim, struct VMap *vmap) {
  int i, row, ond, offd;
  int x, y;
  int r;
  int **matrix;
  int *lo;
  int tp, tn, fp,fn;

  matrix = (int**)malloc(vmap->numclasses * sizeof(int*));
  for (i = 0; i < vmap->numclasses; i++)
	  matrix[i] = (int*)calloc(vmap->numclasses, sizeof(int));

  for (y = 0; y < ydim; y++) {
	  for (x = 0; x < xdim; x++) {
		  if (vmap->classes[y][x] == NULL)
			  continue;

		  row = vmap->winnerclass[y][x]-1;
		  for (i = 0; i < vmap->numclasses; i++)
			  matrix[i][row] += vmap->classes[y][x][i];
	  }
  }

  ond = 0;
  offd = 0;
  lo = GetSortedLabelIndex();
  for (x = 0; x < vmap->numclasses; x++)
	  fprintf(stdout, " %d", lo[x]);
  fprintf(stdout, "\n");
  for (x = 0; x < vmap->numclasses; x++)
	  fprintf(stdout, " %s", GetLabel(lo[x]));
  fprintf(stdout, "\n");
  for (y = 0; y < vmap->numclasses; y++) {
	  r = 0;
	  for (x = 0; x < vmap->numclasses; x++) {
		  r+= matrix[lo[y]-1][lo[x]-1];
		  if (x == y)
			  ond += matrix[lo[y]-1][lo[x]-1];
		  else
			  offd += matrix[lo[y]-1][lo[x]-1];
		  fprintf(stdout, " %4d", matrix[lo[y]-1][lo[x]-1]);
	  }
	  if (r > 0)
		  fprintf(stdout, " #%.4f", (float)100*matrix[lo[y]-1][lo[y]-1]/r);
	  fprintf(stdout, "\n");
  }
  fprintf(stdout, "On diagonal: %d\n", ond);
  fprintf(stdout, "Off diagonal: %d\n", offd);
  if (ond > 0)
	  fprintf(stdout, "Confusion: %f\n", (float)offd*100/ond);


  //For mutag 2-class problems
  fprintf(stdout, "Train ACC: %f ", (float)ond*100/(ond+offd));
  /*tp = matrix[lo[1]-1][lo[1]-1];
  fn = matrix[lo[1]-1][lo[2]-1];
  fp = matrix[lo[2]-1][lo[1]-1];
  float	re = (float)tp/(tp+fn);
  float	p = (float)tp/(tp+fp);
  float	f1 = 2 * (float)(re*p)/(p+re);
  fprintf(stdout, "F1: %f\n", f1);
*/
  return matrix;
}
void GetClusterID(struct Map map, struct Graph *graph, struct VMap *vmap) {
  UNSIGNED *frequency;
  int numlabels;
  int n, max, id, x, y;
  struct Node *node;
  struct Graph *gptr;
  int xdim, ydim;
  int noactive = 0;

  xdim = map.xdim;
  ydim = map.ydim;
  vmap->winnerclass = (UNSIGNED**)MyMalloc(ydim * sizeof(UNSIGNED*));
  vmap->classes = (UNSIGNED***)MyMalloc(ydim * sizeof(UNSIGNED**));
  for (n = 0; n < ydim; n++) {
      vmap->winnerclass[n] = (UNSIGNED*)MyCalloc(xdim, sizeof(UNSIGNED));
      vmap->classes[n] = (UNSIGNED**)MyCalloc(xdim, sizeof(UNSIGNED*));
  }

  for (y = 0; y < ydim; y++) {
     for (x = 0; x < xdim; x++) {
    	 vmap->winnerclass[y][x] =-1;
     }
  }
  if (graph == NULL)
     return;

  numlabels = GetNumLabels();
  vmap->numclasses = numlabels;
  //fprintf(stderr, "number of classes = %d\n", numlabels);

  for (y = 0; y < ydim; y++) {
     for (x = 0; x < xdim; x++) {
    	 if (vmap->activation[y][x] == 0)
    		 continue;
    	 frequency = (int*) MyCalloc(numlabels, sizeof(int));
    	 for (gptr = graph; gptr != NULL; gptr = gptr->next) {
    		 for (n = 0; n < gptr->numnodes; n++) {
    			 node = gptr->nodes[n];
    			 if (node->x == x && node->y == y)
    				 if (GetLabel(node->label) != NULL)
    					 if (strcmp(GetLabel(node->label), "*"))
    						 frequency[node->label-1]++;
    		 }
    	 }
    	 id = -1;
    	 max = 0;
    	 for (n = 0; n < numlabels; n++) {
    		 if (frequency[n] > max) {
    			 max = frequency[n];
    			 id = n;
    		 }
    	 }
    	 if (id >= 0) {
    		 vmap->winnerclass[y][x] = id+1;

    		 winnerclass_train[y][x] = id+1;

    		 vmap->classes[y][x] = frequency;
    	 } else {
    		 noactive++;
    		 if (noactive < 10) {
    			 for (n = 1; n <= numlabels; n++)
    				 fprintf(stderr, "%s->%d ", GetLabel(n), frequency[n-1]);
    			 fprintf(stderr, "\n");
    		 }
    		 free(frequency);
    	 }
     }
  }
  if (noactive)
	  fprintf(stderr, "There were %d activated neurons without label\n", noactive);

  /*for (y = 0; y < ydim; y++) {
    for (x = 0; x < xdim; x++) {
    	fprintf(stderr, "Class at (%d,%d) is %d\n",x,y,vmap->winnerclass[y][x]);
    }
  }*/
}
//Compute activation of every node of the map, and the maximum activation of any node on the map. Result is stored in vmap.activation[y][x], and vmap.max
struct VMap GetHits(int xdim, int ydim, struct Graph *graph, int mode) {

  struct VMap vmap = { 0 };
  struct Graph *gptr;
  struct Node *node;
  int n, N, nN, hits, x, y, nac;
  vmap.activation = (UNSIGNED**)MyMalloc(ydim * sizeof(UNSIGNED*));
  for (n = 0; n < ydim; n++)
	  vmap.activation[n] = (UNSIGNED*)MyCalloc(xdim, sizeof(FLOAT));

  hits = 0;
  nac=0;
  vmap.max = 0;
  N = 0;
  nN = 0;
  for (gptr = graph; gptr != NULL; gptr = gptr->next) {
    nN += gptr->numnodes;
    for (n = 0; n < gptr->numnodes; n++) {
      node = gptr->nodes[n];

      if (mode & ROOT && IsRoot(node)) {

    	  N++;
    	  //	if(!strcmp(GetLabel(node->label), "*"))
    	  //	  printf("Root without target. Graph:%s,Node:%d\n",gptr->gname,node->nnum);
    	  vmap.activation[node->y][node->x] += 1;
    	  if (vmap.activation[node->y][node->x] > vmap.max)
    		  vmap.max = vmap.activation[node->y][node->x];
    	  continue;
      } else if (mode & LEAF && IsLeaf(node)) {

    	  vmap.activation[node->y][node->x] += 1;
    	  if (vmap.activation[node->y][node->x] > vmap.max)
    		  vmap.max = vmap.activation[node->y][node->x];
    	  continue;
      } else if (mode & INTERMEDIATE && IsIntermediate(node)) {

    	  vmap.activation[node->y][node->x] += 1;
    	  if (vmap.activation[node->y][node->x] > vmap.max)
    		  vmap.max = vmap.activation[node->y][node->x];
    	  continue;
      } else if(mode & OTHER /*&& !strcmp(GetLabel(node->label), "*")*/){

    	  vmap.activation[node->y][node->x] += 1;
    	  if (vmap.activation[node->y][node->x] > vmap.max)
    		  vmap.max = vmap.activation[node->y][node->x];
    	  continue;
      } else if(mode & LABELED && strcmp(GetLabel(node->label), "*")){
    	  //printf("OK06 %d %d \n", node->y, node->x);

    	  vmap.activation[node->y][node->x] += 1;

    	  if (vmap.activation[node->y][node->x] > vmap.max)

    		  vmap.max = vmap.activation[node->y][node->x];

    	  continue;
      }
    }
  }

  for (y = 0; y < ydim; y++) {
	  for (x = 0; x < xdim; x++) {
		  if (vmap.activation[y][x] != 0) {
			  nac+=vmap.activation[y][x];
			  hits++;
		  }
	  }
  }

  //fprintf(stderr, "CT_CODE OFF!\n");
  /*
    ct = 0;
    for (y = 1; y < ydim-2; y++){
    for (x = 1; x < xdim-2; x++){
    if (vmap.activation[y][x] > 0){
    for (sx = x-1; sx < x+2; sx++){
    for (sy = y-1; sy < y+2; sy++){
    if ( ((sx+sy) % 2) && vmap.activation[sy][sx] == 0){
    vmap.activation[sy][sx]=-1;
    }
    }
    }
    }
    }
    }
    for (y = 0; y < ydim; y++){
    for (x = 0; x < xdim; x++){
    if (vmap.activation[y][x] == -1){
    vmap.activation[y][x] = 1;
    ct++;
    }
    }
    }
    fprintf(stderr, "ct %d added\n", ct);
  */
  /*if (!(mode & QUIET)) {
    fprintf(stderr, "Neurons activated: %d\n", hits);
    //fprintf(stderr, "Nodes maped: %d\n", nac);
    fprintf(stderr, "Max activation strength: %d\n", vmap.max);
    fprintf(stderr, "Compression ratio: %f (root nodes only)\n",(float)N/hits);
    fflush(stderr);
    }*/
  if (mode & REPORTONLY) {
    for (n = 0; n < ydim; n++)
      free(vmap.activation[n]);
    free(vmap.activation);
  }
  //  else{
  //    fprintf(stderr, "Setting vmax to 10899\n");
  //    vmap.max = 10899;
  //    }

  return vmap;
}
void ComputeNetPerformance(struct Parameters parameters) {
  int flag = 1;
  struct VMap vmap;
  //struct VMap tvmap; //for the test set
  float R, P;
  int C, n, i, nnum = 0, tot_gnum;
  int winnerx, winnery;
  struct Graph *gptr;
  struct Node *node;
  struct Winner winner = { 0 };
  struct Map *map;
  int prech = 0, curch=0;
  int winsize = 8;
  int mutag[320]; /*mutag special can be removed*/
  int b, bmax, goodmutag;

  /*
  if (parameters.test == NULL) {
    printf("crp-Warning: No test file given. Will use training data for testing.\n");
    parameters.test = parameters.train;
    flag = 1;
  }
*/
  map = &parameters.map;


#ifdef EXCLUSIVEMAPPING
    memset(parameters.map.activation, 0, parameters.map.xdim * parameters.map.ydim *sizeof(FLOAT));
#endif


    vmap = GetHits(parameters.map.xdim, parameters.map.ydim, parameters.train, LABELED);

    GetClusterID(parameters.map, parameters.train, &vmap);

    ComputeConfusionMatrix(parameters.map.xdim, parameters.map.ydim, &vmap);

    //tvmap = TestGetHits(parameters.map.xdim, parameters.map.ydim, parameters.train, ROOT);
    //GetClusterID(parameters.map, parameters.train, &tvmap);
    // ComputeConfusionMatrix(parameters.map.xdim, parameters.map.ydim, &tvmap);

  /* if (parameters.test != NULL){
     tvmap = TestGetHits(parameters.map.xdim, parameters.map.ydim, parameters.test, LABELED);
     GetClusterID(parameters.map, parameters.test, &tvmap);

     R = 0.0;
     C = 0;
     n = 0;
     //For (every node in the test set){
     for (gptr = parameters.test; gptr != NULL; gptr = gptr->next) {
       for (nnum = 0; nnum < gptr->numnodes; nnum++) {
    	   node = gptr->nodes[nnum];

    	   FindWinnerEucledian(map, node, gptr, &winner);
    	   winnerx = map->codes[winner.codeno].x;
    	   winnery = map->codes[winner.codeno].y;
    	   node->x = winnerx;
    	   node->y = winnery;
    	   //fprintf(stdout,"%d %d %d %d %s\n", gptr->gnum, node->nnum, map->codes[winner.codeno].x, map->codes[winner.codeno].y,GetLabel(node->label));
    	   //if (strcmp(GetLabel(node->label), "*")) //If LABELED
    		   //	if (!IsRoot(node))
    		   //n++;
    	  //if (classifyflag == 1)
    	    //fprintf(stdout, "Graph:%s:(%d,%d)\n", gptr->gname, vmap.winnerclass[winnery][winnerx], node->label);
    	  //else if (classifyflag == 2)
    	    //fprintf(stdout, "Graph:%s:Node:%d:%s:%d:(%d,%s)\n", gptr->gname, node->nnum, GetLabel(vmap.winnerclass[winnery][winnerx]), vmap.winnerclass[winnery][winnerx], node->label, GetLabel(node->label));
    	  if (node->label == vmap.winnerclass[winnery][winnerx]) {
    	    //	fprintf(stdout, "G\n");
    	    C++;
    	  }
    	  n++;

       }
     }
	printf("Test ACC: %f\n", (float)100.0*C/n);

      //ComputeConfusionMatrix(parameters.map.xdim, parameters.map.ydim, &tvmap);
   }*/

  if (flag)
    parameters.test = NULL; /* Restore pointer to testset */
}

#ifndef COMMON_H_DEFINED
#define COMMON_H_DEFINED

#include <limits.h>
#include <float.h>

#define PROG_VERSION "0.9.1"

//This is temporarily used for nsight

/* Define precision used by package */
#if defined(USE_LONG_DOUBLE_PRECISION)
#define FLOAT long double
#define MAX_FLOAT LDBL_MAX
#define MIN_FLOAT -LDBL_MAX
#define EPSILON LDBL_EPSILON
#define UNSIGNED long int /* signed int seem faster than unsigned int */
#define MAX_UNSIGNED LONG_MAX
#elif defined(USE_DOUBLE_PRECISION)
#define FLOAT double
#define MAX_FLOAT DBL_MAX
#define MIN_FLOAT -DBL_MAX
#define EPSILON DBL_EPSILON
#define UNSIGNED int  /* signed int operations seem faster than unsigned*/
#define MAX_UNSIGNED LONG_MAX
#else
#define FLOAT float
#define MAX_FLOAT FLT_MAX
#define MIN_FLOAT -FLT_MAX
#define EPSILON FLT_EPSILON
#define UNSIGNED int  /*  On intel-cpu's int operations seem faster than uint*/
#define MAX_UNSIGNED INT_MAX
#endif

/* Network save modes */
#define SAVE_REGULAR 0x00001   //Save network to designated file
#define SAVE_BACKUP  0x00002   //Make a backup of the network
#define SAVE_TMP     0x00004   //Save network to a emporary file

//Color codes for printf
#define KNRM  "\x1B[0m"
#define KBOLT "\033[1m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET "\033[0m"
#define PRT_NOTE  "\x1B[36mNOTE: \033[0m"
#define PRT_WARNING  "\x1B[34mWARNING: \033[0m"
#define PRT_ERROR "\x1B[31mERROR: \033[0m"


/*  Kernel methods */
#define KERNEL_NONE    0  /* None          */
#define KERNEL_SIMPLE  1  /* Simple kernel */
#define KERNEL_FULL    2  /* Full kernel   */

/* Supervised modes */
#define KOHONEN     1
#define INHERITANCE 2
#define REJECT      3
#define LOCALREJECT 4

/* Map types */
#define TYPE_UNKNOWN  0
#define TYPE_SOM      1
#define TYPE_SOMSD    2
#define TYPE_CSOMSD   3
#define TYPE_GRAPHSOM 4

/* topology types */
#define TOPOL_RECT    1  /* Rectagonal topology  */
#define TOPOL_HEXA    2  /* Hexagonal topology   */
#define TOPOL_OCT     3  /* Octagonal topology   */
#define TOPOL_VQ      4  /* No topology (VQ mode)*/


/* Network initialization modes */
#define INIT_LINEAR  0x0001      /* Linear initialization       */
#define INIT_RANDOM  0x0002      /* Random value initialization */
#define INIT_DEFAULT INIT_RANDOM /* Default initialization mode */

/* neighborhood types */
#define NEIGH_BUBBLE   1  /* Neighborhood defined strictly by radius */
#define NEIGH_GAUSSIAN 2  /* Neighborhood defined by a Gaussian bell */
#define NEIGH_NONE     3  /* No Neighborhood is defined in VQ mode   */

/* alpha function types */
#define ALPHA_LINEAR      1  /* Linear decreasing alpha            */
#define ALPHA_EXPONENTIAL 2  /* Exponential decreasing alpha       */
#define ALPHA_SIGMOIDAL   3  /* Sigmoidal decrease curve for alpha */
#define ALPHA_CONSTANT    4  /* Alpha remains constant             */

/* Node types */
#define NONE          0x00  /* Undefined            */
#define LEAF          0x01  /* A leaf node          */
#define ROOT          0x02  /* A root node          */
#define INTERMEDIATE  0x04  /* An intermediate node */

/* Vector components */
#define DATALABEL    1  /* A numeric label associated with a node        */
#define CHILDSTATES  2  /* Network state for a nodes' offsprings         */
#define PARENTSTATES 3  /* Network state for the parents of a node       */
#define TARGETS      4  /* Target vector component of vector             */

/* Platform environment */
#define GPU_MODE  0x010000

/* Parameter flags */
#define NO           0x0000000
#define UNKNOWN      0x0000000
#define YES          0x0000001
#define FAST_MODE    0x0000002
#define FAST_FORWARD 0x0000004
#define PAUSE        0x0000008
#define EVALUATE     0x0000010

/* Structure to hold sparse vectors */
struct SparseV{          /* For sparce vectors                 */
  UNSIGNED pos;          /* Position of the non-zero value     */
  FLOAT value;           /* The actual value                   */
  struct SparseV *next;  /* Pointer to the next non-zero value */
};

/* Structure to hold a single graph node */
struct Node{
  union{
    FLOAT *points;         /* Pointer to concatenated label,coordinates, etc */
    struct SparseV *sparse;/* Same as above if vector is sparse */
  };
  UNSIGNED nnum;           /* Logical number of node            */
  UNSIGNED depth;          /* Depth of node (for LEAF: depth=0) */
  FLOAT *mu;               /* Set of weights for this node      */
  union{
    struct{
      int x, y;            /* Winner coordinate of codebook for this node */
    };
  int winner;            /* Winner codebook ID (in VQ mode)             */
  };
  UNSIGNED label;          /* Index of Symbolic class label if available  */
  UNSIGNED numparents;     /* Number of pointers to parents for this node */
  UNSIGNED numchildren;    /* Number of pointers to children for this node */
  struct Node **parents;   /* Pointer to parents of this node   */
  union{
    struct Node **children; /* Directed links: Pointer to children of node */
    struct Node **neighbors;/* Pointer to neighbors of this node, used for */
                            /* undirected links                            */
  };
};

/* Structure for a single graph */
struct Graph{
  UNSIGNED numnodes;     /* Number of nodes in this graph   */
  struct Node **nodes;   /* Array of pointers to nodes      */
  char *gname;           /* Pointer to name of graph (if vailable) */
  UNSIGNED gnum;         /* A logical ID-number for this graph     */
  UNSIGNED ldim;         /* Dimension of datalabel attached to each node */
  UNSIGNED dimension;    /* Dimension of 'points' vector of each node    */
  UNSIGNED FanOut;       /* Max. outdegree of this graph    */
  UNSIGNED FanIn;        /* Max. indegree of this graph     */
  UNSIGNED tdim;         /* Dimension of target vector      */
  UNSIGNED depth;        /* Max. depth of this graph        */
  struct Graph *next;    /* Pointer to next graph structure */
};


struct Codebook{
  FLOAT *points;         /* Pointer to codebook vector        */
  union{
    struct{
      int x, y;          /* Coordinates of codebook entry     */
    };
    struct{
      FLOAT a;           /* Summarize child state in VQ mode  */
      FLOAT b;           /* Summarize parent state in VQ mode */
    };
  };
  UNSIGNED label;        /* Index of class label (in supervised mode only) */
};

struct Map{    /* Structure for the map data */
  struct Codebook *codes;  /* Pointer to codebook entries        */
#ifdef EXCLUSIVEMAPPING
  FLOAT *activation;  /* Pointer to codebook entries        */
#endif
  UNSIGNED dim;            /* Dimension of codebook entries      */
  int xdim;                /* horizontal dimension of map        */
  int ydim;                /* vertical dimension of map          */
  UNSIGNED iter;           /* Iteration number the map is in     */
  UNSIGNED topology;       /* Typology of the map                */
  UNSIGNED neighborhood;   /* neighborhood type                  */
  UNSIGNED type;           /* network type som,somsd,gsom,csdomsd*/
  UNSIGNED goptx;          /* horizontal size of group in GSOM   */
  UNSIGNED gopty;          /* vertical size of group in GSOM     */
};

struct Parameters{
  char *inetfile;    /* Name of network file (input)               */
  char *onetfile;    /* Name of file to which to save the trained network. */
  char *datafile;    /* File that contains the training data       */
  char *validfile;   /* File that contains validation data         */
  char *testfile;    /* File that contains test data               */
  char *logfile;     /* File to which to write logging information */
  UNSIGNED rlen;     /* Number of training iterations              */
  UNSIGNED radius;   /* Size of initial neighborhood radius        */
  FLOAT alpha;       /* Initial learning rate                      */
  FLOAT beta;        /* Rejection rate (for some supervised modes) */
  FLOAT *mu1, *mu2, *mu3, *mu4; /* Weight value vector             */
  int nmu1, nmu2, nmu3, nmu4; /* Size of weight value vector       */
  FLOAT initmu1, initmu2; /* Initial Weight values                 */
  UNSIGNED seed;     /* Seed for random number generator           */
  UNSIGNED ncpu;     /* Number of parallel tasks to use            */
  UNSIGNED removed;  /* Graph-ID to ignore during training         */
  UNSIGNED percentremoved;  /* % of Graphs to ignore during training*/
  UNSIGNED precision;  /* Precision level;disable some optimizations*/
  int kfoldtime; /*Use for training k fold cross validation*/
  int mode;   /* CPU (0) or GPU (1) */
  int device; /* GPU device number  */

  struct{    /* Structure used for snapshot purposes */
    int interval;    /* Take a snapshot every so often       */
    char *file;      /* Save snapshot to this file           */
    char *command;   /* Execute this command every time a snapshot is taken. */
  } snap;

  /* Bitfields & flags */
  unsigned batch:1;     /* Batch mode (0=no, 1=yes)                  */
  unsigned momentum:1;  /* With momentum term (0=no, 1=yes)          */
  unsigned super:3;     /* Supervised mode (KOHONEN, REJECTION, etc.)*/
  unsigned alphatype:3; /* Type of alpha value decrease (LINEAR, GAUSS, etc. */
  unsigned kernel:2;    /* Kernel method (SIMPLE or FULL)            */
  unsigned nice:1;      /* Nice mode (0=off, 1=on)                   */
  unsigned verbose:1;   /* Verbose mode (0=no, 1=yes)                */
  unsigned contextual:1;/* Single map contextual mode (0=no, 1=yes)  */
  unsigned nodeorder:1; /* Randomize order of nodes (0=no, 1=yes)    */
  unsigned graphorder:1;/* Randomize order of graphs (0=no, 1=yes)   */
  unsigned undirected:1;/* Temporary use until undirected graph file format is supported */
  unsigned fuzzy:1;     /* Fuzzy map (0=no, 1=yes) */
  unsigned compact:1;   /* Compact state vector (0=no, 1=yes) */
  unsigned memorysave:1;/* memory save mode (0=no, 1=yes)*/
  int savenet;          /* Indicate whether network should be saved */
  unsigned flags;
  int argc;             /* Number of command line arguments  */
  char **argv;          /* Pointer to command line arguments */

  struct Graph *train;  /* Pointer to training data    */
  int **winnerclass_train; /*store the winner class at each coordinate*/
  struct Graph *valid;  /* Pointer to validation data  */
  struct Graph *test;   /* Pointer to test data        */
  struct Map map;       /* The Self-Organizing Map     */
};

struct Winner { /* Structure used to store best matching codebook */
  UNSIGNED codeno;   /* Index number of best matching codebook */
  FLOAT diff;        /* The error value with this node    */
};

/* Macros */
#ifndef max
#define max(x,y) ((x) > (y) ? (x) : (y))/* Returns the maximum of two values */
#endif
#ifndef min
#define min(x,y) ((x) < (y) ? (x) : (y))/* Returns the minimum of two values */
#endif
#ifndef SQR
#define SQR(s)  ((s) * (s))             /* Compute the squared value         */
#endif

/* Prototypes */
void Hausekeeping(struct Parameters *param);
UNSIGNED GetTopologyID(char *cptr, UNSIGNED *uval);  /*Get ID of topology    */
UNSIGNED GetNeighborhoodID(char *ptr, UNSIGNED *val);/*Get ID of neighborhood*/
UNSIGNED GetAlphaType(char *atype);               /* Get alpha type          */
UNSIGNED GetMapTypeID(char *mtype, UNSIGNED *val);/* Get map type, and a     */
UNSIGNED GetMapType(char *mtype);                 /* Content aware version   */
void VQSet_ab(struct Parameters *parameters);     /* Init a and b in VQ mode */
const char *GetTopologyName(UNSIGNED ID);         /* Get name of topology    */
const char *GetNeighborhoodName(UNSIGNED ID);     /* Get name of neighborhood*/
const char *GetMapTypeName(UNSIGNED type);        /* Get name of type of map */
UNSIGNED InitCodes(struct Map *, struct Graph *, UNSIGNED mode);/* init a map*/
UNSIGNED InitGsomCodes(struct Map *, struct Graph *, UNSIGNED mode);/* init a gsom_map*/
UNSIGNED InitGsomCodesCompact(struct Map *, struct Graph *, UNSIGNED mode);/* init a gsom_map*/
void SuggestMu(struct Parameters *params);        /* Suggest optimal mu vals */
void GetMuValues(struct Parameters *params, FLOAT *mu1, FLOAT *mu2, FLOAT *mu3, FLOAT *mu4, FLOAT *initmu1, FLOAT *initmu2);                                     /* Compute optimal mu-vals */
void GetMuValuesforFuzzymap(struct Parameters *params, FLOAT *mu1, FLOAT *mu2, FLOAT *mu3, FLOAT *mu4, FLOAT *initmu1, FLOAT *initmu2);                                     /* Compute optimal mu-vals */
void PrintNetworkInfo(struct Parameters *params, FILE *ofile);/*Print netconf*/
void GetUniqueStruct(struct Parameters *params);
char *GetStructure(struct Node *node, struct Node *ndtmp, int fout, int flag);
FLOAT BuildIntegral(struct Parameters *params);

#endif

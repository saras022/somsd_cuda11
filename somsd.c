/*
  Contents: The main module of the som-sd package.

  Author: Markus Hagenbuchner

  Comments and questions concerning this program package may be sent
  to 'markus@artificial-neural.net'

  ChangeLog:
    30/05/2008
      - Consolidation of somsd and initsom into somsd
      - Added support for GraphSOM.
      - Revision of function Usage()
    03/10/2006
      - Port to CYGWIN complete.
      - Be more verbose about bad command line parameters, and attempts to
        access features that are not yet implemented.
      - All command line parameters (other than -cin) now have default values:
      - Compute default neighborhood radius if not specified at command line.
      - Use 64 as default number of training iterations if -iter was not
        specified at the command line.
      - Use 1.0 as default value for the learning rate if -alpha was not
        specified at the command line.
    14/09/2006
      - BugFix: Specification of seed at command line option had no effect.
Can't re-allocate 6004 bytes of memory Rediming data  Out of memory
  ToDo:
    - Since supporting command line option -maptype, this should make the 
      command line option -contextual obsolete. We will leave it in for a while
      for backward compatibility.
    - A nice user manual would be nice.
    - Supervised mode is not yet fully implemented.
 */


/************/
/* Includes */
/************/
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "data.h"
#include "fileio.h"
#include "train_cpu.h"
#include "train.h"
#include "utils.h"
#include "cuda_utils.h"
#include "testsom.h"
#include "train_gpu.h"


#define COMMANDMODE 0x0001

extern char *fileresult;

/* Begin functions... */

/******************************************************************************
Description: Print usage information to the screen

Return value: The function does not return.
******************************************************************************/
void Usage()
{
  fprintf(stderr, "\n\
\033[01;01mNAME\033[00m\n\
       somsd - Graph Self-Organizing Map simulator\n\n\
\033[01;01mSYNOPSIS\033[00m\n\
       somsd [parameters]\n\
\033[01;01mDESCRIPTION\033[00m\n\
       somsd is a simulator for Self-Organizing Maps that are capable of\n\
       mapping graph structured data.\n\n\
\033[01;01mPARAMETERS\033[00m\n\
    -alpha <float>        initial learning rate alpha value\n\
    -alpha_type <type>    Type of alpha decrease. Type can be either:\n\
                          sigmoidal    (default) sigmoidal decrease.\n\
                          linear       linear decrease.\n\
                          exponential  exponential decrease.\n\
                          constant     no decrease. Alpha remains constant.\n\
    -batch                use batch mode training\n\
    -cin <filename>       initial codebook file\n\
    -cout <filename>      the trained map will be saved in filename.\n\
    -din <filename>       The file which holds the training data\n\
	-vin <filename>       The file which holds the validation data (optional)\n\
	-tin <filename>       The file which holds the testing data (optional)\n\
    -exec <command>       Execute the <command> every <snapinterval>.\n\
    -gpu[:<num>]          Enable GPU computing if available. On multi-GPU systems\n\
                          use <num> to set the GPU device number. For example:\n\
	  	          -gpu:0 uses the first GPU\n\
    -iter <int>           The number of training iterations.\n\
    -kfoldtime <kfold> In cross validation setting.\n\
    -log <filename>       Print loging information to <filename>. If <filename>\n\
                          is '-' then print to stdout. At current, only the\n\
                          running quantization error is logged.\n\
    -momentum <float>     use momentum term (implies -batch)\n\
    -mu1 float            Weight for the label component.\n\
    -mu2 float            Weight for the child-state component.\n\
    -mu3 float            Weight for the parent-state component.\n\
    -mu4 float            Weight for the target label component.\n\
    -nice                 Be nice, sleep while system load is high.\n\
    -precision <int>      Some of the implemented optimizations may produce\n\
                          inaccurate results. This option allows to set a\n\
                          precision level to a value between [0;3]. The higher\n\
                          the precision level the fewer optimizations are\n\
                          taken. Warning: A precision level larger than 0 can\n\
                          slow down training times very significantly!\n\
                          Default precision level is 0\n\
    -randomize <entity>   Randomize the order of an entity. Valid entities are:\n\
                          nodes, graphs. By default, the order of graphs is\n\
                          maintained as read from a datafile while nodes are\n\
                          sorted in an inverse topological order. This option\n\
                          allows to change this behaviour.\n\
    -radius <float>       initial radius of neighborhood\n\
    -res <fileresult>.\n\
    -seed <int>           seed for random number generator. 0 is current time\n\
    -snapfile <filename>  snapshot filename\n\
    -snapinterval <int>   interval between snapshots\n\
    -tin <filename>       The file which holds the testing data\n\
    -undirected           Treat all links as undirected links.\n\
    -v                    Be verbose.\n\
    -help                 Print this help.\n\
    \n\
The following parameters are supported if option -cin is not specified:\n\
    -maptype <type>    Type of the map. The type can be either:\n\
                       som         The standard SOM.\n\
                       somsd       The SOMSD.\n\
       		       pmgraphsom  The PM-GraphSOM.\n\
    -neigh <type>      The neighborhood type which can be\n\
                       bubble       Limit neighborhood relationship\n\
                       gaussian     Gaussian bell relationship (default)\n\
    -topol <type>      Specify the topology type of the map. Type can be\n\
                       rectagonal   Neurons are 4-connected\n\
                       hexagonal    Neurons are 6-connected. (the default)\n\
                       octagonal    Neurons are 8-connected.\n\
                       vq           VQ mode (no topology).\n\
    -xdim <xdim>       Horizontal extension of the map.\n\
    -ydim <ydim>       Vertical extension of the map.\n\
 \n");

  exit(0);
}

/******************************************************************************
Description: Convert the value string associated with command line option
             -super into a corresponding type value.

Return value: The value associated with the constant KOHONEN, INHERITANCE, 
              REJECT, LOCALREJECT, or UNKNOWN depending on whether the value
              string was "kohonen", "inheritance", "rejection", "localreject",
              or neither of these.
******************************************************************************/
UNSIGNED GetSuperMode(char *arg)
{
  if (arg == NULL)
    return UNKNOWN;
  else if (!strncasecmp(arg, "kohonen", 3))
    return KOHONEN;
  else if (!strncasecmp(arg, "inheritance", 3))
    return INHERITANCE;
  else if (!strncasecmp(arg, "rejection", 3))
    return REJECT;
  else if (!strncasecmp(arg, "localreject", 3))
    return LOCALREJECT;
  else
    return UNKNOWN;
}

/******************************************************************************
Description: Retrieves recognized command line options and stores given
             options in the Parameters structure.

Return value: A pointer to a dynamically allocated Parameters structure which
              is initialized with values specified in the command line, or null
              for elements not touched by the given command line parameters.
******************************************************************************/
struct Parameters* GetParameters(struct Parameters *parameters, char **argv)
{
  int i;
  fileresult = strdup("/dev/null");

  if (argv == NULL || argv[0] == NULL) /* Sanity check */
    return parameters;
    
  //parameters->fuzzy = 0;
  
  for (i = 1; argv[i] != NULL; i++){
    //Tuc Add fileresult
    if (!strcmp(argv[i], "-res")){
      //i++;
      //fileresult = malloc(100*sizeof(char));
      //sscanf(argv[i], "%s", fileresult);
      free(fileresult);
      GetArg(argv, i++, TYPE_STRING, &fileresult);
    }

    if (!strcmp(argv[i], "-cin"))
      GetArg(argv, i++, TYPE_STRING, &parameters->inetfile);
    else if (!strcmp(argv[i], "-din"))
      GetArg(argv, i++, TYPE_STRING, &parameters->datafile);
    else if (!strcmp(argv[i], "-vin"))
      GetArg(argv, i++, TYPE_STRING, &parameters->validfile);
    else if (!strcmp(argv[i], "-tin"))
      GetArg(argv, i++, TYPE_STRING, &parameters->testfile);
    else if (!strcmp(argv[i], "-cout"))
      GetArg(argv, i++, TYPE_STRING, &parameters->onetfile);
    else if (!strcmp(argv[i], "-gopt"))
      GetMultiArg(argv, i++, "i[:i]", &parameters->map.goptx, &parameters->map.gopty);
    else if (!strcmp(argv[i], "-log"))
      GetArg(argv, i++, TYPE_STRING, &parameters->logfile);
    else if (!strcmp(argv[i], "-iter"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->rlen);
    else if (!strcmp(argv[i], "-alpha"))
      GetArg(argv, i++, TYPE_FLOAT, &parameters->alpha);
    else if (!strcmp(argv[i], "-alpha_type"))
      parameters->alphatype = GetAlphaType(argv[++i]);
    else if (!strcmp(argv[i], "-beta"))
      GetArg(argv, i++, TYPE_FLOAT, &parameters->beta);
    else if (!strcmp(argv[i], "-precision"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->precision);
    else if (!strcmp(argv[i], "-radius"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->radius);
    else if (!strcmp(argv[i], "-seed"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->seed);
    else if (!strcmp(argv[i], "-exec"))
      GetArg(argv, i++, TYPE_STRING, &parameters->snap.command);
    else if (!strcmp(argv[i], "-batch"))
      parameters->batch = 1;
    else if (!strcmp(argv[i], "-cpu"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->ncpu);
    else if (!strcmp(argv[i], "-simple_kernel"))
      parameters->kernel = KERNEL_SIMPLE;
    else if (!strcmp(argv[i], "-kernel"))
      parameters->kernel = KERNEL_FULL;
    else if (!strcmp(argv[i], "-remove%"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->percentremoved);
    else if (!strcmp(argv[i], "-remove"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->removed);
    else if (!strcmp(argv[i], "-momentum"))
      parameters->momentum = 1;
    else if (!strcmp(argv[i], "-mu1"))
      parameters->nmu1 = GetArg(argv, i++, TYPE_FLOAT_VECT, &parameters->mu1);
    else if (!strcmp(argv[i], "-mu2"))
      parameters->nmu2 = GetArg(argv, i++, TYPE_FLOAT_VECT, &parameters->mu2);
    else if (!strcmp(argv[i], "-mu3"))
      parameters->nmu3 = GetArg(argv, i++, TYPE_FLOAT_VECT, &parameters->mu3);
    else if (!strcmp(argv[i], "-mu4"))
      parameters->nmu4 = GetArg(argv, i++, TYPE_FLOAT_VECT, &parameters->mu4);
    else if (!strcmp(argv[i], "-kfoldtime")){
    	i++;
    	sscanf(argv[i], "%d", &parameters->kfoldtime) ;

      //parameters->kfoldtime = GetArg(argv, i++, TYPE_UNSIGNED, &parameters->kfoldtime);
    printf("kfoldtime %d\n", parameters->kfoldtime);}
    else if (!strncmp(argv[i], "-gpu", 4)){
      if (GetAttribVal(argv[i], 4, TYPE_INT, &parameters->device) >= 0)
	parameters->mode |= GPU_MODE;
    }
    else if (!strcmp(argv[i], "-nice"))
      parameters->nice = 1;
    else if (!strncmp(argv[i], "-randomize", 7)){
      if (!(parameters->nodeorder = (strncmp(argv[++i], "node", 4) ? 0 : 1)) &&
	  !(parameters->graphorder = (strncmp(argv[i], "graph", 5) ? 0 : 1))){
	fprintf(stderr, "Warning: Ignoring unrecognized value '%s' for option -randomize.\n", argv[i]);
      }
    }
    else if (!strcmp(argv[i], "-snapfile"))
      GetArg(argv, i++, TYPE_STRING, &parameters->snap.file);
    else if (!strcmp(argv[i], "-snapinterval"))
      GetArg(argv, i++, TYPE_UNSIGNED, &parameters->snap.interval);
    else if (!strcmp(argv[i], "-super"))
      parameters->super = GetSuperMode(argv[++i]);
    else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "-verbose"))
      parameters->verbose = 1;
    else if (!strcmp(argv[i], "-undirected")){
      parameters->undirected = 1;
      parameters->contextual = 1;
    }
    else if (!strcmp(argv[i], "-neigh") || !strcmp(argv[i], "-topol") || !strcmp(argv[i], "-xdim") || !strcmp(argv[i], "-ydim") || !strcmp(argv[i], "-linear")|| !strcmp(argv[i], "-maptype"))
      i++;  /* ignore these options for now (see function InitMap() )    */
    else if (!strcmp(argv[i], "-help") || !strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")){
      Usage();
    }
    else if (*argv[i] == '-'){
      fprintf(stderr, "Warning: Ignoring unrecognized command line option '%s'\n", argv[i]);
    }

    if (CheckErrors() != 0)
      break;
  }

  return parameters;
}

/******************************************************************************
Description: Check given command line parameters for consistency, set default
          values if necessary, and print warnings if there unusual parameters
          are used.

Return value: The function does not return a value. 
******************************************************************************/
void CheckParameters(struct Parameters* parameters)
{
  char msg[256];

  fprintf(stderr, "Checking parameters");
  if (parameters->map.codes == NULL)
    AddError("No Map, or Map is empty.");
  else if (parameters->map.type == TYPE_UNKNOWN) /* Set default map type */
    parameters->map.type = TYPE_SOMSD;

  if (parameters->train == NULL){    /* No training set */
    if (parameters->datafile == NULL) /* May this was missed at command line */
      AddError("Parameter -din is missing. No training file given.");
    else            /* File did not exist or could not be read successfully */
      AddError("Training data failed to load successfully.");
     }
  
  if (parameters->onetfile == NULL){
    parameters->onetfile = strdup("trained.net");  // Set default output file
    AddMessage(PRT_WARNING "Will save trained network to file 'trained.net'.");
    AddMessage("         Use option -cout to alter this behaviour.");
  }

  if (parameters->rlen == 0){
    AddMessage(PRT_WARNING "Number of training iterations not specified or zero.");
    AddMessage("         Training iterations defaults to: 64");
    AddMessage("         Use option -iter to alter this behaviour.");
    parameters->rlen  = 64;
  }

  if (parameters->alpha == 0.0)
    AddMessage(PRT_WARNING "Learning rate is zero.");

  if (parameters->alpha < 0.0)
    AddMessage(PRT_WARNING "Learning rate is negative!");

  if (parameters->alpha > 2.0){
    AddMessage(PRT_WARNING "Learning rate is likely to be too large.");
    AddMessage("         Suggested values for learning rate are within [0;1]");
  }

  if (parameters->precision > 3){
    AddMessage(PRT_WARNING "Precision level larger than 3 has no effect.");
    AddMessage("         Resetting precision level to 3.");
    parameters->precision = 3;
  }
  if (parameters->precision > 0){
    AddMessage(PRT_WARNING "Non-zero value given for 'precision level'.");
    AddMessage("         This may result in significantly increased training times.");
  }

  if (parameters->alphatype == UNKNOWN)
    parameters->alphatype = ALPHA_SIGMOIDAL; /* Set default alpha type */

  if ((parameters->super == REJECT ||  parameters->super == LOCALREJECT)){
    if (parameters->beta == 0.0)
      AddMessage(PRT_WARNING "Rejection rate is zero in supervised mode.");
  }
  else if (parameters->beta != 0.0){
    parameters->beta = 0.0;
    AddMessage("Note: Not in corresponding supervised mode. Will ignore rejection rate.");
  }

  /* Check if a useful neigborhood radius is specified. In some cases a
     radius can be zero or negative. But in most cases it cannot. */
  if (parameters->radius == 0 && parameters->map.topology != TOPOL_VQ){
    AddMessage(PRT_WARNING "Neigborhood radius not specified or zero!");
    parameters->radius  = 1.0+sqrt(parameters->map.xdim * parameters->map.xdim + parameters->map.ydim * parameters->map.ydim)/9.0;
    sprintf(msg, "         Neigborhood radius defaults to: %d", parameters->radius);
    AddMessage(msg);
    AddMessage("         This is most likely not what you want.");
    AddMessage("         Use option -radius to alter this behaviour.");
  }
  /* Check seed values and initialize random number generator if needed */
  if (parameters->seed > 0){
    srand48(parameters->seed);
  }

  if (parameters->batch == 0 && parameters->momentum != 0){
    parameters->batch = 1;   /* momentum term requires batch mode processing */
  }

  if (parameters->batch != 0){
    AddMessage(PRT_WARNING "Batch mode processing not yet implemented!");
    AddMessage("         Will proceed in default online mode.");
  }

  if (parameters->super != 0 && parameters->super != REJECT){
    AddMessage(PRT_WARNING "Supervised processing not yet implemented!");
    AddMessage("         Will proceed in default unsupervised mode.");
  }

  if (parameters->kernel != 0){
    AddMessage(PRT_WARNING "Kernel mode processing not yet implemented!");
    AddMessage("         Will proceed in default mode.");
  }

  if (parameters->seed == 0)
    parameters->seed = (UNSIGNED)clock();
  //if (parameters->ncpu == 0)
  //  parameters->ncpu = GetNumCPU();

  if (parameters->logfile == NULL || strlen(parameters->logfile) == 0)
    parameters->logfile = strdup("somsd.log");

  if (parameters->snap.interval > 0 && parameters->snap.file == NULL && parameters->snap.command == NULL){
    parameters->snap.file = strdup("snapshot.net");
    AddMessage(PRT_NOTE "Will save a snapshots to file 'snapshot.net'");
  }
  else if (parameters->snap.interval == 0 && parameters->snap.file != NULL){
    parameters->snap.interval = 1;
    AddMessage(PRT_NOTE "Will save a snapshots at every iteration");
  }

  SuggestMu(parameters);/* Compute optimal weight parameters */

  /* Print error and warning messages if there are any */
  if (CheckErrors()){
    fprintf(stderr, "\033[01;31m%55s\033[00m\n", "[FAILED]");
    return;
  }
  else{
    if (CheckMessages()){
      fprintf(stderr, "\033[01;34m%55s\033[00m\n", "[CAUTION]");
      PrintMessages();
      fprintf(stderr, "\n");
    }
    else
      fprintf(stderr, "\033[01;32m%55s\033[00m\n", "[OK]");
  }
}

/****************************************************************************
 Function: 

 Parameter:

 Return: NULL
****************************************************************************/
void *ProcessUserCommands(void *arg)
{
  int i,c;
  int flag = 0;
  struct Parameters *param = (struct Parameters *)arg;
  time_t start_time;

  start_time = time(NULL);
  while(1){
    if (time(NULL) - start_time > 7200){//Backup network once every 2 hours
      param->savenet |= SAVE_BACKUP;
      start_time = time(NULL);
    }
    c = getch();
    if(c == '#'){
      flag ^= COMMANDMODE;
      if (flag & COMMANDMODE)
	printf("\rCommand mode enabled\n");
      else
	printf("\rCommand mode disabled\n");
    }
    else if (c == 'X') //quit now!
      exit(0);
    else if (c == 'h' || c == '?'){ //Print help
      fprintf(stderr, "\r \nAvailable commands are:\n");
      fprintf(stderr, "\tI\tPrint current state.\n");
      fprintf(stderr, "\tX\tQuit right now!\n");

      if (flag & COMMANDMODE)
	fprintf(stderr, "\t#\tExit the command mode\n");
      else{
	fprintf(stderr, "\t#\tEnable the advanced command mode\n");
	fprintf(stderr, "When in advanced command mode:\n");
      }
      fprintf(stderr, "\ta<num>\tSet 'alpha' to <num>\n");
#ifdef __USECUDA__
      fprintf(stderr, "\tg\tToggle GPU mode.\n");
#endif
      fprintf(stderr, "\ti<num>\tSet rlen to <num>\n");
      fprintf(stderr, "\tp\tToggle Pause.\n");
      fprintf(stderr, "\ts\tSave network after curent iteration.\n");
      fprintf(stderr, "\tS<name>\tSave network to <file>.\n");
      fprintf(stderr, "\tx\tQuit after current iteration\n");
      fprintf(stderr, "\n");
    }
    else if (c == 'I'){ //Print value of parameters
      fprintf(stderr, "\r \nCodebase in version %s\n", PROG_VERSION);
      fprintf(stderr, "Startup command was:\n");
      for (i = 0; i < param->argc; i++)
	fprintf(stderr, " %s", param->argv[i]);
      fprintf(stderr, "\nCurrent:\n");
      if (param->alpha > 0.0001f)
	fprintf(stderr, "Learning rate  : %f", param->alpha);
      else
	fprintf(stderr, "Learning rate  : %e", param->alpha);
      fprintf(stderr, "\n");
      fprintf(stderr, "Training mode  : ");
      if (param->map.type == TYPE_GRAPHSOM) {
	if (param->fuzzy && param->memorysave)
	  fprintf(stderr, "PMGraphSOM\n");
	else
	  fprintf(stderr, "Check command line parameters!\n");
      }else if (param->map.type == TYPE_SOMSD)
	fprintf(stderr, "SOMSD\n");
      else
	fprintf(stderr, "SOM\n");
      fprintf(stderr, "\n");
    }
    else if (c > 0 && !(flag & COMMANDMODE)){
      printf("\bPress 'h' for help.\n");
      sleep(1);
    }
    else if (c == 'a'){ //set learning rate
      fprintf(stderr, "\nSet 'alpha' to: ");
      scanf("%e", &param->alpha);
      fprintf(stderr, "\nAdjusted to: alpha=%e \n", param->alpha);
    }
#ifdef __USECUDA__
    else if (c == 'g'){ //toggle gpu mode
      if (param->mode & GPU_MODE){
	param->mode &= (~GPU_MODE);
	fprintf(stderr, "\nWill switch to CPU mode at end of current iteration...\n");
      }
      else{
	param->mode |= GPU_MODE;
	fprintf(stderr, "\nWill switch to GPU mode at end of current iteration...\n");
      }
    }
#endif
    else if (c == 'i'){ //set rlen (maximum number of training iterations)
      fprintf(stderr, "\b\nSet 'rlen' to (currently %d): ", param->rlen);
      scanf("%d", &param->rlen);
      fprintf(stderr, "\nAdjusted to: rlen=%d \n", param->rlen);
    }
    else if (c == 'p'){ //toggle pause
      if (param->flags & PAUSE){
	param->flags &= ~PAUSE;
	fprintf(stderr, "\rContinuing training.\n");
      }
      else{
	param->flags |= PAUSE;
	fprintf(stderr, "\rWill pause at end of current iteration.\n");
      }
    }
    else if (c == 'S'){ //Set network output file name
      fprintf(stderr, "\b\nWill save network at end of current iteration...\n");
      fprintf(stderr, "Warning: SaveNet function not yet activated!\n");
    }
    else if (c == 'S'){ //Set network output file name
      fprintf(stderr, "\b\nSave network to file: ");
      if (param->onetfile)
	free(param->onetfile);
      param->onetfile = (char*)calloc(256, sizeof(char));
      scanf("%255s", param->onetfile);
      param->savenet = 1;
      fprintf(stderr, "Warning: SaveNet function not yet activated!\n");
    }
    else if (c == 'x'){ //quit
      param->rlen = 0;
      fprintf(stderr, "\nQuitting....wait for current iteration to complete\n");
    }
    else{
      sleep(1);
      continue;
    }
  }
}


/******************************************************************************
Description: Create a newly initialized map

Return value: 
******************************************************************************/
void InitMap(struct Parameters *params, int argc, char **argv)
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
    else if (!strcmp(argv[i], "-help") || !strcmp(argv[i], "-h") || !strcmp(argv[i], "-?")){
      Usage();
    }

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
      AddMessage(PRT_NOTE "Neighborhood defaulted to 'gaussian'");
      map->neighborhood = NEIGH_GAUSSIAN;
    }
    if (map->topology == UNKNOWN){
      AddMessage(PRT_NOTE "Topology defaulted to 'hexagonal'");
      map->topology = TOPOL_HEXA;
    }
    if (map->xdim * map->ydim == 0)
      AddError("Network dimension not specified, or is zero.");
  }

  if (params->train == NULL)
    AddError("Can't initialize map without training data.");

  if (CheckErrors() == 0){     /* If there were no errors so far then      */
    fprintf(stderr, "Initializing network...."); /* Print what's happening */
    srand48(params->seed);     /* Initialize random number generator       */
    
    if (map->type == TYPE_GRAPHSOM){ /*     Allocate and initialize a map   */
      if (params->compact)
	InitGsomCodesCompact(map, params->train, mode);/* Init for compact GRAPHSOM*/
      else
	InitGsomCodes(map, params->train, mode); /* Special Init for GRAPHSOM*/
    }
    else
      InitCodes(map, params->train, mode);     /* Init any other SOM       */

    if (CheckErrors() == 0)
      fprintf(stderr, "\033[01;32m%50s\033[00m\n", "[OK]");    /* print confirmation          */
    else
      fprintf(stderr, "\033[01;31m%50s\033[00m\n", "[FAILED]");/*Network initialization failed*/
  }

  return;
}

/****************************************************************************
 Function: 

 Parameter:

 Result: 
****************************************************************************/
pthread_t SystemInit(struct Parameters *param, int argc, char *argv[])
{
  pthread_t tid = 0;
  //  int i;
  //  time_t t;
  //  char *cwd, *hname;

  param->argc = argc;
  param->argv = argv;
#ifdef __INTERACTIVE__
  pthread_create(&tid, NULL, ProcessUserCommands, param);
  InstallHandlers();
#endif

#ifdef __USECUDA__
 if (param->mode & GPU_MODE){
    param->device = findCudaDevice(param->device);
    if (param->verbose)
      PrintCUDAInformation(stderr);
  }
#endif

  /*
  if (param->logfile.fname){
    if (param->logfile.flags & F_APPEND)
      param->logfile.ofile = fopen(param->logfile.fname, "a");
    else
      param->logfile.ofile = fopen(param->logfile.fname, "w");
    if (param->logfile.ofile == NULL){
      fprintf(stderr, "WARNING: Unable to write to logfile `%s`\n", param->logfile.fname);
      fprintf(stderr, "         Check write permissions & disc status.\n");
    }
  }

  if (param->logfile.ofile){//Write startup condition into lofile
    cwd = getcwd(NULL, 0);  //Get current working directory
    hname = GetHostName();  //Get hostname
    if (hname == NULL || !strncmp(hname, "localhost", 9)) //No hostname?
      hname = GetIPaddress();      //Get IP address
    if (hname == NULL)             //Not even an IP?
      hname = strdup("localhost"); 
    
    fprintf(param->logfile.ofile, "# Startup:");
    for (i = 0; i < param->argc; i++)
      fprintf(param->logfile.ofile, " %s", param->argv[i]);
    fprintf(param->logfile.ofile, "\n");
    t = time(NULL);
    fprintf(param->logfile.ofile, "# Date: %s", ctime(&t));
    fprintf(param->logfile.ofile, "# Location: %s:%s\n", hname, cwd);
    fflush(param->logfile.ofile);
    if (cwd) free(cwd);
    free(hname);
  }
  */
  return tid;
}

/******************************************************************************
Description: main

Return value: zero
******************************************************************************/
int main(int argc, char **argv)
{
  struct Parameters parameters;
  time_t starttime;

  starttime = time(NULL);
  memset(&parameters, 0, sizeof(struct Parameters));
  parameters.alpha = 1.0; /* Default learning rate  */
  parameters.device = -1; /* Default: no GPU device */

  GetParameters(&parameters, argv);

  //if (parameters.verbose){
  //  PrintSoftwareInfo(stderr); /* Print a nice header with software and */
  //  PrintSystemInfo(stderr);   /* hardware information */
  //}

  if (CheckErrors() == 0 && parameters.inetfile != NULL){
    LoadMap(&parameters);                /* Load the map */
    if (CheckErrors() == 0)              /* If load was successful then */
      GetParameters(&parameters, argv);  /* Overwrite network parameters*/
  }

  SystemInit(&parameters, argc, argv);

  if (CheckErrors() == 0)
    parameters.train = LoadData(parameters.datafile, &parameters); /* Load training data  */

  if (CheckErrors() == 0 && parameters.validfile != NULL)
    parameters.valid = LoadData(parameters.validfile, &parameters);/*Load validation data */

  if (CheckErrors() == 0 && parameters.testfile != NULL)
      parameters.test = LoadData(parameters.testfile, &parameters);/*Load testing data */
  
  if (CheckErrors() == 0 && parameters.inetfile == NULL)
    InitMap(&parameters, argc, argv);    /* Create a new map if none loaded  */

  if (CheckErrors() == 0 && parameters.map.type == TYPE_GRAPHSOM)/* re-scale */
    Data2Gsom(&parameters);

  CheckParameters(&parameters);          /* Check for parameter consistancy  */

  if (CheckErrors() == 0 && parameters.undirected)      /* treat undirected  */
    ConvertToUndirectedLinks(parameters.train);

  if (CheckErrors() == 0)
    PrepareData(&parameters);/* Prepare data for training phase */

  if (parameters.verbose)  /* Print final network configuration */
    PrintNetworkInfo(&parameters, stderr);

  fprintf(stderr, "Time Used so far: %s\n", PrintTime(time(NULL) - starttime));
      
  starttime = time(NULL);

#ifndef __USECUDA__
  parameters.mode &= ~GPU_MODE;//Ensure that gpu mode is disabled if Cuda unavailable
#endif

  if (CheckErrors() == 0){
    //PrintNetworkInfo(&parameters, stdout);
	//for (parameters.kfoldtime=0;parameters.kfoldtime<11;parameters.kfoldtime++)//11 participants 0f PA project data
	//for (parameters.kfoldtime=0;parameters.kfoldtime<7;parameters.kfoldtime++)//7 participants 0f HASC data
	//for (parameters.kfoldtime=0;parameters.kfoldtime<3;parameters.kfoldtime++)//7 participants 0f HASC and TROST data


//This is temprory commented out for nsight
#ifdef __USECUDA__
    if (parameters.mode & GPU_MODE)
      TrainMap_GPU(&parameters);
    else
#endif
     TrainMap_CPU(&parameters);   /* Train the network               */


	//fprintf(stderr, "Time Used for training: %s\n", PrintTime(time(NULL) - starttime));
	 
    //if(!strcmp(parameters.datafile,"xmllabelgraph.gph"))
    //if(!strcmp(parameters.train->gname, "INEX")){
      //MapNode(parameters.map, parameters.train);
      //fprintf(stdout, "---\n");
      //ClassificationINEX2008(parameters, 0);
      //fprintf(stdout, "---\n");       
     //ClusterINEX2008(parameters, 0);
    //}
    //else{
      //ComputeRetrievalPerformance(parameters, 0);
      //ComputeMutagPerformance(parameters, 0);
    //}
      /*if(parameters.test!=NULL)
	MapGraphTest(parameters, -1, -1);
	else
	MapGraph(parameters, -1, -1);*/
    //}
    //fprintf(stderr, "Time Used so far: %s\n", PrintTime(time(NULL) - starttime));
	   	  
    //SaveMapAscII(&parameters);    /* Save the trained map */
  }

  //fprintf(stderr, "Time Used so far: %s\n", PrintTime(time(NULL) - starttime));

  Cleanup(&parameters);       /* Free allocated memory and flush errors */

  return 0;
}

/* End of file */

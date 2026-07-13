/*
  Contents: Functions that may be of general use.

  Author: Markus Hagenbuchner

  Comments and questions concerning this program package may be sent
  to 'markus@artificial-neural.net'

  Changelog:
    02/06/2008:
      - Added function GetMultiArg()
      - Some minor bug fixes and better documentation of the code.
    10/10/2006:
      - Corrected some source code comments.
      - Bug fix: SlideIn(): now prints correctly if not at beginning of a line

 */


/************/
/* Includes */
/************/
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include "utils.h"

#ifdef DEBUG
unsigned long int memusage = 0L;
#endif

/* Begin functions... */


/* Memory management utilities */
/*****************************************************************************
Description: A fail save version of malloc. MyMalloc allocates size bytes and
             returns a pointer to the allocated memory. The memory is not
             cleared. The function aborts immediately with an error message if
             the memory could not be allocated.

Return value: A pointer to the allocated memory.

Comment: Linux and some other operating systems do not immediately allocate the
         physical memory unless it is written to. Thus, malloc may not fail
         in the event that unsufficient memory is available, but any accesses
         to the memory at a later stage may fail. As a general rule of thumb,
         it is best to use calloc instead.
*****************************************************************************/
void *MyMalloc(size_t size)
{
  void *ptr;
  if ((ptr = malloc(size)) == NULL){
    fprintf(stderr, "\nOut of memory while trying to allocate %d bytes\n", (int)size);
    exit(0);
  }
#ifdef DEBUG
  memusage += size;
  fprintf(stderr, "M:%lu\n", memusage);
#endif


  return ptr;
}

/*****************************************************************************
Description: A fail save version of calloc. MyCalloc allocates memory for an
             array of nmemb elements of size and returns a pointer to the
             allocated memory. The memory is set to zero. The function aborts
             immediately with an error message if the memory could not be
             allocated.

Return value: A pointer to the allocated memory.
*****************************************************************************/
void *MyCalloc(size_t nmemb, size_t size)
{
  void *ptr;

  if ((ptr = calloc(nmemb, size)) == NULL){
    fprintf(stderr, "\nError: Out of memory while trying to allocate %ld bytes\n", (long)size*(long)nmemb);
    exit(0);
  }
#ifdef DEBUG
  memusage += nmemb*size;
  fprintf(stderr, "M:%lu\n", memusage);
#endif
  return ptr;
}

/*****************************************************************************
Description: A fail save version of realloc.

Return value: A pointer to the allocated memory.
*****************************************************************************/
void *MyRealloc(void *ptr, size_t size)
{
  void *newarray;

  if ((newarray = realloc(ptr, size)) == NULL){
    //fprintf(stderr, "Can't re-allocate %ld bytes of memory.\n", (long)size);
    //return NULL;
  }
#ifdef DEBUG
  memusage += size;
  fprintf(stderr, "M:%lu\n", memusage);
#endif
  return newarray;
}

/*****************************************************************************
Description: Duplicate a memory area. This is similar to strndup but works on
             binary content.

Return value: A pointer to the duplicated memory area.
*****************************************************************************/
void *memdup(void *ptr, size_t size)
{
  void *dest;

  dest = MyMalloc(size);
  return memcpy(dest, ptr, size);
}


/* String functions */

/*****************************************************************************
Description: Concatenates the two strings str1 and str2. The result is stored
             in a new array for which sufficient memory is allocated.

Return value: A pointer to an array which holds the concatenated strings.
*****************************************************************************/
char *stradd(char *str1, char *str2)
{
  char *buffer = NULL;

  if (str1 == NULL){
    if (str2 == NULL)
      return NULL;         /* Return NULL if both strings are NULL */
    else
      return strdup(str2); /* Return a copy of str2 if str1 is NULL */
  }
  if (str2 == NULL)
    return strdup(str1);   /* Return a copy of str1 if str2 is NULL */

  /* allocate memory for concatenated string */
  buffer = (char*) MyMalloc(strlen(str1) + strlen(str2) + 1);
  strcpy(buffer, str1);  /* Copy str1 into the buffer */
  strcat(buffer, str2);  /* then concatenate str2 to the buffer */
  return buffer;         /* Return a pointer to the buffer */
}

/*****************************************************************************
Description: Return a pointer pointing to the end of str. If str is an empty
             string or if str is a NULL pointer, then the returned pointer is 
             the same as str. Otherwise, the pointer to the first occurence of
             the end_of_string marker ('\0') is returned.

Return value: A pointer to the end of str.
*****************************************************************************/
char *strend(char *str)
{
  if (str == NULL)
    return NULL;
  else
    return &str[strlen(str)];
}

/*****************************************************************************
Description: Return pointer to first occurence of a character in str that is
             not a white space.

Return value: A pointer to the first occurence of a character in str that is
              not a white space, or '\0' if str contains only white spaces.
*****************************************************************************/
char *strnspc(char *str)
{
  if (str == NULL)
    return NULL;

  while (*str != '\0' && isspace((int)*str))
    str++;

  return str;
}

/****************************************************************************
Description: The strnstr() function finds the first occurrence of characters 
             not found in substring needle in the string haystack. The
             terminating '\0' characters are not compared.

Return value: The strnstr() function returns a pointer to the beginning of
              the substring, or NULL if the substring is not found.
****************************************************************************/
char *strnstr(char *haystack, const char *needle)
{
  const char *cptr;

  if (haystack == NULL)/* Nothing to do and nothing to find, so return NULL */
    return NULL;

  if (needle == NULL)  /* No needle, so all characters in haystack match */
    return haystack;

  for (;*haystack != '\0'; haystack++){
    for (cptr = needle; *cptr != '\0'; cptr++){
      if (*haystack == *cptr)
	break;
    }
    if (*cptr == '\0')  /* No character in needle matched */
      return haystack;
  }
  return NULL;          /* all of haystack consists of characters in needle */
}

/*****************************************************************************
Description: Same as atoi but returns a default value if the initial portion
             of the string pointed to by cptr does not hold an integer value.
              
Return value: The converted value.
*****************************************************************************/
int oatoi(char *cptr, int idefault)
{
  if (cptr == NULL || !(isdigit((int)*cptr) || *cptr == '+' || *cptr == '-'))
    return idefault;
  else
    return atoi(cptr);
}

/*****************************************************************************
Description: Same as atoi but returns a unsigned falue or a default value if
             the initial portion of the string pointed to by cptr does not
             hold an unsigned value.
              
Return value: The converted value.
*****************************************************************************/
unsigned oatou(char *cptr, int udefault)
{
  if (cptr == NULL || !(isdigit((int)*cptr) || *cptr == '+'))
    return udefault;
  else
    return (unsigned)atoi(cptr);
}

/*****************************************************************************
Description: Same as atof but returns a default value if the initial portion
             of the string pointed to by cptr does not hold a float value.

Return value: The converted value.
*****************************************************************************/
float oatof(char *cptr, float fdefault)
{
  if (cptr == NULL || !(isdigit((int)*cptr) || *cptr == '+' || *cptr == '-' || *cptr == '.'))
    return fdefault;
  else
    return (float)atof(cptr);
}

/*****************************************************************************
Description: Same as atoi but stores the integer in ival. ival remains
             unchnaged if cptr does not start with integer information. This
             is a fail-safe function.
              
Return value: 1 if conversion succussful, 0 otherwise
*****************************************************************************/
int satoi(char *cptr, int *ival)
{
  /* Ensure that cptr can be converted to a signed integer */
  if (cptr == NULL || *cptr == '\0' || !(isdigit((int)*cptr) || ((*cptr == '+' || *cptr == '-') && (cptr[1] == '\0' || isdigit((int)cptr[1])))))
    return 0;

  *ival = atoi(cptr);
  return 1;
}

/*****************************************************************************
Description: Converts the initial portion of the string pointed to by cptr to
             unsigned and stores the value in uval. uval remains unchanged
             if cptr does not start with unsigned information. This is a fail-
             safe function.
              
Return value: 1 if conversion succussful, 0 otherwise
*****************************************************************************/
int satou(char *cptr, unsigned *uval)
{
  /* Ensure that cptr can be converted to an unsigned (positive) integer */
  if (cptr == NULL || *cptr == '\0' || !(isdigit((int)*cptr) || (*cptr == '+' && (cptr[1] == '\0' || isdigit((int)cptr[1])))))
    return 0;

  *uval = (unsigned)atoi(cptr);
  return 1;
}

/*****************************************************************************
Description: Same as atof but stores the float in fval. fval remains
             unchanged if cptr does not start with float information.

Return value: 1 if conversion succussful, 0 otherwise
*****************************************************************************/
int satof(char *cptr, float *fval)
{
  /* Ensure that cptr can be converted to a float value */
  if (cptr == NULL || *cptr == '\0' || !(isdigit((int)*cptr) || ((*cptr == '+' || *cptr == '-' || *cptr == '.') && (cptr[1] == '\0' || isdigit((int)cptr[1])))))
    return 0;

  *fval = (float)atof(cptr);
  return 1;
}

/*****************************************************************************
Description: Same as atof but stores a string in sptr. sptr remains
             unchanged if cptr is NULL or an empty string.

Return value: 1 if conversion succussful, 0 otherwise
*****************************************************************************/
int satos(char *cptr, char **sptr)
{
  if (cptr == NULL || *cptr == '\0')
    return 0;

  *sptr = strdup(cptr);
  return 1;
}

/*****************************************************************************
Description: Converts a string pointed to by cptr to float. cptr may point to
             a sequence of float values in the format of 
              float[{,:}float]*
             For example: cptr may point to a string "1.2,1.7,1.3"
             The result is stored in dynamically allocated memory holding the
             float values in the order of apperance. Note that missing values
             receive a default value of 0 (i.e. such as in "1.2,,1.3"). This
             is a fail-safe function.

Return value: Number of floats converted, 0 otherwise
*****************************************************************************/
int satofvector(char *cptr, FLOAT **sptr)
{
  int c;
  FLOAT *f = NULL;

  if (cptr == NULL)
    return 0;

  c = 0;
  while(*cptr != '\0'){
    if (*cptr == ','){//missing value -> insert zero
      c++;
      f = (float*) realloc(f, c * sizeof(FLOAT));
      f[c-1] = 0;
      cptr++;
    }/* Ensure that cptr can be converted to a float value */
    else if (!(isdigit((int)*cptr) || ((*cptr == '+' || *cptr == '-' || *cptr == '.') && (cptr[1] == '\0' || isdigit((int)cptr[1])))))
      break;
    else{
      c++;
      f = (float*) realloc(f, c * sizeof(FLOAT));
      f[c-1] = (FLOAT)atof(cptr);
      while(*cptr != '\0' && (isdigit((int)*cptr) || *cptr == '+' || *cptr == '-' || *cptr == '.')){
	cptr++;
      }
      if (*cptr != '\0')
	cptr++;
    }
  }
  *sptr = f;
  return c;
}


/*****************************************************************************
Description: Check whether a given string str starts with a substring sub. This
             is the same as strncmp(sub, str, strlen(sub)).

Return value: length of sub if str starts with sub, or -1 else
*****************************************************************************/
int strstart(char *sub, char *str)
{
  int len;

  len = strlen(sub);
  if (len == 0 || !strncmp(sub, str, len))
    return len;
  else
    return -1;
}


/* Print functions */
/*****************************************************************************
Description: Write text to ostream and flush the output immediately.

Return value: fprint returns the number of characters printed.
*****************************************************************************/
int fprint(FILE *ostream, const char *text)
{
  int res;

  res = fprintf(ostream, text);
  fflush(ostream);
  return res;
}

/******************************************************************************
 Description: Non-blocking function to get a single char from the terminal
 
 Return value: 
******************************************************************************/
int getch(void)
{
  int ch;
  struct termios oldt;
  struct termios newt;

  tcgetattr(STDIN_FILENO, &oldt); /*store old settings */
  newt = oldt; /* copy old settings to new settings */
  newt.c_lflag &= ~(ICANON | ECHO); /* make changes to settings */
  newt.c_cc[VMIN] = 0;
  newt.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &newt); /*apply the new settings */
  ch = getchar(); /* standard getchar call */
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt); /*reapply the old settings */
  return ch; /*return received char */
}
 
/*****************************************************************************
Description: Prints a message by sliding the message in from the left at a
             given speed

Return value: This function does not return a value.
*****************************************************************************/
void SlideIn(FILE *ostream, int speed, const char *msg)
{
  int i, j, len;

  /* Sanity check */
  if (msg == NULL)
    return;

  len = strlen(msg);
  for (i = 1; i <= len; i++){
    fprintf(ostream, "%s", &msg[len-i]);
    usleep(speed * 10000);
    for (j = 0; j < i; j++)
      fputc('\b', ostream);
  }
}

/*****************************************************************************
Description: High precision print of a float f to an output stream ofile.

Return value: fprint does not return a value.
*****************************************************************************/
void PrintFloat(FILE *ofile, float f)
{
  if (f < 0.0){           /* If the value is negative */
    fputc('-', ofile);    /* print the sign, and      */
    f = -f;               /* make it a positive value */
  }
  fprintf(ofile, "%d.", (int)f);/* Print the integer component of the value */
  do{
    f = f-((int)f);       /* Substract the integer value from the value */
    f = f*10.0;           /* Obtain the next decimal value              */
    fprintf(ofile, "%d", (int)f); /* and print it.                      */
  }while(f>0.0);          /* while the value has decimals to print      */
}

/*****************************************************************************
Description: Convert a given number of seconds into minutes, hours, days into
             a static buffer.

Return value: Pointer to the buffer containing the time as a string.
*****************************************************************************/
char *PrintTime(time_t time)
{
  static char buffer[32];

  buffer[0] = '\0';
  if (time/86400 > 0){
    sprintf(buffer, "%2ddays", (int)time/86400);/* number of days  */
    time = time % 86400;
  }
  if (time / 3600 > 0){
    sprintf(strend(buffer), " %2dhrs", (int)time/3600); /* number of hours */
    time = time % 3600;
  }
  if (time / 60 > 0){
    sprintf(strend(buffer), " %2dmins", (int)time/60);  /* number of minutes */
    time = time % 60;
  }
  if (time > 0)
    sprintf(strend(buffer), " %2dsecs", (int)time);     /* number of seconds */

  if (time == 0L)
    sprintf(buffer, " 0secs");          /* Treat 0 seconds as a special case */

  return buffer;
}


/* Functions that help to deal with command line options */
/*****************************************************************************
Description: Retrieve the value of a command line parameter. Some error
             checking is performed to ensure the type of the value is correct.

Return value: Number of successfully assigned values.
*****************************************************************************/
int GetArg(char **argv, int idx, int type, void *dest)
{
  int i, c = 0;
  char cptr[4096];
  FLOAT *fptr;

  if (argv[idx+1] == NULL || strlen(argv[idx+1]) == 0){
    snprintf(cptr, 4096, "Missing value for parameter '%s'.", argv[idx]);
    AddError(cptr);
    return 0;
  }

  /* check: "-" is valid but "-<alpha>" is not */
  if (strlen(argv[idx+1])>1 && *argv[idx+1] == '-' && isalpha((int)argv[idx+1][1])){
    snprintf(cptr, 4096, "Missing value for parameter '%s'.", argv[idx]);
    AddError(cptr);
    return 0;
  }

  cptr[0] = '\0';
  switch (type){
  case TYPE_STRING:
    /* check: "-" is valid but "-<string>" is not */
    if (*argv[idx+1] == '-' && strcmp(argv[idx+1], "-")){
      snprintf(cptr, 4096, "Missing value for parameter '%s'.", argv[idx]);
      *(char**)dest = NULL;
    }
    else{
      *(char**)dest = strdup(argv[idx+1]);
      c = 1;
    }
    break;
  case TYPE_INT:
    if (*argv[idx+1] == '-' || isdigit((int)*argv[idx+1])){
      *(int*)dest = atoi(argv[idx+1]);
      c = 1;
    }
    else{
      snprintf(cptr, 4096, "Invalid value for parameter '%s'.", argv[idx]);
      *(int*)dest = (int)0;
    }
    break;
  case TYPE_UNSIGNED:
    if (isdigit((int)*argv[idx+1])){
      *(unsigned*)dest = atoi(argv[idx+1]);
      c = 1;
    }
    else{
      snprintf(cptr, 4096, "Invalid value for parameter '%s'.", argv[idx]);
      *(unsigned*)dest = 0u;
    }
    break;
  case TYPE_FLOAT:
    if (*argv[idx+1] == '-' || isdigit((int)*argv[idx+1])|| *argv[idx+1] == '.'){
      *(float*)dest = atof(argv[idx+1]);
      c = 1;
    }
    else{
      snprintf(cptr, 4096, "Invalid value for parameter '%s'.", argv[idx]);
      *(float*)dest = (float)0.0;
    }
    break;
  case TYPE_FLOAT_VECT:  //Float vector in the form of float[{,:}float]*
    //count number of commas, an colons in argument string
    for (i = 0, c = 1; i < (int)strlen(argv[idx+1]); i++){
      if (argv[idx+1][i] == ',' || argv[idx+1][i] == ':')
	c++;
    }
    //ToDo: below, Strict checking of format float[{,:}float]*
    fptr = (FLOAT*)calloc(c+1, sizeof(FLOAT));
    *((FLOAT**)dest) = fptr;
    fptr[0] = atof(argv[idx+1]);
    for (i = 0, c = 1; i < (int)strlen(argv[idx+1]); i++){
      if (argv[idx+1][i] == ',' || argv[idx+1][i] == ':'){
	fptr[c] = atof(&argv[idx+1][i+1]);
	c++;
      }
    }

    break;
  default:
    snprintf(cptr, 4096, "Internal: Unable to assign value to option '%s'.", argv[idx]);
  }
  if (cptr[0] != '\0')
    AddError(cptr);

  return c;
}

/*****************************************************************************
Description: Retrieve attribute value for a given command line parameter.
             The expected format is:
             -par:<val>

Return value: Number of successfully assigned values.
*****************************************************************************/
int GetAttribVal(char *argv, int offset, int type, void *dest)
{
  int i, c = 0;
  char cptr[4096];
  FLOAT *fptr;

  if (argv[offset] != ':')
    return 0; //No attribute value

  offset++;
  cptr[0] = '\0';
  switch (type){
  case TYPE_STRING:
    /* check: "-" is valid but "-<string>" is not */
    if (argv[offset] == '-' && strcmp(&argv[offset], "-")){
      snprintf(cptr, 4096, "Missing value for parameter '%s'.", argv);
      return -1;
    }
    else{
      *(char**)dest = strdup(&argv[offset]);
    }
    break;
  case TYPE_INT:
    if (argv[offset] == '-' || isdigit(argv[offset])){
      *(int*)dest = atoi(&argv[offset]);
    }
    else{
      snprintf(cptr, 4096,"Invalid or missing value for parameter '%s'.",argv);
      return -1;
    }
    break;
  case TYPE_UNSIGNED:
    if (isdigit(argv[offset])){
      *(unsigned*)dest = atoi(&argv[offset]);
      c = 1;
    }
    else{
      snprintf(cptr, 4096,"Invalid of missing value for parameter '%s'.",argv);
      return -1;
    }
    break;
  case TYPE_FLOAT:
    if (argv[offset] == '-' || isdigit(argv[offset])|| argv[offset] == '.'){
      *(float*)dest = atof(&argv[offset]);
    }
    else{
      snprintf(cptr, 4096, "Invalid or missing value for parameter '%s'.", argv);
      return -1;
    }
    break;
  case TYPE_FLOAT_VECT:  //Float vector in the form of float[{,:}float]*
    //count number of commas, an colons in argument string
    for (i = 0, c = 1; i < (int)strlen(argv); i++){
      if (argv[i] == ',' || argv[i] == ':')
	c++;
    }
    //ToDo: below, Strict checking of format float[{,:}float]*
    fptr = (FLOAT*)calloc(c, sizeof(FLOAT));
    *((FLOAT**)dest) = fptr;
    for (i = 0, c = 0; i < (int)strlen(argv); i++){
      if (argv[i] == ',' || argv[i] == ':'){
	fptr[c] = atof(&argv[i+1]);
	c++;
      }
    }

    break;
  default:
    snprintf(cptr, 4096, "Internal: Unable to assign value to option '%s'.", argv);
  }
  if (cptr[0] != '\0')
    AddError(cptr);

  return 1;
}

/*****************************************************************************
Description: Some command line parameters can hold several values. This
             function retrieves multiple-values for a command line parameter.
             Error checking is performed to ensure the value is available and
             the type of the value is correct. The format string can contain
             the characters u,i,s,f, and special characters [,].:,and comma.
             The [ and ] encolse optional values, whereas : and comma are
             separate concatenated values. If : or , is not given, then the
             values are assumed to be separately stated in the command line.
             For example, the format string "ui:i[s]" would expect an unsigned
             followed by two integers which are separated by an :, and an
             optional string. Hence, the following command line sub-sequences
             would be correct for this example "123 7:89 test", and "1 -9:88".

Return value: The function does not return a value.
*****************************************************************************/
void GetMultiArg(char **argv, int idx, const char *fmt, ...)
{
  char cptr[4096]; /* A buffer for messages (warnings, errors, etc)      */
  va_list ap;      /* Points to each unnamed argument in turn           */
  void *dest = NULL;
  int i;
  int optional;    /* A flag indicating whether a parameter is optional */
  char *arg;       /* Pointer to an command line argument               */
  const char *p=fmt; /* Pointer to the format string                    */

  i = 1;
  optional = NO;
  cptr[0] = '\0';    /* Init error message buffer       */
  arg = argv[idx+i];
  va_start(ap, fmt); /* point to first element after fmt*/
  while(*p && *arg != '\0'){
    if (isalpha((int)*p))
      dest = va_arg(ap, void*);
    switch(tolower((int)*p)){
    case 's': /* Get a string value */
      /* check: "-" is valid but "-<string>" is not */
      if (*arg == '\0' || (*arg == '-' && strcmp(arg, "-") && optional == NO)){
	snprintf(cptr, 4096, "Invalid or missing value for parameter '%s'.", argv[idx]);
	*(char**)dest = NULL;
      }
      else
	*(char**)dest = strdup(arg);
      break;
    case 'i': /* Get a signed integer value */
      if (satoi(arg, (int*)dest) == 0){
	if (optional == NO){
	  if (*arg == '-')
	    snprintf(cptr, 4096, "Missing value for parameter '%s'.", argv[idx]);
	  else
	    snprintf(cptr, 4096, "Invalid value for parameter '%s'.", argv[idx]);
	  *(int*)dest = (int)0;
	}
      }
      break;
    case 'u': /* Get an unsigned value */
      if (satou(arg, (unsigned*)dest) == 0){
	if (optional == NO){
	  if (*arg == '-' && !isdigit((int)*arg))
	    snprintf(cptr, 4096, "Missing value for parameter '%s'.", argv[idx]);
	  else
	    snprintf(cptr, 4096, "Invalid value for parameter '%s'.", argv[idx]);
	  *(unsigned*)dest = 0u;
	}
      }
      break;
    case 'f': /* Get a float value */
      if (satof(arg, (float*)dest) == 0 && optional == NO){
	snprintf(cptr, 4096, "Missing or invalid value for parameter '%s'.", argv[idx]);
	*(float*)dest = (float)0.0;
      }
      break;
    case ',': case ':':
      for (; *arg != '\0' && *arg != ',' && *arg != ':'; arg++);
      if (*arg != '\0') arg++;
      else {i++; arg = argv[idx+i];}
      break;
    case '[': /* Anything between [] is optional */
      optional = YES;
      break;
    case ']': /* End of optional values */
      optional = NO;
      break;
    default:
      snprintf(cptr, 4096, "Internal: Unable to assign value to option '%s'.", argv[idx]);
    }/*switch*/
    ++p; /* get the next char of the format string */
    if (isalpha((int)(*(p-1))) && isalpha((int)*p)){
      i++;
      arg = argv[idx+i];
    }
  }/*while*/
  if (cptr[0] != '\0')
    AddError(cptr);

  va_end(ap); /*cleanup*/ 
}



/* Error handling and error reporting functions */
int NumErrors  = 0;
char **ErrorMessages = NULL;
/*****************************************************************************
Description: Add a message given by msg to a list of error messages.

Return value: The function does not return a value.
*****************************************************************************/
void AddError(const char* msg)
{
  if (msg == NULL)
    return;

  NumErrors++;
  ErrorMessages = (char**) MyRealloc(ErrorMessages, NumErrors * sizeof(char*));
  ErrorMessages[NumErrors-1] = strdup(msg);
}

/*****************************************************************************
Description: Print all error messages in the list, then clear all messages
             from the list.

Return value: The function does not return a value.
*****************************************************************************/
void PrintErrors()
{
  int i;

  for (i = 0; i < NumErrors; i++){
    fprintf(stderr, "Error: %s\n", ErrorMessages[i]);
    free(ErrorMessages[i]);
  }
  free(ErrorMessages);
  ErrorMessages = NULL;
  NumErrors = 0;
}

/*****************************************************************************
Description: Clear all messages from the list of error messages.

Return value: The function does not return a value.
*****************************************************************************/
void ClearErrors()
{
  int i;

  for (i = 0; i < NumErrors; i++)
    free(ErrorMessages[i]);
  free(ErrorMessages);
  ErrorMessages = NULL;
  NumErrors = 0;
}

/*****************************************************************************
Description: Return the number of error messages in the list.

Return value: The number of error messages in the list.
*****************************************************************************/
unsigned CheckErrors()
{
  return NumErrors;
}


/* Message buffering and reporting functions */
int NumMessages  = 0;
char **Messages = NULL;
/*****************************************************************************
Description: Add a message given by msg to a list of messages.

Return value: The function does not return a value.
*****************************************************************************/
void AddMessage(const char* msg)
{
  if (msg == NULL)
    return;

  NumMessages++;
  Messages = (char**) MyRealloc(Messages, NumMessages * sizeof(char*));
  Messages[NumMessages-1] = strdup(msg);
}

/*****************************************************************************
Description: Clear all messages from the list of messages.

Return value: The function does not return a value.
*****************************************************************************/
void ClearMessages()
{
  int i;

  for (i = 0; i < NumMessages; i++)
    free(Messages[i]);
  free(Messages);
  Messages = NULL;
  NumMessages = 0;
}

/*****************************************************************************
Description: Print all error messages in the list, then clear all messages
             from the list.

Return value: The function does not return a value.
*****************************************************************************/
void PrintMessages()
{
  int i;

  for (i = 0; i < NumMessages; i++)
    fprintf(stderr, "%s\n", Messages[i]);

  ClearMessages();
}

/*****************************************************************************
Description: Return the number of messages in the list.

Return value: The number of error messages in the list.
*****************************************************************************/
unsigned CheckMessages()
{
  return NumMessages;
}


/* Functions for a simple progress meter */
int ProgressTargetValue = 0;
int ProgressOldState = -1;
time_t ProgressOldTime = 0;
time_t ProgressStartTime = 0;
/*****************************************************************************
Description: Initialize the progress meter. The progress meter assumes that the
             state of the progress is indicated by a value which ranges from
             0 (no progress yet) to max (task completed). A negative value for
             max indicates that the maximum value is not known.

Return value: The function does not return a value.
*****************************************************************************/
void InitProgressMeter(int max)
{
  ProgressTargetValue = max;
  ProgressOldState = -1;
  ProgressOldTime = 0;
  ProgressStartTime = time(NULL);
}

/*****************************************************************************
Description: Print the progress. state indicates the progress where 0 (no
             progress yet), and max (task completed). If max is not known
             (a negative value for max indicates this), then the state value
             is printed directly. If max is known, then an estimated time
             remaining value is printed. Old states are deleted before the
             new state is printed.

Return value: The function does not return a value.
*****************************************************************************/
void PrintProgress(int state)
{
  time_t t;

  t = time(NULL);
  if (t - ProgressOldTime < 1) /* Print progress no more than every 1 second */
    return;

  if (ProgressTargetValue < 0){ /* If we don't know the target value */
    if (ProgressOldState >= 0){ /* This is not the first time we print   */
      int i;
      for (i = 0; i <= (int)log10f(ProgressOldState); i++)
	fputc('\b', stderr);  /* Remove the old state */
    }
    fprintf(stderr, "%u", state);
  }
  else{
    time_t remain;
    remain = (time_t)(t - ProgressStartTime)*((float)ProgressTargetValue/state-1.0);
    if (ProgressOldState >= 0) /* This is not the first time we print   */
      fprintf(stderr, "\b\b\b\b\b");  /* Remove the old state */

    if (remain < 100)
      fprintf(stderr, "%2dsec", (int)remain);    /* Remaining time is seconds*/
    else if (remain/60 < 100)
      fprintf(stderr, "%2dmin", (int)remain/60); /* Remaining time is minutes*/
    else if (remain/3600 < 100)
      fprintf(stderr, "%2dhrs", (int)remain/3660);/* Remaining time is hours */
    else if (remain/86400 < 100)
      fprintf(stderr, "%2dday", (int)remain/8640);/* Remaining time is days  */
    else
      fprintf(stderr, "TIME!");     /* Remaining time exceeds 99 days!!  */
  }
  ProgressOldState = state;
  ProgressOldTime = t;
}

/*****************************************************************************
Description: Delete old progress values if there was progress printed in the
             past.

Return value: The function does not return a value.
*****************************************************************************/
void StopProgressMeter()
{
  if (ProgressOldState < 0)
    return;

  if (ProgressTargetValue < 0){ /* If we don't know the target value */
    int i;
    for (i = 0; i <= (int)log10f(ProgressOldState); i++)
      fputc('\b', stderr);  /* Remove the old state */
  }
  else
    fprintf(stderr, "\b\b\b\b\b");  /* Remove the old state */
}


/* Math and logic functions */
/*****************************************************************************
Description: Check whether the values in val1 and val2 are approximately the
             same. The deviation value specifies by how many percent val1 and
             val2 may differ in order to qualify as being approximately the
             same.

Return value: 1 if val1 and val2 are approximately the same in value.
              0 else
*****************************************************************************/
int approx(float val1, float val2, float deviation)
{
  if (fmaxf(fabsf(val2), fabsf(val1)) * deviation >= fabsf(val2 - val1))
    return 1;
  else
    return 0;
}

/*****************************************************************************
Description: Check whether val1 and val2 are similar in value. The threshold
             value for the similarity specifies by how much the val1 and val2
             may be different in order to qualify as being similar.

Return value: 1 if val1 and val2 are similar.
              0 else
*****************************************************************************/
int similar(float val1, float val2, float threshold)
{
  if (fabs(val2 - val1) <= fabs(threshold))
    return 1;
  else
    return 0;
}

/*****************************************************************************
Description: Initializer for function BitCount (see below)

Return value: Pointer to a pre-initialized bit-array. This array acts as a mask
              which allows fast counting of bits in another array.
*****************************************************************************/
int *BitInit(int size)
{
  int *array, i;

  array = (int*) MyMalloc(size * sizeof(int));
  for (i = 0; i < size; i++)
    array[i] = i;

  return array;
}

/*****************************************************************************
Description: Count number of bits in array. Array is of size bytes. Works only
             for arrays with 32-bit integers. This function currently is
             limited to arrays of maximum size 256*32=8192 bits.

Return value: number of '1' bits set in array.
*****************************************************************************/
int BitCount(int *array, int size)
{
  int count;
  static int *bits_in_char = NULL;

  if (!bits_in_char)
    bits_in_char = BitInit(256);

  count = 0;
  while (--size >= 0){
    count += bits_in_char [array[size]     & 0xffu]
      +  bits_in_char [(array[size] >>  8) & 0xffu]
      +  bits_in_char [(array[size] >> 16) & 0xffu]
      +  bits_in_char [(array[size] >> 24) & 0xffu] ;
  }
  return count;
}


/*****************************************************************************
Description: Sum all elements in a given float vector.

Return value: Sum of elements.
*****************************************************************************/
FLOAT fsum(int num, FLOAT *fptr)
{
  FLOAT sum = 0;

  while(--num >= 0){
    sum += fptr[num];
  }
  return sum;
}



/* File handling functions */
/*****************************************************************************
Description: This is a fail-save equivalent to fopen which expects the same
             parameters, and returns the same value as fopen. The difference
             is that the function does not return on error. Instead it prints
             an error message to stderr, then exits immediately.

Return value: Same is in fopen.
*****************************************************************************/
FILE *MyFopen(const char *path, const char *mode)
{
  FILE *stream = NULL;

  if (path == NULL || mode == NULL){
    fprintf(stderr, "\nError: Call to MyFopen with NULL parameters\n");
    exit(0);
  }
  else if (!strcmp(path, "-")){
    if (*mode == 'r')
      return stdin;
    else if (*mode == 'w')
      return stdout;
  }
  if ((stream = fopen(path, mode)) == NULL){
    fprintf(stderr, "\nError: Unable to open file '%s'\n", path);
    exit(0);
  }
  return stream;
}

/*****************************************************************************
Description: This is a fail-save equivalent to fclose which expects the same
             parameters, and returns the same value as fclose.

Return value: Same as in fclose
*****************************************************************************/
int MyFclose(FILE *stream)
{
  if (stream != NULL)
    return fclose(stream);
  else
    return 0;
}

/*****************************************************************************
Description: Returns the compress status of a existing file pointed to by
             <path>.

Return value: GZIP if file is gzip compressed.
              BZIP if file is bzip2 compressed.
              RAW  else.
*****************************************************************************/
int GetCompressStatus(const char *path, const char *mode)
{
  FILE *myfile;
  unsigned int buf = 0;

  myfile = MyFopen(path, mode);
  if (fread(&buf, 1, 2, myfile) < 2){ /* Attempt to get the magic number */
    MyFclose(myfile);    /* Empty or very small files are */
    return RAW;          /* assumed RAW                   */ 
  }
  MyFclose(myfile);

  if (buf == 0x8b1f)     /* Magic number of a gzip file  */   
    return GZIP;
  else if (buf == 0x5a42)/* Magic number of a bzip2 file */    
    return BZIP;
  else                                  
    return RAW;          /* Neither BZIP2 nor GZIP */ 
  }
/* End of file */

#ifndef TRAIN_H_DEFINED
#define TRAIN_H_DEFINED

void InstallHandlers();
void FindWinnerEucledian(struct Map*,struct Node*,struct Graph*,struct Winner*);
void VQFindWinnerEucledian(struct Map *map, struct Node *node, struct Graph *gptr, struct Winner *winner);
int TrainMap(struct Parameters *parameters);
float ComputeHexaDistance(int bx, int by, int tx, int ty);
void ComputeRetrievalPerformance(struct Parameters parameters, int classifyflag);

#endif

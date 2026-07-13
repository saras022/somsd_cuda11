
void FindWinnerEucledian_CPU(struct Map*,struct Node*,struct Graph*,struct Winner*);
void GaussianAdapt_CPU(struct Graph *gptr, struct Map *map, struct Node *node,
		   struct Winner *winner, FLOAT radius, FLOAT alpha);

int TrainMap_CPU(struct Parameters *parameters);


#ifndef LOSS
#define LOSS
#include"batch.h"
Batch d_L2_Loss (Batch& prediciton, Batch& target);
double L2_Loss (Batch& prediciton, Batch& target);
Batch d_Log_Loss (Batch& prediction, Batch& target);
double Log_Loss(Batch& prediction, Batch& target);
#endif
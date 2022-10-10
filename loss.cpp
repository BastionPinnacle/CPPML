#include"batch.h"
#include<cmath>
Batch d_L2_Loss (Batch& prediction, Batch& target){
    Batch newBatch;
    uint batch_size = prediction.size();
    for(uint batch = 0 ; batch<batch_size ; batch++){
        uint vector_size = prediction[batch].size();
        std::vector<double> w(vector_size,0.0);
        for(uint i = 0 ; i < vector_size ; i++){
            w[i] = (prediction[batch][i]-target[batch][i])/batch_size;
        }
        newBatch.push_back(w);
    }
    return newBatch;
}

double L2_Loss (Batch& prediction, Batch& target){
    uint batch_size = prediction.size();
    double loss = 0.0;
    for(uint batch = 0 ; batch<batch_size ; batch++){
        uint vector_size = prediction[batch].size();
        for(uint i = 0 ; i < vector_size ; i++){
            loss += ((prediction[batch][i]-target[batch][i])*(prediction[batch][i]-target[batch][i]))/(2*batch_size);
        }
    }
    return loss;
}

double Log_Loss(Batch& prediction, Batch& target){
    uint batch_size = prediction.size();
    double loss = 0.0;
    for(uint batch = 0 ; batch<batch_size ; batch++){
        uint vector_size = prediction[batch].size();
        for(uint i = 0 ; i < vector_size ; i++){
            //std::cout<<loss<<std::endl;
            loss -= target[batch][i] * std::log(prediction[batch][i]);
        }
    }
    return loss;
}

Batch d_Log_Loss(Batch& prediction, Batch& target){
    Batch newBatch;
    uint batch_size = prediction.size();
    for(uint batch = 0 ; batch<batch_size ; batch++){
        uint vector_size = prediction[batch].size();
        std::vector<double>w(vector_size,0.0);
        for(uint i = 0 ; i < vector_size ; i++){
            w[i] = -target[batch][i]/prediction[batch][i];
        }
        newBatch.push_back(w);
    }
    return newBatch;
}
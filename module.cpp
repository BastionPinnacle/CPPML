#include"module.h"

Module::Module(){}
void Module::info(){
    std::cout<<"You are interacting with an object of class 'Module'"<<std::endl;
}
Module::~Module(){}

Batch Module::backward(Batch& batch){
    std::cout<<"Module base class backward called, returning the same batch"<<std::endl;
    return batch;
}

Batch Module::forward(Batch& batch){
    std::cout<<"Module base class forward called, returning the same batch"<<std::endl;
    return batch;
}

void Module::step(double learning_rate){
    //std::cout<<"Module base class step called, nothing is done"<<std::endl;
}
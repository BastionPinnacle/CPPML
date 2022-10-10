#include"sequential.h"
#include"linear.h"

Sequential::Sequential(){}

Batch Sequential::forward(Batch& batch){
    Batch input = std::move(batch);
    for(auto element : sequence){
        Batch output = element->forward(input);
        input = output;
    }
    return input;
}

Batch Sequential::backward(Batch& batch){
    Batch input = std::move(batch);
    for(auto it = sequence.rbegin(); it!=sequence.rend();it++){
        Module* element = *it;
        //element->info();
        Batch output = element->backward(input);
        input = output;
    }
    return input;
}

void Sequential::step(double learning_rate){
    for(auto element : sequence){
        element->step(learning_rate);
    }
}

void Sequential::add(Module* module){
    sequence.push_back(module);
}
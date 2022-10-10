#include"function.h"
Function::Function(){}

Softmax::Softmax(){};
Batch Softmax::forward(Batch& batch){
    for(auto iterator = batch.begin(); iterator!=batch.end(); iterator++){
        auto& v = *iterator;
        double max = *std::max_element(v.begin(), v.end());
        double denominator = 0;
        std::for_each(v.begin(), v.end(), [max,&denominator](double &n){ n-=max;n = std::exp(n);denominator+=n; });
        std::for_each(v.begin(), v.end(), [&denominator](double &n){ n/=denominator; });
    }
    this->softmax = batch;
    return batch;
}
Batch Softmax::backward(Batch& batch){
    Batch newBatch;
    for(auto iterator = batch.begin(); iterator!=batch.end(); iterator++){
        auto v = *iterator;
        auto softmax_iterator = this->softmax.begin();
        auto s = *softmax_iterator;
        uint size = v.size();
        std::vector<double> w(size, 0.0);
        for(uint i = 0; i < size ; i++){
            for(uint j = 0 ; j < size; j++){
                if(i==j) w[i] += s[i]*(1-s[i])*v[j];
                else w[i] += -s[i]*s[j]*v[j];
            }            
        }
        newBatch.push_back(w);
        softmax_iterator++;
    }
    return newBatch;
}

void Softmax::info(){
    std::cout<<"You're interacting with object of class 'Softmax'"<<std::endl;
}

Sigmoid::Sigmoid(){};
Batch Sigmoid::forward(Batch& batch){
    for(auto iterator = batch.begin(); iterator!= batch.end(); iterator++){
        auto& v = *iterator;
        std::for_each(v.begin(), v.end(), [](double &n){ n = 1/(1+std::exp(-n)); });
    }
    this->sigmoid = batch;
    return batch;
}

Batch Sigmoid::backward(Batch& batch){
    Batch newBatch;
    auto sigmoid_iterator = sigmoid.begin();
    for(auto iterator = batch.begin(); iterator!= batch.end(); iterator++,sigmoid_iterator++){
        auto v_it = *iterator;
        auto s_it = *sigmoid_iterator;
        auto sigmoid_iterator2 = s_it.begin();
        auto w = std::vector<double>(v_it.size(),0.0);
        uint i = 0 ;
        for(auto iterator2 = v_it.begin(); iterator2!= v_it.end(); iterator2++,sigmoid_iterator2++,i++){
            auto v = *iterator2;
            auto s = *sigmoid_iterator2;
            w[i] = v*s*(1-s);
        }
        newBatch.push_back(w);
    }
    return newBatch;
}

void Sigmoid::info(){
    std::cout<<"You're interacting with object of class 'Sigmoid'"<<std::endl;
}

    

ReLU::ReLU(double a = 0.1):a(a){};
Batch ReLU::forward(Batch& batch){
    Batch newBatch;
    for(auto iterator = batch.begin(); iterator!= batch.end(); iterator++){
        auto v = *iterator;
        std::for_each(v.begin(), v.end(), [this](double &n){ n = n>0 ? this->a*n : 0;  });
        this->relu.push_back(v);
    }
    return this->relu;
}

Batch ReLU::backward(Batch& batch){
    Batch newBatch;
    auto relu_iterator = relu.begin();
    for(auto iterator = batch.begin(); iterator!= batch.end(); iterator++,relu_iterator++){
        auto v_it = *iterator;
        auto r_it = *relu_iterator;
        auto relu_iterator2 = r_it.begin();
        auto w = std::vector<double>(v_it.size(),0.0);
        uint i = 0 ;
        for(auto iterator2 = v_it.begin(); iterator2!= v_it.end(); iterator2++,relu_iterator2++,i++){
            auto v = *iterator2;
            auto s = *relu_iterator2;
            w[i] = v*s;
        }
        newBatch.push_back(w);
    }
    return std::move(newBatch);
}
    
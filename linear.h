#ifndef LINEAR
#define LINEAR
#include<random>
#include"module.h"
extern std::normal_distribution<double> normal_00_10;
extern std::uniform_real_distribution<double> uniform_neg100_100;
static std::default_random_engine generator;

template<int N, int M>
class Linear : public Module{
    public:
        Linear();
        ~Linear();
        Batch forward(Batch& batch);
        Batch backward(Batch& batch);
        void step(double learning_rate);
        void print();
        void info();

    private:
        Batch x_batch;
        double d_weights[N][M];
        double d_biases[N];
        double weights[N][M];
        double biases[N];
};

template<int N, int M>
Linear<N,M>::Linear(){
    for(uint i = 0 ; i < N;i++){
        for(uint j = 0 ; j < M ; j++) weights[i][j] = normal_00_10(generator);
        biases[i] = normal_00_10(generator);
    }
    x_batch;
}
template<int N, int M>
Linear<N,M>::~Linear(){}
template<int N, int M>
Batch Linear<N,M>::forward(Batch& batch){
    Batch newBatch;
    for(auto batch_iterator = batch.begin(); batch_iterator!=batch.end(); batch_iterator++){
        std::vector<double> w(N,0.0);
        std::vector<double> v = *batch_iterator;
        for(uint i = 0 ; i < N ; i++){
            for(uint j = 0 ; j < M ; j++){
                w[i]+=weights[i][j]*v[j];
            }
            w[i]+=biases[i];
        }
        newBatch.push_back(w);
    }
    this->x_batch = std::move(batch);
    return newBatch;
}
template<int N, int M>
Batch Linear<N,M>::backward(Batch& batch){
    //STEP ONE - BACKWARD FLOW OF INFORMATION
    Batch newBatch;
    for(auto batch_iterator = batch.begin(); batch_iterator!=batch.end(); batch_iterator++){
        std::vector<double> w(M,0.0);
        std::vector<double> v = *batch_iterator;
        for(uint i = 0 ; i < M ; i++){
            for(uint j = 0 ; j < N ; j++){
                w[i]+=weights[j][i]*v[j];
            }
        }
        newBatch.push_back(w);
    }
    //STEP TWO - CALCULATING WEIGHT GRADIENTS
    //d_weights = {0};
    for(uint i = 0 ; i < N ; i++){for(uint j = 0 ; j < M ; j++) d_weights[i][j]=0;}
    
    auto batch_iterator = batch.begin();
    auto x_batch_iterator = x_batch.begin();
    for(batch_iterator, x_batch_iterator; batch_iterator!=batch.end(), x_batch_iterator!=x_batch.end(); batch_iterator++,x_batch_iterator++){
        auto v = *batch_iterator;
        auto w = *x_batch_iterator;
        for(uint i = 0 ; i < N ; i++){
            for(uint j = 0 ; j  < M ; j++){
                d_weights[i][j]+=(v[i]*w[j])/batch.size();
            }
        }
    }
    //STEP THREE CALCULATING BIAS GRADIENTS
    //d_biases = {0};
    for(uint i = 0 ; i < N ; i++) d_biases[i]=0.0;
    
    for(auto batch_iterator = batch.begin(); batch_iterator!=batch.end(); batch_iterator++){
        std::vector<double> v = *batch_iterator;
        for(uint i = 0 ; i < v.size();i++){
            d_biases[i] += v[i]/batch.size();    
        }
    }
    return newBatch;
}
template<int N, int M>
void Linear<N,M>::step(double learning_rate){
    for(uint i = 0 ; i < N ; i++){
        for(uint j = 0 ; j < M ; j++){
            weights[i][j]-= learning_rate*d_weights[i][j];
        }
        biases[i] -= d_biases[i];
    }
}
template<int N, int M>
void Linear<N,M>::print(){
    for(uint i = 0 ; i < N ; i++){
        for(uint j = 0 ; j < M ; j++){
            std::cout<<weights[i][j]<<" ";
        }
        std::cout<< biases[i]<<std::endl;
    }
}

template<int N, int M>
void Linear<N,M>::info(){
    std::cout<<"You're interacting with object of class 'Linear'"<<std::endl;
    this->print();
}
#endif
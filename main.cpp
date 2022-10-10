#include"batch.h"
#include"loss.h"
#include"module.h"
#include"function.h"
#include"linear.h"
#include"sequential.h"
#include<tuple>
#include<iomanip>


double accuracy(Batch& prediction, Batch& target){
    uint size = std::min(prediction.size(),prediction.size());
    uint correct = 0;
    for(uint i = 0 ;  i < size ; i++){
        std::vector<double> p = prediction[i];
        std::vector<double> t = target[i];
        std::vector<double>::iterator result_p = max_element(p.begin(), p.end());
        int argmaxVal_p = distance(p.begin(), result_p);
        std::vector<double>::iterator result_t = max_element(t.begin(), t.end());
        int argmaxVal_t = distance(t.begin(), result_t);
        if(argmaxVal_p==argmaxVal_t) correct++;
    }
    return ((double)correct)/((double)size);
}

std::tuple<Batch,Batch> generate_linear(uint batch_size, std::vector<double> a, double b){
    Batch x_batch;
    Batch y_batch;
    for(uint i = 0 ; i < batch_size ; i++){
        std::vector<double> x(a.size(),0.0);
        std::vector<double> y(1,0.0);
        for(uint j = 0 ; j < a.size(); j++){
            x[j] = uniform_neg100_100(generator);
            y[0] += a[j]*x[j];
        }
        y[0]+=b;
        std::normal_distribution<double> normal(y[0],10.0);
        y[0] = normal(generator);
        x_batch.push_back(x);
        y_batch.push_back(y);
    }
    return {x_batch,y_batch};
}

std::tuple<Batch,Batch> generate_gaussians(uint batch_size, std::vector<std::vector<double>> mean, std::vector<std::vector<double>> std_dev ){
    Batch x_batch;
    Batch y_batch;
    for(uint vector_num = 0 ; vector_num<batch_size; vector_num++){
        uint size = mean.size();
        std::uniform_int_distribution<int> range(0,size-1);
        uint index = range(generator);
        uint vector_size = mean[0].size();
        std::vector<double> v(vector_size,0.0);
        for(uint i = 0 ; i < vector_size ; i++){
            std::normal_distribution<double> normal(mean[index][i],std_dev[index][i]);
            v[i] = normal(generator);
        }
        x_batch.push_back(v);
        std::vector<double> id_x(vector_size,0.0);
        id_x[index] = 1;
        y_batch.push_back(id_x);
    }
    return {x_batch,y_batch};
}




/*
int main(){
    //GENERATING DATASET X,Y ~ P(Y|X) = NORMAL(A*X+B,1.0)
    Batch x_batch_main;
    Batch y_batch_main;
    static std::default_random_engine generator;
    std::tie(x_batch_main, y_batch_main) = generate_linear(100,{3.0,5.0,7.0},9.0);

    //x_batch_main.print();
    //y_batch_main.print();
    //CREATING LINEAR LAYER
    Linear<1,3> linear;
    //LEARNING PROCESS    
    double min_loss = 100000;
    for(double learning_rate = 0.01; learning_rate>=0.006 ; learning_rate/=2){
        double loss = min_loss;
        for(uint i = 0 ; i <= 20000; i++){
            Batch x_batch = x_batch_main;
            Batch y_batch = y_batch_main;
            Batch prediction = linear.forward(x_batch);
            loss = L2_Loss(prediction,y_batch);
            Batch d_Loss = d_L2_Loss(prediction,y_batch);
            linear.backward(d_Loss);
            linear.step(learning_rate);
        }
        if(loss<min_loss) min_loss=loss;
    }
    std::cout<<min_loss<<std::endl;
    linear.print();
    linear.info();
}
*/
int main(){
    //GENERATING DATASET X,Y ~ P(Y|X) = NORMAL(A*X+B,1.0)
    Batch x_train;
    Batch y_train;
    static std::default_random_engine generator;
 
    std::vector<std::vector<double>> mean; 
    std::vector<std::vector<double>> variance;
    mean.push_back({5.0,10.0});
    mean.push_back({7.0,2.0});
    variance.push_back({1.0,1.0});
    variance.push_back({1.0,1.0});

    std::tie(x_train, y_train) = generate_gaussians(10,mean,variance);

    //CREATING NEURAL NET WITH
    Linear<4,2> linear1;
    Sigmoid sigmoid1;
    Linear<2,4> linear2;
    Softmax softmax1;
    Sequential neural_net;
    neural_net.add(&linear1);
    neural_net.add(&sigmoid1);
    neural_net.add(&linear2);
    neural_net.add(&softmax1);
    //LEARNING PROCESS    
    for(double learning_rate = 0.01; learning_rate>=0.01 ; learning_rate/=2){
        //double loss = min_loss;

        for(uint i = 0 ; i <= 10000; i++){
            Batch train = x_train;
            Batch prediction = neural_net.forward(train);
            Batch loss = d_Log_Loss(prediction,y_train);
            neural_net.backward(loss);
            neural_net.step(learning_rate);
            if(i%1000==0) std::cout<<"LOSS := "<<Log_Loss(prediction,y_train)<<std::endl;
        }
    }
    Batch x_test,y_test;
    std::tie(x_test,y_test) = generate_gaussians(10000,mean,variance);
    Batch prediction = neural_net.forward(x_test);
    double acc = accuracy(prediction, y_test);
    std::cout<<acc<<std::endl;
}
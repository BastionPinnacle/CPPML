#include"batch.h"
Batch::Batch(){
    //std::cout<<"Default constructor called"<<std::endl;
    batch_size = 0;
}
Batch::Batch(Batch&& oldBatch){
    //std::cout<<"Move constructor called"<<std::endl;
    batch = std::move(oldBatch.batch);
    batch_size = std::move(oldBatch.batch_size);
}
Batch::Batch(Batch& oldBatch){
    //std::cout<<"Copy constructor called"<<std::endl;
    batch = oldBatch.batch;
    batch_size = oldBatch.batch_size;
}
Batch& Batch::operator=(Batch&& oldBatch){
    //std::cout<<"Move assignment operator called"<<std::endl;
    batch = std::move(oldBatch.batch);
    batch_size = std::move(oldBatch.batch_size);
    return *this;
}
Batch& Batch::operator=(Batch& oldBatch){
    //std::cout<<"Copy assignment operator called"<<std::endl;
    batch = oldBatch.batch;
    batch_size = oldBatch.batch_size;
    return *this;
}
std::vector<double>& Batch::operator[](int i){
    return batch[i];
}
uint Batch::size(){
    return batch_size;
}
std::vector<std::vector<double>>::iterator  Batch::begin() { return batch.begin(); }
std::vector<std::vector<double>>::iterator  Batch::end()   { return batch.end(); }
void Batch::push_back(std::vector<double> v){batch.push_back(v);batch_size++;}
void Batch::print(){
    for(auto iterator = batch.begin(); iterator!=batch.end();iterator++){
        auto& vector = *iterator;
        for(auto& v : vector){
            std::cout<<v<<" ";
        }
        std::cout<<std::endl;
    }
}
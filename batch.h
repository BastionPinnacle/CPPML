#ifndef BATCH
#define BATCH
#include<vector>
#include<iostream>
class Batch{
    public:        
        Batch();
        Batch(Batch&& oldBatch);
        Batch(Batch& oldBatch);
        Batch& operator=(Batch&& oldBatch);
        Batch& operator=(Batch& oldBatch);
        std::vector<double>& operator[](int i);
        uint size();
        std::vector<std::vector<double>>::iterator begin();
        std::vector<std::vector<double>>::iterator end();
        void push_back(std::vector<double> v);
        void print();
    private:
        uint batch_size;
        std::vector<std::vector<double>> batch;
};
#endif
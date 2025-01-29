#pragma once
#include"Batch.hpp"
#include"Module.hpp"

class Sequential{
    public:
        Sequential();
        Batch forward(Batch&);
        Batch backward(Batch&);
        void step(double);
        void add(Module*);
    private:
        std::vector<Module*> sequence;
};
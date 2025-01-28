#ifndef SEQUENTIAL
#define SEQUENTIAL
#include"batch.h"
#include"module.h"

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
#endif

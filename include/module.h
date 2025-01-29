#pragma once
#include"batch.h"
#include<algorithm>
#include<cmath>
class Module{
    public:
        Module();
        virtual Batch forward(Batch&);
        virtual Batch backward(Batch&);
        virtual void step(double learning_rate);
        virtual void info();
        virtual ~Module();
};
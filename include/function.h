#ifndef FUNCTION
#define FUNCTION
#include"module.h"
class Function : public Module {
    public: 
        Function();
};

class Softmax : public Function{
    public:
        Softmax();
        Batch forward(Batch& batch);
        Batch backward(Batch& batch);
        void print();
        void info();
    private:
        Batch softmax;
};

class Sigmoid: public Function{
    public:
        Sigmoid();
        Batch forward(Batch& batch);
        Batch backward(Batch& batch);
        void info();
    private:
        Batch sigmoid;
};

class ReLU: public Function{
    public:
        ReLU(double);
        Batch forward(Batch& batch);
        Batch backward(Batch& batch);
    private:
        double a;
        Batch relu;
};
#endif
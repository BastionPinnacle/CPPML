#!/bin/bash
g++ -c batch.cpp loss.cpp module.cpp function.cpp sequential.cpp
g++ -c main.cpp
g++ -o main batch.o loss.o module.o function.o linear.o sequential.o main.o
#include <iostream>
#include <fstream>
#include <math.h>
#include "SA.h"
#include <vector>
#include <chrono>

int main(int argc, char * argv[])
{
    std::string filename;
    float coolingRate;
    int temperature;
    std::string fname;
    if (argc == 2)
    {
        fname = argv[1];
    }
    // char filename[] = "data/berlin52.txt";
    else{
        fname = "data/berlin52.txt";
    }
        std::fstream myfile(fname, std::ios_base::in);

    int size;
    int id,x,y;
    myfile >> size;
    float** matrix = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; i++)
        matrix[i] = (float*)malloc(size * sizeof(float));
    


    int XValues[size];
    int YValues[size];


    for (size_t i = 0; i < size; i++)
    {
        myfile >> id >> x >>y;
        XValues[i] = x;
        YValues[i] = y;
    }
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            float distance = sqrt(pow(XValues[i] - XValues[j], 2) + pow(YValues[i] - YValues[j], 2));
            matrix[i][j] = (i==j)?0:distance;
        }
    }   


    SA sa = SA(matrix, size );
    
    auto start = std::chrono::steady_clock::now();

    float result = sa.apply();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout <<  diff.count() << " "<<result << std::endl;
    free(matrix);

      




    return 0;
}
#include <iostream>
#include <fstream>
#include <math.h>
#include "SA_MP.h"
#include <vector>

int main(int argc, char * argv[])
{
    std::string filename;
    float coolingRate;
    int temperature;
    std::string fname;
    int workers;
    if (argc == 3)
    {
        fname = argv[1];
        workers = atoi(argv[2]);
    }
    // char filename[] = "data/berlin52.txt";
    else{
        workers = 5;
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


    SA sa = SA(matrix, size,workers );
   
    std::cout<<"Running with OPENMP"<<std::endl;
    sa.parallelApply();

    free(matrix);

      




    return 0;
}
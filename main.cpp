#include <iostream>
#include <fstream>
int main(int argc, char * argv[])
{
    std::fstream myfile("data/berlin52.txt", std::ios_base::in);

    int size;
    int id,x,y;
    myfile >> size;
    int **matrix = (int **)malloc(size * sizeof(int *));
    for (size_t i = 0; i < size; i++)
    {
        matrix[i] = (int *)malloc(size * sizeof(int));
    }
    for (size_t i = 0; i < size; i++)
    {
        myfile >> id >> x >>y;
        std::cout <<id<<" "<< x << " " << y << std::endl;
    }
    
    

    return 0;
}
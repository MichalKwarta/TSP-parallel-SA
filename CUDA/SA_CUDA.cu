#include "SA.h"
#include <vector>
#include <algorithm>
#include <time.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <omp.h>

void SA::apply()
{
	srand(time(NULL));
	std::vector<int> current = greedy();
	float currentCost = costFunction(current);

	std::vector<int> next(current);
	std::vector<int> best(current);

	int firstToSwap;
	int secondToSwap;
	double temperature = initialTemperature;
	float nextCost;
	float bestCost = currentCost;

	for (temperature = initialTemperature; temperature >= TEMP_LIMIT; temperature *= coolingRate)
	{

		for (int i = 0; i < STEPS; i++)

		{
			next = current;
			firstToSwap = rand() % size;

			do
			{

				secondToSwap = rand() % size;
			} while (firstToSwap == secondToSwap);

			// std::swap(next[firstToSwap], next[secondToSwap]);
			std::reverse(next.begin() + firstToSwap, next.begin() + secondToSwap);
			nextCost = costFunction(next);

			double difference = currentCost - nextCost;

			if (currentCost >= nextCost)
			{
				current = next;
				currentCost = nextCost;

				if (nextCost < bestCost)
				{
					bestCost = nextCost;
				}
			}
			else
			{

				if (exp((currentCost - nextCost) / temperature) > (float)rand() / RAND_MAX)
				{
					current = next;
					currentCost = nextCost;
					// break;
				}
			}
		}
	}

	std::cout << bestCost << std::endl;

	std::cout << std::endl;
}

void SA::parallelApply()
{
	srand(time(NULL));
	std::vector<int> current = greedy();
	float currentCost = costFunction(current);
	omp_set_num_threads(WORKERS);
	std::vector<int> next(current);
	std::vector<int> best(current);
	std::vector<std::vector<int>> workersPaths(WORKERS, current);
	std::vector<float> workersCosts(WORKERS,currentCost);
	std::vector<float> workersBest(WORKERS,currentCost);


	double temperature = initialTemperature;
	float nextCost;



	float bestCost = currentCost;
	int firstToSwap,secondToSwap;
	#pragma omp parallel private(current,currentCost,firstToSwap, secondToSwap, next, nextCost) firstprivate(matrix)
		{
	for (temperature = initialTemperature; temperature >= TEMP_LIMIT; temperature *= coolingRate)
	{
		
			current = workersPaths[omp_get_thread_num()];
			currentCost = workersCosts[omp_get_thread_num()];
			bestCost = workersBest[omp_get_thread_num()];
			#pragma omp for schedule(dynamic) nowait
			for (int i = 0; i < STEPS; i++)
			{
				next = current;
				firstToSwap = rand() % size;

				do
				{
					secondToSwap = rand() % size;
				} while (firstToSwap == secondToSwap);

				// std::swap(next[firstToSwap], next[secondToSwap]);
				std::reverse(next.begin() + firstToSwap, next.begin() + secondToSwap);

				nextCost = costFunction(next);

				double difference = currentCost - nextCost;

				if (currentCost >= nextCost)
				{

					current = next;
					currentCost = nextCost;
					if (nextCost <= bestCost)
					{
						bestCost = nextCost;
					}
				}
				else
				{

					if (exp((currentCost - nextCost) / temperature) > (float)rand() / RAND_MAX)
					{
						current = next;
						currentCost = nextCost;
					}
				}
			}

			workersPaths[omp_get_thread_num()] = current;
			workersCosts[omp_get_thread_num()] = currentCost;
			workersBest[omp_get_thread_num()] = bestCost;
		}


	}

	std::cout <<*std::min_element(workersBest.begin(),workersBest.end()) << std::endl;
}

float SA::costFunction(std::vector<int> path)
{
	float cost = 0;
	for (int i = 0; i < path.size() - 1; ++i)
	{
		cost += matrix[path[i]][path[i + 1]];
	}
	cost += matrix[path[size - 1]][path[0]];

	return cost;
}

std::vector<int> SA::greedy()
{
	std::vector<int> path = {0};
	std::vector<int> nodesToVisit;
	for (int i = 1; i < this->size; i++)
	{

		nodesToVisit.push_back(i);
	}
	while (nodesToVisit.size() > 0)
	{
		int min = nodesToVisit[0];
		for (int i = 1; i < nodesToVisit.size(); i++)
		{
			if (matrix[path[path.size() - 1]][nodesToVisit[i]] < matrix[path[path.size() - 1]][min])
			{
				min = nodesToVisit[i];
			}
		}
		path.push_back(min);
		nodesToVisit.erase(std::remove(nodesToVisit.begin(), nodesToVisit.end(), min), nodesToVisit.end());
	}

	return path;
}

SA::SA(float **matrixarg, int sizearg,int workers)
{
	matrix = matrixarg;
	size = sizearg;

	WORKERS = workers;
}

SA::~SA()
{
}

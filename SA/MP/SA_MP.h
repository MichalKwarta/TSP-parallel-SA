#include <vector>

class SA
{
	std::vector<int> solution;
	double initialTemperature = 1000;
	double coolingRate = 0.99;
	float** matrix;
	int size;
	float TEMP_LIMIT = 0.001;
	int STEPS = 400*12;
	int WORKERS;

public:

	float parallelApply();
	float costFunction(std::vector<int> path);
	SA(float** matrixarg, int sizearg,int workers);
	std::vector<int> greedy();
	~SA();
};

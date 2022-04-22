#include <vector>

class SA
{
	std::vector<int> solution;
	double initialTemperature = 0;
	double coolingRate = 0;
	float** matrix;
	int size;
	float TEMP_LIMIT = 0.001;
	int STEPS = 400*12;

public:

	void apply();
	void parallelApply();
	float costFunction(std::vector<int> path);
	SA(float** matrixarg, int sizearg, int temp,double rate);
	double calculateTemperature();
	std::vector<int> greedy();
	~SA();
};

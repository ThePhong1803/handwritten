#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <image.h>
//#include <neuralnetwork.h>

#define MAX 100
float addOne(float x){
	return x + 1;
}
int main() {
	srand(time(NULL));
	Image img("dataset/training_dataset/00001.bmp", 1);
	img.hexdump();
	return 0;
}
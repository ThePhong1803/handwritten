#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <fstream>
#include <image.h>
#include <data.h>
#include <neuralnetwork.h>

int minus1(int x){
	return x - 1;
}

int add1(int x){
	return x + 1;
}

std::vector<std::pair<int (*)(int), int (*)(int)>> actFun;

int main() {
	// use for testing library purpose
	actFun.push_back(std::pair<int (*)(int), int (*)(int)> (add1, minus1));
	Eigen::RowVectorXf vec(3);
	vec << 1, 2, 3;
	std::cout << vec.unaryExpr(std::ptr_fun(actFun[0].first)) << std::endl;
	std::cout << vec.unaryExpr(std::ptr_fun(actFun[0].second)) << std::endl;
	std::cout << vec << std::endl;
	return 0;
}
#include "image.h"
#include <eigen3/Eigen/Eigen>
#pragma once

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;


class Data {
	public:
	int number;
	int id;
	std::vector<float> pixels;
	Data() {}
	Data(int _number, int _id, std::vector<float> _pixels) : number(_number), id(_id), pixels(_pixels) {}
	
	void getPixels(RowVector * temp) {
		for(uint32_t i = 0; i < pixels.size(); i++){
			(*temp)[i] = pixels[i];
		}
	}
};
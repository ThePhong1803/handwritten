// NeuralNetwork.h
#pragma once
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <iomanip>

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef uint32_t uint;

// neural network implementation class!
class NeuralNetwork {
public:
	// constructor
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));
	~NeuralNetwork();

	// function for forward propagation of data
	void propagateForward(RowVector& input);

	// function for backward propagation of errors made by neurons
	void propagateBackward(RowVector& output);

	// function to calculate errors made by neurons in each layer
	void calcErrors(RowVector& output);

	// function to update the weights of connections
	void updateWeightsAndBiases();

	// function to train the neural network give an array of data points
	void train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int batchSize);

	// storage objects for working of neural network
	/*
		use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
		Class as soon as it is pushed back! when we use pointers it can't do that, besides
		it also makes our neural network class less heavy!! It would be nice if you can use
		smart pointers instead of usual ones like this
		*/
	std::vector<RowVector*> neuronLayers; // stores the different layers of out network
	std::vector<RowVector*> cachesLayers; // stores the unactivated (activation fn not yet applied) values of layers
	std::vector<RowVector*> layersError;  // stores the error contribution of each neurons
	std::vector<Matrix*> 	weights; 	  // the connection weights itself
	std::vector<Matrix*> 	dweights;	  // store the average change of weights matrix, and will be apply to the weights matrix after a batch training
	std::vector<RowVector*> bias;		  // store the bias value for all layers except input layers
	std::vector<uint> 		topology;	  // network topology configuraton vector
	Scalar learningRate;


	void printNetwork(){
		for(uint i = 0; i < neuronLayers.size(); i++){
			std::cout << "Neuron layer: " << i << std::endl;
			std::cout << *neuronLayers[i] << std::endl;
		}

		for(uint i = 0; i < cachesLayers.size(); i++){
			std::cout << "Caches layer: " << i << std::endl;
			std::cout << *cachesLayers[i] << std::endl;
		}

		for(uint i = 0; i < layersError.size(); i++){
			std::cout << "Layer errors: " << i << std::endl;
			std::cout << *layersError[i] << std::endl;
		}

		for(uint i = 0; i < weights.size(); i++){
			std::cout << "Weight layer: " << i << std::endl;
			std::cout << *weights[i] << std::endl;
			std::cout << "dWeight layer: " << i << std::endl;
			std::cout << *dweights[i] << std::endl;
		}
		for(uint i = 0; i < bias.size(); i++){
			std::cout << "Bias layer: " << i << std::endl;
			std::cout << *bias[i] << std::endl;
		}
	}
};
#include <neuralnetwork.h>

// activation function for network
Scalar _debug_activationFunction(Scalar x)
{
	return x;
}

Scalar _debug_activationFunctionDerivative(Scalar x)
{
	return 1;
}

Scalar activationFunction(Scalar x)
{
	// sigmoid
	return 1.0f/(1 + exp(-x));
}

Scalar _activationFunction(Scalar x)
{
	// ReLU
	return (x > 0) ?  x : 0;
}

Scalar activationFunctionDerivative(Scalar x)
{
	// sigmiod derivative
	return activationFunction(x) * (1 - activationFunction(x));
}

Scalar _activationFunctionDerivative(Scalar x)
{
	// ReLU derivative
	return (x > 0) ?  1 : 0;
}

// constructor of neural network class
/*
	@param topology: stl vector input for network topology
	@param learningRate: define how fast the network learn
	*/
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
	this -> topology 		= topology;
	this -> learningRate 	= learningRate;
	// init network
	for(size_t i = 0; i < this -> topology.size(); i++){
		
		/* create neuron layers and caches layer*/
		this -> neuronLayers.push_back(new RowVector(this -> topology[i]));
		this -> neuronLayers.back() -> setZero();
		/* create and init weights matrix */
		if (i > 0) {
			// add new vector to the container
			this -> weights.push_back		(new Matrix		(topology[i - 1], topology[i]	));
			this -> dweights.push_back		(new Matrix		(topology[i - 1], topology[i]	));
			this -> cachesLayers.push_back	(new RowVector	(this -> topology[i]			));	// i.e no caches layers for input	
			this -> layersError.push_back	(new RowVector	(this -> topology[i]			));
			this -> bias.push_back			(new RowVector	(this -> topology[i]			));	// i.e no bias for input layers
			
			// initialization
			this -> cachesLayers.back() -> setZero();
			this -> layersError.back()	-> setZero();
			this -> bias.back()			-> setRandom();
			this -> weights.back() 		-> setRandom();
			this -> dweights.back()		-> setZero();
		}
	}
};

// deconstructor of neural network class
NeuralNetwork::~NeuralNetwork()
{
	// iterate each layer, deallocate used memory
	try{
		for(size_t i = 0; i < this -> topology.size(); i++){
			delete this -> neuronLayers.back();
			this -> neuronLayers.pop_back();
			
			if(i > 0){
				delete this -> cachesLayers.back();
				delete this -> layersError.back();
				delete this -> weights.back();
				delete this -> dweights.back();
				delete this -> bias.back();

				this -> cachesLayers.pop_back();
				this -> layersError.pop_back();
				this -> weights.pop_back();
				this -> dweights.pop_back();
				this -> bias.pop_back();
			}
		}
	} catch(std::exception &e){
		std::cout << e.what() << std::endl;
		exit(-1);
	}
};
void NeuralNetwork::propagateForward(RowVector& input)
{
	// set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
	(*neuronLayers.front()) = input;
	for (uint i = 0; i < topology.size() - 1; i++) {
		// calculate the next layer caches valued
		(*cachesLayers[i]) = (*neuronLayers[i]) * (*weights[i]) + (*bias[i]);
		// propagate the data forward and then
		// apply the activation function to your network
		// unaryExpr applies the given function to all elements of CURRENT_LAYER
		(*neuronLayers[i + 1]) = cachesLayers[i] -> unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::propagateBackward(RowVector& output)
{
	// calc output error
	/*
		For output layers error calculation:
		@param: output is the desire output
		@param: (*neuronLayers[neuronLayers.size() - 1) is the activated output of the network or predicted output
		#param: (*cachesLayers(cachesLayers.size() - 1) is the unactivated output of the network or weighted value;
		*/
	RowVector errors = (output - (*neuronLayers[neuronLayers.size() - 1]));
	for (int i = layersError.size() - 1; i >= 0; i--) {
		// iterate throught each error vector and calculate error, and update weights and biases
		RowVector prevErrors = errors * weights[i] -> transpose();

		Matrix gradients = (errors.array() * (cachesLayers[i] -> unaryExpr(std::ptr_fun(activationFunctionDerivative))).array());

		(*bias[i]) += learningRate * gradients;
		(*dweights[i]) = (neuronLayers[i] -> transpose()) * gradients;
		(*weights[i]) += learningRate * (*dweights[i]);
		errors = prevErrors;
    }
}

void NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data, int batchSize)
{
	float MSE = 0;
	for (int i = 0; i < batchSize; i++) {
		int index = rand() % batchSize;
		//this -> printNetwork();
		// std::cout << "___________________________________________" << std::endl;
        // std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
        propagateForward(*input_data[index]);
        // std::cout << "Expected output is : " << *output_data[index] << std::endl;
        propagateBackward(*output_data[index]);
		// std::cout << "Actual output is : " << *neuronLayers.back() << std::endl;
        // std::cout << "MSE : " << std::sqrt((*neuronLayers.back() - *output_data[index]).dot((*neuronLayers.back() - *output_data[index])) / neuronLayers.back()->size()) << std::endl;
		// std::cout << "___________________________________________" << std::endl;
		// this -> printNetwork();
		MSE += std::sqrt((*neuronLayers.back() - *output_data[index]).dot((*neuronLayers.back() - *output_data[index])) / neuronLayers.back()->size());
    }
	MSE = MSE / batchSize;
	// Validate model accuracy
	std::cout << "MSE: " << MSE << std::endl;
}
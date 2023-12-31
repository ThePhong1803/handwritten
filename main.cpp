#include <iostream>
#include <iomanip>
#include <time.h>
#include <sstream>
#include <functional>
#include "data.h"
#include "neuralnetwork.h"
#include "image.h"

void loadBatches(   std::vector<RowVector*> &input_data,             // input data
                    std::vector<RowVector*> &output_data,            // output data
                    std::vector<RowVector>  &targetOutputs,          // target output for mapping
                    std::vector<int*>       &labelVec,               // label vector for mapping
                    int batch                                        // batch size of the training
                )
{
    /* - Loading image data into data container */
	char buff[128];
	for(int n = 0; n < batch; n++){
		try{
			int imageNum = rand() % (labelVec.size()) + 1;
			sprintf(buff, "dataset/training_dataset/%05d.bmp", imageNum);
			Image img(buff, imageNum);
			Data obj(imageNum, n, img.getPixelArray());
			obj.getPixels(input_data[n]); 
			output_data[n] = &targetOutputs[*labelVec[imageNum - 1]];
		} catch (std::exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
}

void loadTestData(  std::vector<RowVector*> &input_data,            // input data
                    std::vector<RowVector*> &output_data,            // output data
                    std::vector<RowVector>  &targetOutputs,          // target output for mapping
                    std::vector<int*>       &labelVec,               // label vector for mapping
                    int batch                                        // batch size of the training
                )
{
    /* - Loading image data into data container */
	char buff[128];
	for(int n = 0; n < batch; n++){
		try{
			int imageNum = rand() % (labelVec.size()) + 1;
			sprintf(buff, "dataset/test_dataset/%05d.bmp", imageNum);
			Image img(buff, imageNum);
			Data obj(imageNum, n, img.getPixelArray());
			obj.getPixels(input_data[n]); 
			output_data[n] = &targetOutputs[*labelVec[imageNum - 1]];
		} catch (std::exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
}

int outputToLabelIdx(RowVector * out){
	// Sigmoid or ReLU always produce output greater than 0
    float max = 0;
	int idx = 0;
	for(int i = 0; i < out -> cols(); i++){
		if(out -> coeff(i) > max){
			max = out -> coeff(i);
			idx = i;
		}
	}
	return idx;
}

int main(int argc, char ** argv)
{
	// Reading input argument
	srand(time(NULL));
	if(argc < 4) {
		std::cout << "Using: out --learning-rate --epoch --batch" << std::endl;
		exit(-1);
	}
	float learnRate = atof(argv[1]);
	int epoch		= atoi(argv[2]);
	int batch		= atoi(argv[3]);
	std::cout << "Learning rate: " << learnRate << std::endl;
	std::cout << "Epoch: " << epoch << std::endl;
	std::cout << "Batch size: " << batch << std::endl;

	// loading dataset
	/* - Fist is setup input data containter, then the expected output data container */
	std::vector<RowVector*> input_data(batch);
	std::vector<RowVector*> output_data(batch);
	std::vector<RowVector> 	targetOutputs(10, RowVector(10));
	
	for(size_t i = 0; i < input_data.size(); i++){
		input_data[i] = new RowVector(784);
	}
	
    /* - Create labeled array for mapping */
    std::vector<int*> labelVec;
    std::ifstream labelFile;
    labelFile.open("dataset/training_dataset/label.txt");
    if(!labelFile.is_open()) {
        std::cout << "File not found" << std::endl;
        exit(-1);
    }

    int label;
    while(labelFile >> label){
        labelVec.push_back(new int(label));
    }
	labelFile.close();   

	std::vector<int*> validateLabels;
    std::ifstream validateLabelsFile;
    validateLabelsFile.open("dataset/test_dataset/00000-labels.txt");
    if(!validateLabelsFile.is_open()) {
        std::cout << "File not found" << std::endl;
        exit(-1);
    }

    int testLabel;
    while(validateLabelsFile >> testLabel){
        validateLabels.push_back(new int(testLabel));
    }
	validateLabelsFile.close(); 
	/* - Mapping the expected output data to the output data container */
	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++){
			if(i == j){
				targetOutputs[i][j] = 1.0f;
			}
			else {
				targetOutputs[i][j] = 0.0f;
			}
		}
	}
	/* Prepare ativation fucntion vecot and topology vector*/
	std::vector<uint32_t> 			topology = {784, 16, 16, 10};						// for 28 x 28 pixel bit map images
	std::vector<std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>>	actFunPtr = {
		std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>(ReLU	 , dReLU),				// hidden layer 1
		std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>(ReLU	 , dReLU   ),			// hidden layer 2
 		std::pair<Scalar (*)(Scalar), Scalar (*)(Scalar)>(Sigmoid, dSigmoid),			// output layer
	};
	/* - Training with loaded data */
	srand(time(NULL));
	NeuralNetwork NN(topology, actFunPtr, learnRate);
	std::ofstream log("./log/RMSE.txt");
	log << "RMSE" << " " << "ACC" << " " << "MSR_Validate" << '\n';
	for(int i = 0; i < epoch; i++){
		srand(time(NULL));
        loadBatches(input_data, output_data, targetOutputs, labelVec, batch);
		// return <MSE , ACC> in the training period
		std::pair<float, float> resTrain = NN.train(input_data, output_data, outputToLabelIdx, batch);
		loadTestData(input_data, output_data, targetOutputs, validateLabels, batch);

		// return <ACC , CON> in the valiadate period
		std::pair<float, float> resValid = NN.validateTrain(input_data, output_data, batch, 100, outputToLabelIdx);

		// log the training MSE and validated ACC
		log << resTrain.first << " " << resValid.first << " " << resValid.second << '\n';
		std::cout << "\rEpoch : " << i + 1 << " MSE: " << resTrain.first;
	}
	std::cout << std::endl;
	log.close();

	/* - Calculate model accuracy */
	loadTestData(input_data, output_data, targetOutputs, validateLabels, batch);
	std::pair<float, float> validate = NN.validate(input_data, output_data, batch, 1000, outputToLabelIdx);
	std::cout << "Model average accuracy  : " << validate.first << std::endl;
	std::cout << "Model average confident : " << validate.second << std::endl;
	std::cout << "===========================================================" << std::endl;

	std::string cmd = "";
	while(cmd != "exit"){
		// get test sample
		Image img("test_data/test.bmp", 11);
        img.setInvert(false);
		Data obj(0, 0, img.getPixelArray());
		RowVector testData(img.getHeight() * img.getWidth());
		obj.getPixels(&testData);
		NN.propagateForward(testData);
		img.testing();

		int num = outputToLabelIdx(NN.neuronLayers.back());
		std::cout << "Predicted output: " << num << " Confident: " << std::fixed << std::setprecision(2) << NN.neuronLayers.back() -> coeff(num) << std::endl;
		std::cout << "Output vector: " << *NN.neuronLayers.back() << std::endl;
		getline(std::cin, cmd);
	}

	// Deallocate memory
	for(size_t i = 0; i < input_data.size()    ; i++) delete input_data[i];
    for(size_t i = 0; i < labelVec.size()      ; i++) delete labelVec[i];
	for(size_t i = 0; i < validateLabels.size(); i++) delete validateLabels[i];
	return 0;
}
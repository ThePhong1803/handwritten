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
                    int batch                                       // batch size of the training
                )
{
    /* - Loading image data into data container */
	char buff[128];
	for(int n = 0; n < batch; n++){
        int imageNum = rand() % (labelVec.size()) + 1;
        sprintf(buff, "dataset/training_dataset/%05d.bmp", imageNum);
        Image img(buff, imageNum);
        Data obj(imageNum, n, img.getPixelArray());
        obj.getPixels(input_data[n]); 
        output_data[n] = &targetOutputs[*labelVec[imageNum - 1]];
	}
}

int main(int argc, char ** argv)
{
	// Reading input argument
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
	std::vector<RowVector> targetOutputs(10, RowVector(10));
	
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
	
	/* - Training with loaded data */
	NeuralNetwork NN({784, 16, 16, 10}, learnRate);
	for(int i = 0; i < epoch; i++){
        loadBatches(input_data, output_data, targetOutputs, labelVec, batch);
		NN.train(input_data, output_data, batch);
		std::cout << "Epoch : " << i << std::endl;
	}
	
	std::string cmd = "";
	while(cmd != "exit"){
		// get test sample
		Image img("test/test.bmp", 11);
        img.setInvert(true);
		Data obj(0, 0, img.getPixelArray());
		RowVector testData(img.getHeight() * img.getWidth());
		obj.getPixels(&testData);
		NN.propagateForward(testData);
		img.testing();
		std::cout << "Predicted output: " << std::fixed << std::setprecision(2) << *NN.neuronLayers.back() << std::endl;
		getline(std::cin, cmd);
	}

	// Deallocate memory
	for(size_t i = 0; i < input_data.size(); i++) delete input_data[i];
    for(size_t i = 0; i < labelVec.size(); i++) delete labelVec[i];
	return 0;
}
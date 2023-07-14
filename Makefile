all:
	g++ -Wall -I./inc src/neuralnetwork.cpp src/image.cpp main.cpp -o out

clean:
	del out.exe

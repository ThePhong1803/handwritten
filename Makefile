all:
	g++ -Wall -I./inc src/neuralnetwork.cpp src/image.cpp main.cpp -o out

clean:
	del *.exe

test:
	g++ -Wall -I./inc src/neuralnetwork.cpp src/image.cpp test.cpp -o test
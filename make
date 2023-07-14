all:
	g++ -Wall -I./inc src/neuralnetwork.cpp test.cpp -o out

clean:
	del out.exe

CXX=g++
CXXFLAGS=-Wall -Wextra -pedantic -Wconversion -std=c++98 -DNDEBUG -O3

all: main

main: *.cpp
	$(CXX) $(CXXFLAGS) *.cpp -o main

clean:
	rm -fr *.o main

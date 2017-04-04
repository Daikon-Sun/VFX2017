CXX=g++
CXXFLAGS=-std=c++14 -O3
LIBS=`pkg-config --libs opencv`
OBJ=$(addprefix obj/, $(patsubst %.hpp, %.o, $(wildcard *.hpp)))

.PHONY: clean

main: $(OBJ)
	$(CXX) $(CXXFLAGS) main.cpp -o $@ $^ $(LIBS)

obj/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

run:
	./main test_pics2 7 4 1

clean:
	@rm obj/*.o

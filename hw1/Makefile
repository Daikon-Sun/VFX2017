CXX=g++
CXXFLAGS=-std=c++14 -O3
LIBS=`pkg-config --libs opencv`
OBJ=$(addprefix obj/, $(patsubst src/%.hpp, %.o, $(wildcard src/*.hpp)))

.PHONY: clean

main: $(OBJ)
	$(CXX) $(CXXFLAGS) src/main.cpp -o $@ $^ $(LIBS)

obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

run:
	./main test_pics2 7 4 100 result/out.hdr result/out.jpg

clean:
	@rm obj/*.o

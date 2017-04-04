CXX=g++
CXXFLAGS=-std=c++14 -O3
LIBS=`pkg-config --libs opencv`
OBJ=$(addprefix obj/, $(patsubst %.hpp, %.o, $(wildcard *.hpp)))

HDR: $(OBJ)
	$(CXX) $(CXXFLAGS) hdr.cpp -o $@ $^ $(LIBS)

obj/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

clean:
	@rm obj/*.o

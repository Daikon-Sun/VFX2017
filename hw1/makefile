CXX=g++
CXXFLAGS=-std=c++14 -O3
LIBS=`pkg-config --libs opencv`
OBJ=$(addprefix obj/, $(patsubst %.hpp, %.o, $(wildcard *.hpp)))

.PHONY: clean

HDR: $(OBJ)
	$(CXX) $(CXXFLAGS) hdr.cpp -o $@ $^ $(LIBS)

obj/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBS)

run:
	./HDR test_pics1 8 4 1

clean:
	@rm obj/*.o

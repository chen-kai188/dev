CC=g++
CXXFLAGS:=-Wall -ggdb -Wextra -std=c++11 `pkg-config --cflags opencv4`
LDFLAGS:=`pkg-config --libs opencv4`
target=APP
SRC=$(wildcard *.cpp)
OBJ=$(patsubst %.cpp,%.o,$(SRC))

$(target):$(OBJ)
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

$(OBJ):%.o:%.cpp
	$(CC) -c $< -o $@ $(CXXFLAGS)

.PHONY:clean
clean:
	rm -rf $(OBJ) $(target)

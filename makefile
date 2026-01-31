CC=g++
CFLAGS= -Wall
target=APP
SRC=$(wildcard *.cpp)
OBJ=$(patsubst %.cpp,%.o,$(SRC))

$(target):$(OBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(OBJ):%.o:%.cpp
	$(CC) -c $< -o $@ $(CFLAGS)

.PHONY:clean
clean:
	rm -rf $(OBJ) $(target)

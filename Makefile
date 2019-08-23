SRC := src/main.cc src/cudnnOpDriver.cc
TARGET := op_driver

CPPFLAGS = -std c++11 -O2 -DDEBUG

op_driver: ${SRC}
	nvcc ${CPPFLAGS} -o $@ $^ -lcudnn

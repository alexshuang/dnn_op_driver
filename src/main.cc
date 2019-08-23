#include <cudnn.h>
#include <iostream>
#include <random>
#include "Backend.hpp"
#include "cudnnOpDriver.hpp"

#define CONV_HP_INIT {.kernel_size=3, .padding=1, .stride=1, .dilation=1,\
	.groups=1, .bias=1}

/*
// https://github/carlushuang/miopen_cudnn_ops/src/executable/op_driver.cc
static void rand_float(float * vec, int len){
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for(int i=0;i<len;i++){
        vec[i] = dist(mt);
    }
}
*/

int main(void)
{
	conv_hp_t params = CONV_HP_INIT;
	int n = 128, c = 3, h = 28, w = 28, nh = 100;

	cudnnOpDriver *op_drv = new cudnnOpDriver();

	op_drv->createConvModule(n, c, nh, h, w, &params);

	return 0;
}

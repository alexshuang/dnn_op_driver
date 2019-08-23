#ifndef __CUDNNOPDRIVER_HPP__
#define __CUDNNOPDRIVER_HPP__

#include "Backend.hpp"
#include <cudnn.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#define checkCUDNN(f) do { \
	cudnnStatus_t res = (f); \
	if (res != CUDNN_STATUS_SUCCESS) { \
		std::cerr << __FILE__ << ": Error on " << __LINE__ << " line, " << \
		cudnnGetErrorString(res) << std::endl; \
		exit(EXIT_FAILURE); \
	} \
} while (0)

inline cudnnTensorFormat_t cudnn_tensor_format(tFormat_t layout)
{
	cudnnTensorFormat_t res = CUDNN_TENSOR_NCHW;

	switch (layout) {
		case TENSOR_NCHW:
			res = CUDNN_TENSOR_NCHW;
			break;
		case TENSOR_NHWC:
			res = CUDNN_TENSOR_NHWC;
			break;
		default:
			break;
	}

	return res;
}

inline cudnnDataType_t cudnn_datatype(tDataType_t dtype)
{
	cudnnDataType_t res = CUDNN_DATA_FLOAT;

	switch (dtype) {
		case DATA_FLOAT32:
			res = CUDNN_DATA_FLOAT;
			break;
		case DATA_INT32:
			res = CUDNN_DATA_INT32;
			break;
		case DATA_HALF:
			res = CUDNN_DATA_HALF;
			break;
		default:
			break;
	}

	return res;
}

//class cudnnOpDriver: public Backend {
class cudnnOpDriver {
	public:
		cudnnOpDriver() {
			checkCUDNN(cudnnCreate(&this->handle));
		}

		~cudnnOpDriver() {
			checkCUDNN(cudnnDestroy(this->handle));
		}

		void createConvModule(int n, int in_c, int out_c, int in_h, int in_w,
				conv_hp_t *params, tFormat_t tensor_format=TENSOR_NCHW,
				tDataType_t dtype=DATA_FLOAT32);
		/*
		void forward(void *module);
		void backward(void *module);
		*/

		/*
		void createConvModule(opTensor *input, opTensor *output,
				conv_hp_t params);
		void createReLUModule(opTensor *input, opTensor *output,
				conv_relu_t params);
		void createPoolingModule(opTensor *input, opTensor *output,
				conv_pooling_t params);
		void createLinearModule(opTensor *input, opTensor *output);
		void createSoftmaxModule(opTensor *input, opTensor *output);
		void destroyConvnDModule(void);
		void destroyReLUModule(void);
		void destroyPoolingModule(void);
		void destroyLinearModule(void);
		void destroySoftmaxModule(void);
		*/

	private:
		cudnnHandle_t handle;
};

#endif

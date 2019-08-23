#ifndef __BACKEND_HPP__
#define __BACKEND_HPP__

//#include "opTensor.hpp"

enum tDataType_t {
	DATA_FLOAT32,
	DATA_INT32,
	DATA_HALF,
};

enum tFormat_t {
	TENSOR_NCHW,
	TENSOR_NHWC,
};

typedef struct conv_hyperparameters {
	int kernel_size;
	int padding;
	int stride;
	int dilation;
	int groups;
	int bias;
} conv_hp_t;

typedef struct relu_hyperparameters {
	int in_place;
} conv_relu_t;

enum pooling_mode {
	MAX_POOLING,
	AVG_POOLING,
};

typedef struct pooling_hyperparameters {
	int pooling_mode;
	int *kernel_size;
	int *stride;
} conv_pooling_t;

class Backend {
	public:
		Backend() {}
		void createConvModule(int n, int in_c, int out_c, int in_h, int in_w,
				conv_hp_t *params, tFormat_t tensor_format=TENSOR_NCHW,
				tDataType_t dtype=DATA_FLOAT32);
		/*
		virtual void forward(void *module) {}
		virtual void backward(void *module) {}

		virtual void createReLUModule(opTensor *input, opTensor *output,
				conv_relu_t params);
		virtual void createPoolingModule(opTensor *input, opTensor *output,
				conv_pooling_t params);
		virtual void createLinearModule(opTensor *input, opTensor *output);
		virtual void createSoftmaxModule(opTensor *input, opTensor *output);
		virtual void destroyConvnDModule(void);
		virtual void destroyReLUModule(void);
		virtual void destroyPoolingModule(void);
		virtual void destroyLinearModule(void);
		virtual void destroySoftmaxModule(void);
		*/
};

#endif

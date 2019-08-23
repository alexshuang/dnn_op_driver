#include "cudnnOpDriver.hpp"
#include "debug.hpp"

void cudnnOpDriver::createConvModule(int n, int in_c, int out_c, int in_h,
		int in_w, conv_hp_t *params, tFormat_t tensor_format, tDataType_t dtype)
{
	cudnnTensorDescriptor_t input_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_desc,
				cudnn_tensor_format(tensor_format),
				cudnn_datatype(dtype),
				n, in_c, in_h, in_w));
	
	cudnnFilterDescriptor_t filter_desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc,
				cudnn_datatype(dtype),
				CUDNN_TENSOR_NCHW,
				out_c, in_c, params->kernel_size, params->kernel_size));

	cudnnConvolutionDescriptor_t conv_desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
				params->padding, params->padding, params->stride, params->stride,
				params->dilation, params->dilation, CUDNN_CROSS_CORRELATION,
				CUDNN_DATA_FLOAT));

	int out_h, out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc,
				input_desc, filter_desc, &n, &out_c, &out_h, &out_w));

	debug("%d, %d, %d, %d\n", n, out_c, out_h, out_w);

	cudnnTensorDescriptor_t output_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,
				cudnn_tensor_format(tensor_format),
				cudnn_datatype(dtype),
				n, out_c, out_h, out_w));

	debug("out_h: %d, out_w: %d\n", out_h, out_w);
	
	
	/*
	int res_algo_count = 0;
	cudnnConvolutionFwdAlgoPerf_t conv_algos[8];
	checkCUDNN(cudnnFindConvolutionForwardAlgorithm(this->handle,
				input_desc, filter_desc, conv_desc, output_desc,
				8, &res_algo_count, conv_algos));

	printf("res_algo_count: %d\n", res_algo_count);
	*/
}

#ifndef __OPTENSOR_HPP__
#define __OPTENSOR_HPP__

#include <cassert>
#include <cstdlib>

#define MAX_DIMS	4

enum tDataType_t {
	DATA_FLOAT32,
	DATA_INT32,
	DATA_HALF,
};

#define DATA_FLOAT32_SIZE 	4
#define DATA_INT32_SIZE 	4
#define DATA_HALF_SIZE 		2

static inline int sizeof_dtype(tDataType_t t) {
	int res = 0;
	switch (t) {
		case DATA_FLOAT32:
		case DATA_INT32:
			res = 4;
			break;
		case DATA_HALF:
			res = 2;
			break;
		default:
			break;
	}

	return res;
}

enum tFormat_t {
	TENSOR_NCHW,
	TENSOR_NHWC,
};

class opTensor {
	public:
		/* shape order must be same as format's */
		opTensor(int dims, int *shape, tFormat_t format, tDataType_t dtype) {
			this->dims = dims;
			this->dtype = dtype;
			this->format = format;
			int bytes = 1;
			for (int i = 0; i < dims; i++) {
				this->shape[i] = shape[i];
				bytes *= shape[i];
			}
			bytes *= sizeof_dtype(dtype);
			this->data = malloc(bytes);
			assert(this->data != NULL);
		}

		~opTensor() {
			free(this->data);
		}

		tDataType_t dtype;
		tFormat_t format;
		int dims;
		int shape[MAX_DIMS];
		void *data;
};

#endif

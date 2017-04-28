#include <THC/THC.h>
#include "THCUNN.h"
#include "common.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif



// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

struct correct_disp_functor
{
  const float mu, stdval, threshold1, threshold2;
  error_disp_functor(float mu_, float stdval_, float threshold1_, float threshold2_)
    : mu(mu_), stdval(stdval_), threshold1(threshold1_), threshold2(threshold2_) {}

  __host__ __device__ float operator()(const float &est, const float &gt) const
  {
	 float disp_gt  = mu + stdval * gt;
	 float disp_est = mu + stdval * est;
	 float output_val = 0.0f;
	 if (disp_gt > 0.0f) {
    	float diff_abs   = fabsf(disp_gt - disp_est);
	 	float diff_ratio = diff_abs / disp_gt;
		if (diff_abs > threshold1 && diff_ratio > threshold2) {
			output_val = 1.0f;		
		}
	 }
	 return output_val;
  }
};

//extern "C"
void comp_disp_error_map_cuda(
	THCudaTensor *estimation, 
	THCudaTensor *target, 
	THCudaTensor *output, 
	const float threshold1, const float threshold2,
	const float mu, const float stdval)
{
  THCUNN_assertSameGPU(state, 3, estimation, target, output);
  THArgCheck(
    THCudaTensor_nElement(state, estimation) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );
  long size = THCudaTensor_nElement(state, estimation);
  estimation = THCudaTensor_newContiguous(state, estimation);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, output, estimation);

  thrust::device_ptr<float> estimation_data(THCudaTensor_data(state, estimation));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> output_data(THCudaTensor_data(state, output));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    estimation_data, estimation_data+size, target_data, output_data,
    error_disp_functor(mu, stdval, threshold1, threshold2)
  );

  THCudaTensor_free(state, estimation);
  THCudaTensor_free(state, target);
}

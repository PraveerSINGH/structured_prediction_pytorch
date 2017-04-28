void comp_disp_error_map_cuda(
	THCudaTensor *estimation, 
	THCudaTensor *target, 
	THCudaTensor *output, 
	const float threshold1, const float threshold2,
	const float mu, const float stdval);





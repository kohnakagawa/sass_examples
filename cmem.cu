#include <iostream>
#include <algorithm>
#include <helper_cuda.h>

__global__ void kernel1(double* a,
                        const double cf) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  a[tid] *= cf;
}

__constant__ double cf_c;

__global__ void kernel2(double* b) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  b[tid] *= cf_c;
}

__device__ __noinline__ double add_and_mul(const double a,
                                           const double b,
                                           const double c) {
  const double ret = (a + b) * c;
  return ret;
}

__global__ void kernel3(double* a) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto ret = add_and_mul(a[tid], a[tid], a[tid]);
  a[tid] = add_and_mul(ret, ret, ret);
}

__global__ void kernel4(double* a) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 100) {
    a[tid] = 10;
  } else {
    a[tid] = 0;
  }
}

__global__ void kernel5(int64_t* a) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  a[tid] += a[tid];
}

auto main(const int argc,
          const char* argv[]) -> int {
  const auto tb_size = 128;
  const auto gr_size = 1000;
  const auto ar_size = tb_size * gr_size;

  // allocate
  double* a_h = new double [ar_size];
  double* a_d = nullptr;
  checkCudaErrors(cudaMalloc((void**)&a_d,
                             ar_size * sizeof(double)));

  // init
  std::fill(a_h, a_h + ar_size, 2.0);

  const double cf = 2.0;
  checkCudaErrors(cudaMemcpyToSymbol(cf_c, &cf, sizeof(double)));

  // host -> device
  checkCudaErrors(cudaMemcpy(a_d, a_h, ar_size * sizeof(double),
                             cudaMemcpyHostToDevice));

  kernel1<<<gr_size, tb_size>>>(a_d, double(argc));
  checkCudaErrors(cudaDeviceSynchronize());

  kernel1<<<gr_size, tb_size>>>(a_d, 2.0);
  checkCudaErrors(cudaDeviceSynchronize());

  kernel2<<<gr_size, tb_size>>>(a_d);
  checkCudaErrors(cudaDeviceSynchronize());

  // device -> host
  checkCudaErrors(cudaMemcpy(a_h, a_d, ar_size * sizeof(double),
                             cudaMemcpyDeviceToHost));

  // free
  checkCudaErrors(cudaFree(a_d));
  delete [] a_h;
}

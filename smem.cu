#include <iostream>
#include <iomanip>
#include <algorithm>
#include <helper_cuda.h>

__global__ void kernel1(double* a) {
  extern __shared__ double buffer[];
  const auto tid_loc = threadIdx.x;
  const auto tid_glb = tid_loc + blockIdx.x * blockDim.x;

  buffer[tid_loc] = a[tid_glb];
  __syncthreads();

  a[tid_glb] = buffer[tid_loc] * 2.0;
}

__global__ void kernel2(int* a) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  a[tid] = __funnelshift_rc(0x00000000, 0xffffffff, 0x2);
  // a[tid] = __funnelshift_r(0x00000000, 0xffffffff, 0x2);
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
  int* b_h = new int[ar_size];
  int* b_d = nullptr;
  checkCudaErrors(cudaMalloc((void**)&b_d,
                             ar_size * sizeof(int)));

  // init
  std::fill(a_h, a_h + ar_size, 2.0);
  std::fill(b_h, b_h + ar_size, 2);

  // host -> device
  checkCudaErrors(cudaMemcpy(a_d, a_h, ar_size * sizeof(double),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(b_d, b_h, ar_size * sizeof(int),
                             cudaMemcpyHostToDevice));

  kernel1<<<gr_size, tb_size, tb_size * sizeof(double)>>>(a_d);
  checkCudaErrors(cudaDeviceSynchronize());

  kernel2<<<gr_size, tb_size>>>(b_d);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(a_h, a_d, ar_size * sizeof(double),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(b_h, b_d, ar_size * sizeof(int),
                             cudaMemcpyDeviceToHost));

  std::cout << std::hex;
  std::cout << a_h[0] << std::endl;
  std::cout << b_h[0] << std::endl;
}

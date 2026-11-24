// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace onnxruntime {

// cudaSetDevice selects the primary context, even when another context for the same
// device is already current. A borrowed context must be current on the calling thread.
inline cudaError_t CheckUserCudaContext(void* context, int device) {
  if (!context) return cudaSuccess;
  CUcontext current{};
  CUdevice current_device{};
  return cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == context &&
                 cuCtxGetDevice(&current_device) == CUDA_SUCCESS && current_device == device
             ? cudaSuccess
             : cudaErrorIncompatibleDriverContext;
}

inline cudaError_t SelectProviderCudaDevice(int device, void* context) {
  return context ? CheckUserCudaContext(context, device) : cudaSetDevice(device);
}

inline cudaError_t CheckUserCudaStream(void* context, cudaStream_t stream) {
  if (!context || !stream) return cudaSuccess;
  CUcontext owner{};
  return cuStreamGetCtx(stream, &owner) == CUDA_SUCCESS && owner == context ? cudaSuccess : cudaErrorInvalidResourceHandle;
}

// Resource deleters may run on a different thread from Run(). Only push when
// necessary and leave the previous binding (including no binding) intact.
class UserCudaContextScope {
 public:
  UserCudaContextScope() = default;
  UserCudaContextScope(const UserCudaContextScope&) = delete;
  UserCudaContextScope& operator=(const UserCudaContextScope&) = delete;

  CUresult Enter(void* context) {
    if (!context) return CUDA_SUCCESS;
    CUcontext current{};
    CUresult result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS || current == context) return result;
    result = cuCtxPushCurrent(static_cast<CUcontext>(context));
    pushed_ = result == CUDA_SUCCESS;
    return result;
  }

  ~UserCudaContextScope() {
    if (pushed_) {
      CUcontext popped{};
      (void)cuCtxPopCurrent(&popped);
    }
  }

 private:
  bool pushed_{false};
};

}  // namespace onnxruntime

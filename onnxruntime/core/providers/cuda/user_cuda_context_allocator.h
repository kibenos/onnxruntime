// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>

#include "core/framework/allocator.h"
#include "core/providers/cuda/user_cuda_context.h"

namespace onnxruntime {

// Wrap only the raw device and pinned allocators beneath a session's arena.
// An OrtValue can outlive its session and be released on any host thread.
class UserCudaContextAllocator final : public IAllocator {
 public:
  UserCudaContextAllocator(std::unique_ptr<IAllocator> allocator, void* context)
      : IAllocator(allocator->Info()), allocator_(std::move(allocator)), context_(context)
  {}

  void* Alloc(size_t size) override {
    UserCudaContextScope scope;
    ORT_ENFORCE(scope.Enter(context_) == CUDA_SUCCESS, "Unable to enter user_cuda_context for allocation.");
    return allocator_->Alloc(size);
  }

  void Free(void* ptr) override {
    UserCudaContextScope scope;
    if (scope.Enter(context_) != CUDA_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Unable to enter user_cuda_context to free CUDA memory.";
      return;
    }
    allocator_->Free(ptr);
  }

 private:
  std::unique_ptr<IAllocator> allocator_;
  void* context_;
};

inline std::unique_ptr<IAllocator> WithUserCudaContext(std::unique_ptr<IAllocator> allocator, void* context) {
  if (context) return std::make_unique<UserCudaContextAllocator>(std::move(allocator), context);
  return allocator;
}

}  // namespace onnxruntime

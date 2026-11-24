# Using an application-owned CUDA context

An application that already owns device allocations in a non-primary CUDA driver context can run a model against those allocations. The built-in CUDA and TensorRT execution providers accept the pointer-valued option `user_cuda_context`. The value is the `CUcontext` handle itself. Its device must match the provider's `device_id`. Passing a null handle restores the default provider behavior.

Set the option with `UpdateCUDAProviderOptionsWithValue` or `UpdateTensorRTProviderOptionsWithValue` before appending the provider. If TensorRT uses CUDA as a fallback, configure both providers with the same context. An explicit `user_compute_stream` must belong to that context. Use I/O binding to pass device tensors allocated in that context. For graph replay, keep the bound input and output addresses stable across runs.

The application owns the context and its current-thread binding. Push the context before appending providers, creating the session, binding or copying tensors, running inference, and destroying bindings, tensors and the session. Restore the previous binding on every exit path using `cuCtxPopCurrent`. Keep the context alive until outstanding work and resource destruction have finished. The providers do not retain or destroy the handle. Each thread that calls into the session must establish its own binding. A graph replay on a thread with a different context fails, including when both contexts use the same GPU.

Use sequential execution with a single GPU, per-session allocators, and synchronous `Run`. A thread pool that performs GPU work must establish the same context on each worker. This option targets the built-in providers' V2 APIs. The separately registered plugin providers have independent option interfaces.

For example, inside an application scope that keeps `ctx` current:

```cpp
Ort::CUDAProviderOptions cuda;
cuda.UpdateWithValue("user_cuda_context", ctx);
cuda.UpdateWithValue("user_compute_stream", stream);
Ort::SessionOptions options;
options.SetExecutionMode(ORT_SEQUENTIAL);
options.AppendExecutionProvider_CUDA_V2(*cuda);
Ort::Session session(env, model_path, options);
```

---
date: 2024-12-12
title: "GPU-to-GPU Copy over PCIe: From cudaMemcpyAsync to a Custom Kernel"
short: "GPU-to-GPU Copy over PCIe"
titleZh: "手撸一下 GPU D2D 实现（PCIE 版）"
description: How much bandwidth can a device-to-device copy actually reach between two GPUs on a PCIe topology? Four implementations measured on A800 PCIe and RTX 4090, and compared against NCCL.
---

# GPU-to-GPU Copy over PCIe: From cudaMemcpyAsync to a Custom Kernel

> Originally published in Chinese on [Zhihu](https://zhuanlan.zhihu.com/p/2847929235), December 12, 2024.

## 1. Introduction

Alongside the NVLINK-equipped machines that dominate discussion of large-model training and inference, a substantial share of server-side deployments connect their GPUs over PCIe. A common topology puts eight GPUs on a host, four per side, each group reaching a single CPU through a PCIe switch. A100 (A800) PCIe, RTX 4090 and L4 machines are all typically built this way.

![](/blog/gpu-d2d-pcie/fig01.jpg)

*A typical eight-GPU topology connected through PCIe switches*

The cross-GPU collectives — AllReduce, AlltoAll — all reduce to the same primitive: a device-to-device copy between two GPUs. PCIe 4.0 carries 16 GT/s per lane, so 16 lanes give $16/8 \times 16\ \text{lanes} = 32$ GB/s in one direction. By that arithmetic each GPU has 32 GB/s up and down to its PCIe switch, and each switch has 32 GB/s to CPU0. Two questions follow, and both matter for anyone tuning infrastructure:

- Does a transfer between two arbitrary GPUs actually reach the theoretical 32 GB/s? And how does same-socket bandwidth compare with cross-socket?
- Under the same PCIe topology, does the GPU model itself affect achievable P2P bandwidth?

This post answers both by measuring several different implementations.

## 2. Background

### 2.1 GPU Direct P2P over PCIe

When a machine provides a high-speed interconnect — A100 SXM or H100 SXM over NVLINK, or AMD's full-mesh inter-GPU links — device-to-device traffic never touches the CPU. The GPUs permit direct peer-to-peer access and the transfer runs over the fast interconnect. That case is well understood and is not the subject here.

GPU Direct P2P is the NVIDIA feature that lets a CUDA program read and move data from one GPU's memory to another's without routing through the shared system memory pool attached to the CPU. A kernel can dereference an address in another GPU's memory directly, with no explicit copy and no CPU involvement in scheduling.

![](/blog/gpu-d2d-pcie/fig02.jpg)

*Communication paths with and without GPU Direct P2P support. "Chipset" here means the PCIe bus.*

The feature is usually associated with A100 SXM and H100 SXM machines. The question is whether two GPUs connected over PCIe can use it as well. They can.

## 3. Experimental Setup

Two machines, both dual-socket with eight GPUs on PCIe 4.0 x16, CPU and GPU connected over PCIe.

- **Machine 1** — A800 PCIe × 8, which supports peer access (queryable via [`cudaDeviceCanAccessPeer`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html))

  - Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20 GHz
  - Ubuntu 20.04.6 LTS

- **Machine 2** — RTX 4090 × 8, which does not support peer access

  - Intel(R) Xeon(R) Platinum 8352S CPU @ 2.20 GHz
  - Ubuntu 20.04.6 LTS

### 3.1 Experiment Design

The goal is a CUDA program that performs a D2D transfer between any two GPUs in the machine, with a payload large enough (4 GB and above) to saturate the link.

To compare implementation strategies fairly — infrastructure work often ends up hand-writing communication kernels — I implemented several: plain CUDA API calls through to a hand-written kernel. To capture bandwidth contention when multiple GPUs transmit at once, the harness can launch traffic across several device pairs simultaneously.

### 3.2 Implementations

#### 3.2.1 Naive CUDA API call

The direct approach is `cudaMemcpy(Async)` with `cudaMemcpyDeviceToDevice`, called on the source device. The full source is [here](https://github.com/shenh10/awesome-cuda/blob/master/comm/d2d/cudaMemcpy.cu).

```cpp
for (size_t i = 0; i < devicePairs.size(); ++i) {
  int fromDevice = devicePairs[i].first;
  // Ensure device is set to fromDevice
  cudaSetDevice(
      fromDevice);  // Set device to fromDevice where stream[i] resides

  checkCudaErrors(cudaEventCreate(&startEvents[i]));
  checkCudaErrors(cudaEventCreate(&stopEvents[i]));

  checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));
  checkCudaErrors(cudaMemcpyAsync(d_dsts[i], d_srcs[i], size,
                                  cudaMemcpyDeviceToDevice, streams[i]));
  checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
}

// Synchronize all streams to ensure all operations for the current size are
// completed
for (size_t i = 0; i < devicePairs.size(); ++i) {
  int fromDevice = devicePairs[i].first;
  cudaSetDevice(fromDevice);  // Ensure device is set before synchronizing

  checkCudaErrors(cudaStreamSynchronize(streams[i]));

  float milliseconds = 0;
  checkCudaErrors(
      cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]));
  totalMilliseconds[i] += milliseconds;

  checkCudaErrors(cudaEventDestroy(startEvents[i]));
  checkCudaErrors(cudaEventDestroy(stopEvents[i]));
}
```

The Nsight Systems timeline shows what this actually does: `cudaMemcpyAsync` stages through the CPU, decomposing into a D2H copy followed by an H2D copy.

![](/blog/gpu-d2d-pcie/fig03.jpg)

*What a naive cudaMemcpyAsync does underneath — Nsight Systems timeline*

The host memory it stages through is **pageable**, which has two consequences:

1. **The copy cannot overlap with other CPU work.** Neither the launches across multiple device pairs nor successive iterations execute concurrently on the timeline.
2. **The CPU participates in the copy.** A GPU cannot read pageable host memory directly, so the CPU must allocate a temporary pinned staging buffer for the transfer to complete (see [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)). This is inherently inefficient.

![](/blog/gpu-d2d-pcie/fig04.jpg)

*Pageable data transfer vs. pinned data transfer*

Measured on A800 PCIe and 4090 PCIe:

| GPU Type | Scenario | Unidirectional bandwidth (GB/s) |
| --- | --- | --- |
| A800 PCIe | Same socket, different GPUs (e.g. [0, 1], [1, 0]) | 17.8 |
| A800 PCIe | Cross socket, different GPUs (e.g. [0, 4], [4, 0]) | 20.6 |
| 4090 PCIe | Same socket, different GPUs (e.g. [0, 1], [1, 0]) | 19.1 |
| 4090 PCIe | Cross socket, different GPUs (e.g. [0, 4], [4, 0]) | 19.6 |

One observation stands out: on both the A800 and the 4090, **same-socket bandwidth is lower than cross-socket bandwidth**, and markedly so on the A800. I have not found an official explanation. My hypothesis is that scheduling the D2H and H2D halves within one socket contends for the same CPU resources more heavily than splitting them across two.

The communication path:

![](/blog/gpu-d2d-pcie/fig05.jpg)

*Communication path from GPU 0 to GPU 1*

#### 3.2.2 Optimization 1: double-buffered streaming through the CPU

Pageable-memory copies are inefficient, and they also block us from launching several links concurrently. Can we manage the pinned-memory transfer ourselves, and get concurrent traffic on multiple links — a 0-1-2-3 ring, say?

That motivates a double-buffered streaming implementation:

![](/blog/gpu-d2d-pcie/fig06.jpg)

*Double-buffered streaming transfer*

The source data is split into equal chunks, sized against the total payload — for a 4 GB transfer, 64 MB per chunk is reasonable. On the host we allocate two pinned buffers of one chunk each. Two buffers are what allow the H2D of the previous chunk and the D2H of the next to run concurrently.

Because a pinned buffer can only be overwritten once its D2H has completed, ordering has to be enforced with events. The core of the implementation:

```cpp
template<typename T>  
void nonPeerD2DCopyWithDoublePinned(const T* d_src, int srcDevice,  
                                   T* d_dst, int dstDevice,  
                                   size_t numElements,  
                                   cudaStream_t srcStream,  
                                   cudaStream_t dstStream,  
                                   PinnedMemoryPool<T>& memPool) {  
    const size_t CHUNK_SIZE = memPool.getSize();  
    const size_t numChunks = (numElements + CHUNK_SIZE - 1) / CHUNK_SIZE;  
    
    int currentBuffer = 0;  
    size_t offset = 0;  
    
    // Kick off the first transfer  
    if (numChunks > 0) {  
        size_t currentChunkSize = std::min(CHUNK_SIZE, numElements);  
        checkCudaErrors(cudaSetDevice(srcDevice));  
        checkCudaErrors(cudaMemcpyAsync(memPool.getBuffer(currentBuffer),  
                                      d_src,  
                                      currentChunkSize * sizeof(T),  
                                      cudaMemcpyDeviceToHost,  
                                      srcStream));  
        checkCudaErrors(cudaEventRecord(memPool.getSrcEvent(currentBuffer), srcStream));  
    }  
    
    // Steady state: every full chunk  
    for (size_t chunk = 1; chunk < numChunks; ++chunk) {  
        int nextBuffer = 1 - currentBuffer;  
        size_t nextOffset = chunk * CHUNK_SIZE;  
        size_t currentChunkSize = std::min(CHUNK_SIZE, numElements - offset);  
        size_t nextChunkSize = std::min(CHUNK_SIZE, numElements - nextOffset);  
        
        checkCudaErrors(cudaSetDevice(dstDevice));  
        checkCudaErrors(cudaStreamWaitEvent(dstStream, memPool.getSrcEvent(currentBuffer)));  
        checkCudaErrors(cudaMemcpyAsync(d_dst + offset,  
                                      memPool.getBuffer(currentBuffer),  
                                      currentChunkSize * sizeof(T),  
                                      cudaMemcpyHostToDevice,  
                                      dstStream));  
        checkCudaErrors(cudaEventRecord(memPool.getDstEvent(currentBuffer), dstStream));  
        
        checkCudaErrors(cudaSetDevice(srcDevice));  
        checkCudaErrors(cudaStreamWaitEvent(srcStream, memPool.getDstEvent(nextBuffer)));  
        checkCudaErrors(cudaMemcpyAsync(memPool.getBuffer(nextBuffer),  
                                      d_src + nextOffset,  
                                      nextChunkSize * sizeof(T),  
                                      cudaMemcpyDeviceToHost,  
                                      srcStream));  
        checkCudaErrors(cudaEventRecord(memPool.getSrcEvent(nextBuffer), srcStream));  
        
        offset = nextOffset;  
        currentBuffer = nextBuffer;  
    }  
    
    // Drain the final chunk  
    if (numChunks > 0) {  
        size_t currentChunkSize = std::min(CHUNK_SIZE, numElements - offset);  
        checkCudaErrors(cudaSetDevice(dstDevice));  
        checkCudaErrors(cudaStreamWaitEvent(dstStream, memPool.getSrcEvent(currentBuffer)));  
        checkCudaErrors(cudaMemcpyAsync(d_dst + offset,  
                                      memPool.getBuffer(currentBuffer),  
                                      currentChunkSize * sizeof(T),  
                                      cudaMemcpyHostToDevice,  
                                      dstStream));  
        checkCudaErrors(cudaEventRecord(memPool.getDstEvent(currentBuffer), dstStream));  
    }  
}
```

The memory pool exists so that device memory can be reclaimed without an explicit `cudaFree`, which would force a synchronization. Full source [here](https://github.com/shenh10/awesome-cuda/blob/317d037928c0d9eaaf33f069d3146cd1e4e42004/comm/d2d/noPaDoubleBuffer.cu#L78). The Nsight timeline confirms both properties: multiple links transfer concurrently, and each link streams chunk by chunk.

![](/blog/gpu-d2d-pcie/fig07.jpg)

*Concurrent traffic on two links, GPU 0 → GPU 1 and GPU 2 → GPU 3, on the A800*

A question worth sitting with: **why does D2H appear to run faster than H2D?**

| GPU Type | Scenario | Unidirectional bandwidth (GB/s) |
| --- | --- | --- |
| A800 PCIe | Same socket, different GPUs (e.g. [0, 1], [1, 0]) | 20.34 |
| A800 PCIe | Cross socket, different GPUs (e.g. [0, 4], [4, 0]) | 24.97 |
| 4090 PCIe | Same socket, different GPUs (e.g. [0, 1], [1, 0]) | 18.47 |
| 4090 PCIe | Cross socket, different GPUs (e.g. [0, 4], [4, 0]) | 24.91 |

This is a substantial improvement over plain `cudaMemcpyAsync`, and it achieves the concurrency we wanted. In the concurrent 0→1 and 2→3 case, both links traverse the same PCIe-switch-to-CPU path, so they contend: each link reaches roughly half the bandwidth it gets on its own.

| GPU Type | Scenario | Unidirectional bandwidth (GB/s) |
| --- | --- | --- |
| A800 PCIe | Same socket, two disjoint pairs concurrently (e.g. [0,1], [2,3]) | 10.98, 10.98 |
| A800 PCIe | Cross socket, two disjoint pairs concurrently (e.g. [0,4], [1,5]) | 12.79, 12.79 |
| A800 PCIe | Same socket, 4-GPU ring | 4.84, 4.81, 4.81, 4.81 |
| A800 PCIe | Cross socket, 8-GPU ring | 4.72, 4.71, 4.71, 4.72, 4.72, 4.72, 4.72, 4.72 |
| 4090 PCIe | Same socket, two disjoint pairs concurrently (e.g. [0,1], [2,3]) | 10.14, 10.14 |
| 4090 PCIe | Cross socket, two disjoint pairs concurrently (e.g. [0,4], [1,5]) | 12.75, 12.75 |
| 4090 PCIe | Same socket, 4-GPU ring | 5.1, 5.1, 5.1, 5.1 |
| 4090 PCIe | Cross socket, 8-GPU ring | 4.68, 4.67, 4.68, 4.68, 4.67, 4.67, 4.67, 4.67 |

Worth noting from the timeline: the first D2H and the last H2D have nothing to overlap with, so they cost latency and bandwidth outright. When the total payload is small the pipeline's head and tail dominate — at a 64 MB chunk size and a 256 MB payload, the overhead is significant. Shrinking the chunk raises launch overhead instead:

![](/blog/gpu-d2d-pcie/fig08.jpg)

*Double-buffer method: chunk_size = 64 MB, total_size = 256 MB*

| Total size | Chunk size | Unidirectional bandwidth (GB/s) |
| --- | --- | --- |
| 256 MB | 64 MB | 17.50 |
| 256 MB | 1 MB | 17.81 |

A progressive variant addresses this by ramping the chunk size from small to large (512 KB → 64 MB), which at least removes the latency cost of the first D2H. [Implementation](https://github.com/shenh10/awesome-cuda/blob/317d037928c0d9eaaf33f069d3146cd1e4e42004/comm/d2d/noPaDoubleBuffer.cu#L144).

![](/blog/gpu-d2d-pcie/fig09.jpg)

*Progressive double-buffer method: chunk_size = 512 KB … 64 MB, total_size = 256 MB*

This brings unidirectional bandwidth up to 19.54 GB/s, largely resolving the problem.

#### 3.2.3 Optimization 2: enable peer access

Double buffering helps, but it cannot avoid staging through host memory. On devices that support peer access we should skip the host entirely and transfer device to device — which also keeps traffic off the PCIe-to-CPU path and removes the contention that appears when several GPUs transmit at once.

![](/blog/gpu-d2d-pcie/fig10.jpg)

*With peer access, no host-memory staging is required*

The code change over 3.2.1 is small: enable `cudaDeviceEnablePeerAccess`, then call `cudaMemcpy(Async)` with `cudaMemcpyDeviceToDevice` as before.

```cpp
// Enable peer access between devices if possible and requested
  if (enablePeerAccess) {
    for (const auto& pair : devicePairs) {
      int fromDevice = pair.first;
      int toDevice = pair.second;

      int canAccessPeer = 0;
      checkCudaErrors(
          cudaDeviceCanAccessPeer(&canAccessPeer, fromDevice, toDevice));
      if (canAccessPeer) {
        cudaSetDevice(fromDevice);
        cudaError_t err = cudaDeviceEnablePeerAccess(toDevice, 0);
        if (err == cudaSuccess) {
          std::cout << "Peer access enabled from device " << fromDevice
                    << " to device " << toDevice << std::endl;
        } else {
          std::cout << "Failed to enable peer access from device " << fromDevice
                    << " to device " << toDevice << ": "
                    << cudaGetErrorString(err) << std::endl;
        }
      } else {
        std::cout << "Peer access not supported from device " << fromDevice
                  << " to device " << toDevice << std::endl;
      }

      checkCudaErrors(
          cudaDeviceCanAccessPeer(&canAccessPeer, toDevice, fromDevice));
      if (canAccessPeer) {
        cudaSetDevice(toDevice);
        cudaError_t err = cudaDeviceEnablePeerAccess(fromDevice, 0);
        if (err == cudaSuccess) {
          std::cout << "Peer access enabled from device " << toDevice
                    << " to device " << fromDevice << std::endl;
        } else {
          std::cout << "Failed to enable peer access from device " << toDevice
                    << " to device " << fromDevice << ": "
                    << cudaGetErrorString(err) << std::endl;
        }
      } else {
        std::cout << "Peer access not supported from device " << toDevice
                  << " to device " << fromDevice << std::endl;
      }
    }
  }

// ... then call cudaMemcpyAsync as in 3.2.1
```

Measurements (the 4090 does not support peer access):

| GPU Type | Scenario | Unidirectional bandwidth (GB/s) |
| --- | --- | --- |
| A800 PCIe | Same socket, different GPUs (e.g. [0, 1], [1, 0]) | 25.2 |
| A800 PCIe | Cross socket, different GPUs (e.g. [0, 4], [4, 0]) | 18.5 |
| A800 PCIe | Same socket, two disjoint pairs concurrently (e.g. [0,1], [2,3]) | 25.2, 25.2 |
| A800 PCIe | Cross socket, two disjoint pairs concurrently (e.g. [0,4], [1,5]) | 9.1, 9.1 |
| A800 PCIe | Same socket, 4-GPU ring | 25.2, 25.2, 25.2, 25.2 |
| A800 PCIe | Cross socket, 8-GPU ring | 25.2, 25.2, 25.1, 17.4, 25.2, 25.2, 25.1, 17.5 |

Peer access lifts same-socket bandwidth substantially, and two disjoint pairs now transfer concurrently without interfering with each other — the direct benefit of bypassing host memory and CPU scheduling. Where peer access is available, it should be used.

#### 3.2.4 Optimization 3: a custom D2D copy kernel

Everything above goes through the CUDA API. It is also possible to write the transfer as a kernel directly, which is what you want when the communication algorithm is your own design, or when you need fine-grained overlap between computation and communication. A hand-written kernel gets close to the API's performance. There are two forms:

- **Pull mode** — the target device pulls data from the source device
- **Push mode** — the source device pushes data to the target device

For brevity I will skip the host-staging variant needed on devices without peer access; the reasoning is the same. With peer access, pull and push can share the kernel implementation — the difference is only which device launches it.

```cpp
// Vectorized copy kernel — int4 loads and stores
__global__ void vectorizedMemcpyKernel(const int4 *src, int4 *dst,
                                       size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = src[idx];
  }
}

// Kernel for the tail elements
__global__ void remainderMemcpyKernel(const int *src, int *dst, size_t start,
                                      size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
  if (idx < numElements) {
    dst[idx] = src[idx];
  }
}

// Push-mode memcpy
void pushModeMemcpy(int *d_src, int *d_dst, size_t numElements,
                    cudaStream_t stream) {
  const int threadsPerBlock = 256;
  size_t vectorElements = numElements / 4;
  size_t vectorBlocks =
      (vectorElements + threadsPerBlock - 1) / threadsPerBlock;

  // Bulk of the copy, vectorized as int4
  vectorizedMemcpyKernel<<<vectorBlocks, threadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int4 *>(d_src), reinterpret_cast<int4 *>(d_dst),
      vectorElements);

  // Handle the remaining elements
  size_t remainderStart = vectorElements * 4;
  size_t remainderElements = numElements - remainderStart;
  if (remainderElements > 0) {
    size_t remainderBlocks =
        (remainderElements + threadsPerBlock - 1) / threadsPerBlock;
    remainderMemcpyKernel<<<remainderBlocks, threadsPerBlock, 0, stream>>>(
        d_src + remainderStart, d_dst + remainderStart, remainderStart,
        numElements);
  }
}

// Pull-mode memcpy. Identical to push mode — a CUDA kernel does not
// distinguish the two; only the launching device differs.
void pullModeMemcpy(int *d_src, int *d_dst, size_t numElements,
                    cudaStream_t stream) {
  pushModeMemcpy(d_src, d_dst, numElements, stream);
}

int main(int argc, char **argv) {
// ...
        if (mode == "pull") {
          checkCudaErrors(cudaSetDevice(toDevice));
          checkCudaErrors(cudaEventCreate(&startEvents[i]));
          checkCudaErrors(cudaEventCreate(&stopEvents[i]));

          checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));

          pullModeMemcpy(d_srcs[i], d_dsts[i], numElements, streams[i]);

          checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
        } else if (mode == "push") {
          checkCudaErrors(cudaSetDevice(fromDevice));

          checkCudaErrors(cudaEventCreate(&startEvents[i]));
          checkCudaErrors(cudaEventCreate(&stopEvents[i]));

          checkCudaErrors(cudaEventRecord(startEvents[i], streams[i]));

          pushModeMemcpy(d_srcs[i], d_dsts[i], numElements, streams[i]);

          checkCudaErrors(cudaEventRecord(stopEvents[i], streams[i]));
        }
// ...  
}
```

Measurements (the 4090 does not support peer access):

| GPU Type | Scenario | Pull mode (GB/s) | Push mode (GB/s) |
| --- | --- | --- | --- |
| A800 PCIe | Same socket, different GPUs (e.g. [0, 1], [1, 0]) | 26.3 | 21.6 |
| A800 PCIe | Cross socket, different GPUs (e.g. [0, 4], [4, 0]) | 18.1 | 12.5 |
| A800 PCIe | Same socket, two overlapping pairs concurrently (e.g. [0,1], [1,2]) | 22.1, 21.3 | - |
| A800 PCIe | Same socket, two disjoint pairs concurrently (e.g. [0,1], [2,3]) | 26.5, 26.5 | 21.6, 21.6 |
| A800 PCIe | Cross socket, two disjoint pairs concurrently (e.g. [0,4], [1,5]) | 9.1, 9.1 | 6.25, 6.25 |
| A800 PCIe | Same socket, 4-GPU ring | 21.8, 21.8, 21.8, 21.8 | 21, 21, 21, 21 |
| A800 PCIe | Cross socket, 8-GPU ring | 22, 21, 22, 11, 22, 21, 22, 11 | 21, 21, 21, 12, 21, 21, 21, 12 |

Pull and push bandwidth are less symmetric than one might expect. On this A800, pull outperforms push; in earlier work I have seen the reverse. The topology alone does not explain it, and the cause remains open.

A second anomaly: the pull-mode ring does not reach the 26.3 GB/s that a single pair achieves. The [0,1], [1,2] case shows why — when a single GPU transmits and receives at the same time, bandwidth degrades, with kernel time rising from 650 ms without overlap to over 700 ms. This one is also unresolved.

### 3.3 Comparison with NCCL

For reference, the bandwidth reported by nccl-tests:

| GPU Type | Scenario | NCCL | Double buffer | Peer access + cudaMemcpyAsync D2D | Custom kernel, pull | Custom kernel, push |
| --- | --- | --- | --- | --- | --- | --- |
| A800 PCIe | Same socket, 4-GPU ring send/recv | 23.0 | 4.8 | 25.2 | 21.8 | 21 |
| A800 PCIe | Cross socket, 8-GPU ring send/recv | 12.9 | 4.7 | 17.4 | 11 | 12 |
| A800 PCIe | Same socket, 4-GPU ring allreduce | 22.2 | - | - | - | - |
| A800 PCIe | Cross socket, 8-GPU ring allreduce | 14.9 | - | - | - | - |
| 4090 PCIe | Same socket, 4-GPU ring send/recv | 5.0 | 5.1 | - | - | - |
| 4090 PCIe | Cross socket, 8-GPU ring send/recv | 4.9 | 4.7 | - | - | - |
| 4090 PCIe | Same socket, 4-GPU ring allreduce | 5.2 | - | - | - | - |
| 4090 PCIe | Cross socket, 8-GPU ring allreduce | 5.2 | - | - | - | - |

(All figures unidirectional, GB/s.)

NCCL evidently detects the GPU model and enables peer access where it is available, avoiding the host-memory round trip. One result is counterintuitive and worth flagging: ring send/recv measures *lower* than ring allreduce, which I cannot currently account for.

## 4. Conclusions

1. On GPUs that support peer access, such as the A800 PCIe, peak unidirectional bandwidth between two cards reaches roughly 80% of the theoretical 32 GB/s. The best implementation is the CUDA API with peer access enabled; a hand-written kernel comes close.
2. On GPUs without peer access, such as the RTX 4090 PCIe, the ceiling is roughly 60% of theoretical.
3. For collectives, NCCL remains the right default — it selects the best available configuration per hardware automatically. Hand-writing is worth considering only for cases it does not cover, and is not especially difficult.
4. The hand-written kernel still has headroom in its scheduling parameters.

## 5. References

1. [P2P peer-to-peer on NVIDIA RTX 2080Ti vs GTX 1080Ti GPUs](https://www.pugetsystems.com/labs/hpc/p2p-peer-to-peer-on-nvidia-rtx-2080ti-vs-gtx-1080ti-gpus-1331/)

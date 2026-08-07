---
title: 手撸一下 GPU D2D 实现（PCIE 版）
description: PCIE 拓扑下两块 GPU 之间的 D2D 拷贝能跑到多少带宽？从 naive cudaMemcpyAsync 一路优化到自定义向量化 kernel，并与 NCCL 对比实测。
---

# 手撸一下 GPU D2D 实现（PCIE 版）

> 本文最初发表于[知乎](https://zhuanlan.zhihu.com/p/2847929235)。

## 1. 前言

最近有同事在查 PCIE 架构下的大模型的通信效率问题，有时间（闲的）做了一个有趣的基础小实验，觉得可以分享出来供大家讨论，欢迎同行评论区交流拍砖找bug （手动狗头）。

当下服务器端的大模型训练/推理机型，除了提供高带宽 NVLINK 机器之外，有不少机内 通过PCIE 互联的机型。一种常见的网络拓扑如下所示： 8卡GPU的机器通过单边4卡通过 PCIE SW 连接到单个CPU，选择这种架构的常见机器包括 A100(A800) PCIE、4090、L4 等等。

![](/blog/gpu-d2d-pcie/fig01.jpg)

*常见的通过PCIE SW连接的8卡网络拓扑*

常见的跨GPU通信算子如AllReduce、AlltoAll 都涉及到最基本的操作——两个设备间的D2D 拷贝。一般来说，PCIE 4.0 的协议单lane 提供16 GT/s 的比特传输带宽，那么折合16 lanes则是 $16/8 \times 16\ \text{lanes} = 32$ GB/s 的单向带宽。也就是说，上图每块GPU 到PCIE Switch的上下行带宽分别为32GB/s，PCIE switch 到CPU0的单向带宽为32GB/s。于是 做infra加速的同学应该关注以下两个关键的问题：

*   实际情况中，任2个GPU之间的通信能达到理论的32GB/s 带宽吗？同socket和跨socket的通信带宽会有什么差异？
*   在同 PCIE 拓扑结构下，不同GPU卡型是否会影响P2P的通信带宽？

本文希望通过不同的实现方式的实测来回答上述问题。

## 2. 基础概念

### 2.1 GPU Direct P2P for PCIE

在D2D 通信中， 如果配置了高速互联，比如A100 SXM （NVLINK）, H100 SXM（NVLINK），或者AMD 的fullmesh 卡间互联，D2D 通信是不需要经过CPU的，卡本身会直接允许GPU Direct peer-to-peer access，走高速互联如NVLINK 通道通信。这个的带宽计算大家一般比较熟悉了，本文不再赘述。

GPU Direct P2P 是 NVIDIA GPU 中的一项功能，允许 CUDA 程序访问并将数据从一个 GPU 的内存传输到另一个 GPU 的内存，而无需通过连接到 CPU 的共享系统内存池。 这允许程序员从CUDA 内核中直接访问另一个GPU的内存地址，而不需要显示的内存拷贝，也不需要CPU来参与调度。

![](/blog/gpu-d2d-pcie/fig02.jpg)

*是否支持GPU Direct P2P 的通信路径区别，Chipset指的是PCIE 总线*

这个能力一般用来加速D2D 的拷贝，常常在A100 SXM和H100 SXM这些机器中使用。但是，通过 PCIE 连接的两块 GPU 是否也可以利用这项能力呢？答案是：yes！

## 3. 实验环境

我选择了两款卡进行实验，都是PCIE 4.0 x 16 的双CPU 8卡配置，CPU与GPU通过PCIE连接。

*   **机器1:** 本身支持peer access 的A800 PCIE x 8 （可以通过 [cudaDeviceCanAccessPeer](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html)API查询）

    *   Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz
    *   Ubuntu 20.04.6 LTS

*   **机器2:** 卡本身不支持peer access的 RTX 4090 x 8

    *   Intel(R) Xeon(R) Platinum 8352S CPU @ 2.20GHz 
    *   Ubuntu 20.04.6 LTS

### 3.1 实验设计

希望通过CUDA 程序来实现机内任意两卡的 D2D 通信。传输量足够大（4G以上）能打满带宽。

为了充分对比不同的实现方式的效率差异（考虑到infra同学常常会收撸通信算子），我设计了CUDA API 调用及手写算子等多种可能的实现方式；为了模拟多卡同时收发的带宽竞争，我会考虑同时launch 多个device pair之间的通信。

### 3.2 实现方式

#### 3.2.1 Naive CUDA API call

我们可以直接通过CUDA API `cudaMemcpy(Async)` 的 `cudaMemcpyDeviceToDevice` 选项 来实现同机内跨卡D2D的拷贝。实现方式非常简单：在srcDevice 上调用 `cudaMemcpyAsync`。`cudaMemcpyAsync` 示例如下所示，完整代码参考[这里](https://github.com/shenh10/awesome-cuda/blob/master/comm/d2d/cudaMemcpy.cu)。

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

从Nsight System 的timeline 观察到，cudaMemcpyAsync 实际会以CPU 为中转，转为一个D2H 和H2D 的memory copy。

![](/blog/gpu-d2d-pcie/fig03.jpg)

*Naive cudaMemcpyAsync Underneath --- Nsys Timeline*

值得注意的是，这个CPU memory 是 pagable 的，这意味着

1.   **拷贝操作无法和其他CPU操作异步。**可以看到Timeline 上无论是多个devicePair的kernel下发，或者是多轮迭代执行，都无法在timeline上同步执行。
2.   **拷贝过程 CPU的参与**——GPU 无法直接从可分页主机内存访问数据，需要CPU参与分配一块临时pinned memory才能完成到GPU的拷贝（详情可参考[How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)），这无疑是低效的。

![](/blog/gpu-d2d-pcie/fig04.jpg)

*Pageable Data Transfer vs Pinned Data Transfer*

我们在A800 PCIE 和4090 PCIE分别做了实验

| GPU Type | 测试场景 | 单向带宽（GB/s） |
| --- | --- | --- |
| A800 PCIE | 同socket 不同卡(eg. [0, 1], [1, 0]） | 17.8 |
| A800 PCIE | 跨socket 不同卡(eg. [0, 4], [4, 0]) | 20.6 |
| 4090 PCIE | 同socket 不同卡(eg. [0, 1], [1, 0]） | 19.1 |
| 4090 PCIE | 跨socket 不同卡(eg. [0, 4], [4, 0]) | 19.6 |

有一个很有趣的观察，在A800 和4090中，都发现了**_同socket 带宽不如跨socket 带宽高_**的现象，在A800 上提现的更为明显。目前没有查到相关的官方解释，笔者猜测是由于H2D和D2H的调度在同socket竞争同一个CPU资源，会比跨socket 竞争更大。

通信拓扑示意：

![](/blog/gpu-d2d-pcie/fig05.jpg)

*GPU 0 到 GPU 1 通信链路示意*

#### 3.2.2 优化1. 基于CPU 中转的Double Buffer 流式传输

由于上述实现中，pagable memory拷贝时非常低效的，并且阻碍了我们同时launch多个链路的通信。 我们有没有办法自己实现 pinned memory 的传输，同时能使多路通信同时发生（比如0-1-2-3的环状通信）呢？

为此我实现了一个基于 Double buffer 的流式传输代码实现：

![](/blog/gpu-d2d-pcie/fig06.jpg)

*基于Double Buffer 的流式传输*

为了实现流式传输，我们将原数据分成多个相等大小的chunk，每个chunk选择chunk_size左右的大小（根据实际的总负载指定，比如传输4GB，chunk_size 设为 64MB）。在host memory上，我们申请两块 chunk_size 大小的pinned memory 作为buffer。双buffer 是为了实现前一份chunk 的 H2D 和 下一份 chunk 的 D2H 并发执行。

因为前一份chunk的D2H 结束后，pinned buffer才能被覆写以接受下一份chunk，我们需要利用event来进行时序的控制。代码实现核心逻辑如下：

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
    
    // 启动第一次传输  
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
    
    // 处理所有完整的块  
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
    
    // 处理最后一块数据  
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

memPool 的设计是为了最终做显存的回收，避免显示的`cudaFree` 带来的强制同步。完整代码见[这里](https://github.com/shenh10/awesome-cuda/blob/317d037928c0d9eaaf33f069d3146cd1e4e42004/comm/d2d/noPaDoubleBuffer.cu#L78) 。从nsys timeline 上我们可以看到成功实现了多个通信链路的并发传输，并且是一个个chunk串行流式传输的实现。

![](/blog/gpu-d2d-pcie/fig07.jpg)

*GPU 0 -&gt; GPU 1, GPU2 -&gt; GPU 2 双链路并发通信，A800 为例*

这里留下一个思考题：**为什么看起来D2H 要比H2D 快一些？**

实验数据如下

| GPU Type | 测试场景 | 单向带宽（GB/s） |
| --- | --- | --- |
| A800 PCIE | 同socket 不同卡(eg. [0, 1], [1, 0]） | 20.34 |
| A800 PCIE | 跨socket 不同卡(eg. [0, 4], [4, 0]) | 24.97 |
| 4090 PCIE | 同socket 不同卡(eg. [0, 1], [1, 0]） | 18.47 |
| 4090 PCIE | 跨socket 不同卡(eg. [0, 4], [4, 0]) | 24.91 |

可以看到，这个实现显著提升了`cudaMemcpyAsync` 实现的吞吐！并且成功实现了并发。可以看到0->1, 2->3的并发传输案例，因为都需要经过PCIE SW-CPU 的通信链路，导致了带宽的竞争，导致单条链路的单向带宽只能达到无并发时的一半。

| GPU Type | 测试场景 | 单向带宽（GB/s） |
| --- | --- | --- |
| A800 PCIE | 同socket 两组不相关卡同时收发(eg. [0,1], [2,3]) | 10.98, 10.98 |
| A800 PCIE | 不同socket 两组不相关卡同时收发(eg. [0,4], [1,5]) | 12.79, 12.79 |
| A800 PCIE | 同socket 4卡 ring | 4.84, 4.81, 4.81, 4.81 |
| A800 PCIE | 跨 socket 8卡 ring | 4.72, 4.71, 4.71, 4.72, 4.72, 4.72, 4.72, 4.72 |
| 4090 PCIE | 同socket 两组不相关卡同时收发(eg. [0,1], [2,3]) | 10.14, 10.14 |
| 4090 PCIE | 不同socket 两组不相关卡同时收发(eg. [0,4], [1,5]) | 12.75, 12.75 |
| 4090 PCIE | 同socket 4卡 ring | 5.1, 5.1, 5.1, 5.1 |
| 4090 PCIE | 跨 socket 8卡 ring | 4.68, 4.67, 4.68, 4.68, 4.67, 4.67, 4.67, 4.67 |

值得一提的事，我们观察时间线，第一个 D2H 和最后一个H2D 的拷贝因为没有办法并发，往往会带来一定的延迟和带宽损耗。当总传输量较小，比如当我chunk size 设为64MB，而总传输量只有256MB时， double buffer 首尾段的开销就比较大了。如果把chunk size 调小，那么launch overhead 也会变高：

![](/blog/gpu-d2d-pcie/fig08.jpg)

*DoubleBuffer method: chunk_size=64MB, total_size=256MB*

| Total Size | Chunk size | 单向带宽（GB/s） |
| --- | --- | --- |
| 256MB | 64MB | 17.50 |
| 256MB | 1 MB | 17.81 |

这里我实现了一个progressive 的double buffer 方案，可以将传输chunk_size 从小到大依次递增(512KB -> 64MB)，这样至少减少了第一份D2H延迟开销。详细实现可以看[代码](https://github.com/shenh10/awesome-cuda/blob/317d037928c0d9eaaf33f069d3146cd1e4e42004/comm/d2d/noPaDoubleBuffer.cu#L144)。

![](/blog/gpu-d2d-pcie/fig09.jpg)

ProgressiveDoubleBuffer method: chunk_size = 512KB~64MB, total_size=256MB 

这个实现下，单向带宽提升至了19.54 GB/s，有效缓解了上述问题。

#### 3.2.3 优化2：Enable Peer Access

DoubleBuffer的方案虽然取得了一定优化，还是避免不了需要经过host memory中转这件事。因此对于支持Peer Access的设备来说，我们应该避免经过 host memory 的拷贝，直接进行设备到设备的传输。这样各个设备之间的通信不需要经过PCIE-CPU 的通信链路，避免了多卡并行时的竞争。

![](/blog/gpu-d2d-pcie/fig10.jpg)

*支持Peer Access的设备，不需要经过 host memory 中转*

代码实现上，我们只需要在3.2.1 的基础上增加 `cudaDeviceEnablePeerAccess` `的使能，再调用cudaMemcpy(Async)` 的 `cudaMemcpyDeviceToDevice` 选项即可。

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

// ... 调用cudaMemcpyAsync
```

实验数据（4090 不支持PA）

| GPU Type | 测试场景 | 单向带宽（GB/s） |
| --- | --- | --- |
| A800 PCIE | 同socket 不同卡(eg. [0, 1], [1, 0]） | 25.2 |
| A800 PCIE | 跨socket 不同卡(eg. [0, 4], [4, 0]) | 18.5 |
| A800 PCIE | 同socket 两组不相关卡同时收发(eg. [0,1], [2,3]) | 25.2,25.2 |
| A800 PCIE | 不同socket 两组不相关卡同时收发(eg. [0,4], [1,5]) | 9.1, 9.1 |
| A800 PCIE | 同socket 4卡 ring | 25.2, 25.2, 25.2, 25.2 |
| A800 PCIE | 跨socket 8卡 ring | 25.2, 25.2, 25.1, 17.4, 25.2, 25.2, 25.1, 17.5 |

PA的实现显著提升了同socket 内不同卡的通信带宽！ 而且同socket 两组不相关卡同时收发互不影响，这就体现了不需要经过host memory和CPU调度的好处。因此，能使能PA的卡应该尽量用PA模式通信。

#### 3.2.4 优化3: D2D Copy Custom Kernel

上述实现都是通过cuda API完成的通信，我们其实也可以通过手写算子直接撸一个通信kernel，比如有时候我们的通信算法是自己设计的，或者希望做一些复杂的计算通信overlap，通过手写算子写的kernel 会更加可控。这里我也展示一下如何通过手写算子来实现一个跟官方效率差不多的kernel。手写算子有两种实现方式：

*   **pull mode：**target device 向source device 拉取数据
*   **push mode：**source device 向target device 推送数据

为了简单，这里就不讨论不支持 PA 的算子需要经过host memory中转的实现了，道理是相似的。我们展示在PA下，pull mode和push mode 的kernel如何实现。pull/push 本质上可以复用kernel的实现，区别主要在于是在srcDevice还是dstDevice上launch kernel。核心代码逻辑如下：

```cpp
// 使用 int4 进行向量化读写的 kernel
__global__ void vectorizedMemcpyKernel(const int4 *src, int4 *dst,
                                       size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = src[idx];
  }
}

// 处理剩余元素的 kernel
__global__ void remainderMemcpyKernel(const int *src, int *dst, size_t start,
                                      size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + start;
  if (idx < numElements) {
    dst[idx] = src[idx];
  }
}

// Push 模式 memcpy 函数
void pushModeMemcpy(int *d_src, int *d_dst, size_t numElements,
                    cudaStream_t stream) {
  const int threadsPerBlock = 256;
  size_t vectorElements = numElements / 4;
  size_t vectorBlocks =
      (vectorElements + threadsPerBlock - 1) / threadsPerBlock;

  // 使用 int4 进行主要的内存复制
  vectorizedMemcpyKernel<<<vectorBlocks, threadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int4 *>(d_src), reinterpret_cast<int4 *>(d_dst),
      vectorElements);

  // 处理剩余的元素
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

// Pull 模式 memcpy 函数 (与 Push 模式相同，因为 CUDA kernel 不区分 push 和
// pull)
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

实验数据（4090 不支持PA）

| GPU Type | 测试场景 | pull mode 单向带宽（GB/s） | push mode 单向带宽（GB/s） |
| --- | --- | --- | --- |
| A800 PCIE | 同socket 不同卡(eg. [0, 1], [1, 0]） | 26.3 | 21.6 |
| A800 PCIE | 跨socket 不同卡(eg. [0, 4], [4, 0]) | 18.1 | 12.5 |
| A800 PCIE | 同socket 两组重叠卡同时收发(eg. [0,1], [1,2]) | 22.1, 21.3 | - |
| A800 PCIE | 同socket 两组不相关卡同时收发(eg. [0,1], [2,3]) | 26.5,26.5 | 21.6,21.6 |
| A800 PCIE | 不同socket 两组不相关卡同时收发(eg. [0,4], [1,5]) | 9.1, 9.1 | 6.25, 6.25 |
| A800 PCIE | 同socket 4卡 ring | 21.8,21.8,21.8, 21.8 | 21,21,21,21 |
| A800 PCIE | 跨socket 8卡 ring | 22,21,22,11,22,21,22,11 | 21,21,21,12,21,21,21,12 |

实验发现，pull/push mode的实现的带宽在PCIE 通信中常常不如预期中对称，我们测试的这台A800 上pull mode好于push mode，但在我们过去的工作中，也有push model 好于pull mode 的情况。目前从拓扑上没有看出来根本的原因，欢迎有相关研究的朋友解惑。

另外从数据中观察的一个异常点是，pull mode 的ring 没有达到26.3 GB/s量级的单向带宽，我们从实验[0,1], [1,2] 可以看到，当有在同一张卡上同时收/发时，带宽会有损耗，从kernel耗时来说，从没有overlap 的650ms->700+ms，影响了可达带宽。这个疑问也是一个遗留问题。

### 3.3 对比NCCL

我们来看一下NCCL_TEST提供的allreduce带宽：

| GPU Type | 测试场景 | NCCL 单向带宽(GB/s) | Double Buffer 单向带宽(GB/s | Peer Access+cudaMemcpyAsync-D2D 单向带宽(GB/s | Custom Kernel-Pull 单向带宽(GB/s | Custom Kernel-Push 单向带宽(GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| A800 PCIE | 同socket 4卡ring send/recv | 23.0 | 4.8 | 25.2 | 21.8 | 21 |
| A800 PCIE | 跨socket 8卡ring send/recv | 12.9 | 4.7 | 17.4 | 11 | 12 |
| A800 PCIE | 同socket 4卡 ring-allreduce | 22.2 | - | - | - | - |
| A800 PCIE | 跨socket 8卡 ring-allreduce | 14.9 | - | - | - | - |
| 4090 PCIE | 同socket 4卡 ring-send/recv | 5.0 | 5.1 | - | - | - |
| 4090 PCIE | 跨socket 8卡 ring-send/recv | 4.9 | 4.7 | - | - | - |
| 4090 PCIE | 同socket 4卡 ring-allreduce | 5.2 | - | - | - | - |
| 4090 PCIE | 跨socket 8卡 ring-allreduce | 5.2 | - | - | - | - |

从测试结果可以看出，NCCL底层应该是自动做了卡型的判别，在可以使能PA的时候开启了PA，避免了host memory 的中转。NCCL 真香！

而且ring-send/recv 的带宽看起来还不如ring-allreduce? 陷入了沉思

## 4. 总结

综上所述，我们可以有以下结论：

1.   对于支持PA的卡型如A800 PCIE来说，最大的卡间单向收发带宽应能达到理论带宽（32GB/s）的80%左右，最优的实现可能是官方支持Peer Access的API调用。手写算子也基本可以达到接近官方API 的性能。
2.   对于不支持PA 的卡型如4090 PCIE来说，最大卡间收发带宽只能达到理论带宽（32GB/s）的60%左右。
3.   对于collective 的通信方式，还是NCCL 实现比较香！底层自动发挥了不同硬件的最佳带宽配置，在特殊场景再考虑手撸（也不复杂）
4.   手写算子还可以自己调一调scheduling参数优化优化，应该还有优化的空间！

## 5. 参考资料

1. [P2P peer-to-peer on NVIDIA RTX 2080Ti vs GTX 1080Ti GPUs](https://www.pugetsystems.com/labs/hpc/p2p-peer-to-peer-on-nvidia-rtx-2080ti-vs-gtx-1080ti-gpus-1331/)

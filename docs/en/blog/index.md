# Blog

Notes on technical exploration.

## Posts

### [HuggingArch: Automating Model Architecture Analysis](/en/blog/huggingarch)

A harness approach to automating LLM inference cost analysis: a three-tier spec system, sympy symbolic algebra carried end to end from backend to frontend, and a validation system that pins the agent to ground truth. Closes with the DeepSeek V4-Pro crash test — with no guards in place, an agent rewriting a spec from zero declares its own broken output valid 94% of the time.

### [GPU-to-GPU Copy over PCIe: From cudaMemcpyAsync to a Custom Kernel](/en/blog/gpu-d2d-pcie)

How much bandwidth can a device-to-device copy actually reach between two GPUs on a PCIe topology? Starting from a naive `cudaMemcpyAsync`, then adding pinned-memory double buffering, peer access, and a vectorized custom copy kernel — reading the Nsight timeline at each step — and finally comparing against NCCL. Measured on A800 PCIe and RTX 4090.

### [GPU Clock Throttling: Why You Never Reach Peak FLOPS](/en/blog/gpu-throttling)

When a large GEMM falls short of a GPU's rated TFLOPS, the vendor library is usually not the problem — the power budget is. What the SM and memory clocks actually do under load, how to catch throttling as it happens through NVML's `clocks_throttle_reasons`, and an FP16 square-GEMM sweep on T4, A10, A800 SXM/PCIe and H800 SXM showing peak MFU ranging from 56% to 89% of spec.

### [DeepSeek V3/R1 Inference Efficiency (1): A Back-of-the-Envelope Decoding Throughput Ceiling](/en/blog/ds-inference-1-throughput-ceiling)

Before DeepSeek published its inference-system figures, how close could you get from the V3 paper alone? A memory and compute budget for Attention DP + MoE EP, and the tokens/s per H800 it implies.


### [DeepSeek V3/R1 Inference Efficiency (2): Reverse-Engineering the Production Deployment](/en/blog/ds-inference-2-reverse-engineering)

Layer-by-layer profiling of the published EP144 setup — prefill and decode FLOPs, per-operator timings, MFU, and the overlap schedule that follows.


### [DeepSeek V3/R1 Inference Efficiency (3): Generalizing the Decode Configuration](/en/blog/ds-inference-3-decode-generalization)

A simulator over DeepGEMM, FlashMLA and torch, sweeping DP-EP and TP-DP-EP on H800 and H20 — which device count, batch size, KV cache dtype and overlap scheme actually maximize throughput under a latency SLO.

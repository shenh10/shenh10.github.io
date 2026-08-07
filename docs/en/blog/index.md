# Blog

Notes on technical exploration.

## Posts

### [HuggingArch: Automating Model Architecture Analysis](/en/blog/huggingarch)

A harness approach to automating LLM inference cost analysis: a three-tier spec system, sympy symbolic algebra carried end to end from backend to frontend, and a validation system that pins the agent to ground truth. Closes with the DeepSeek V4-Pro crash test — with no guards in place, an agent rewriting a spec from zero declares its own broken output valid 94% of the time.

### [GPU-to-GPU Copy over PCIe: From cudaMemcpyAsync to a Custom Kernel](/en/blog/gpu-d2d-pcie)

How much bandwidth can a device-to-device copy actually reach between two GPUs on a PCIe topology? Starting from a naive `cudaMemcpyAsync`, then adding pinned-memory double buffering, peer access, and a vectorized custom copy kernel — reading the Nsight timeline at each step — and finally comparing against NCCL. Measured on A800 PCIe and RTX 4090.

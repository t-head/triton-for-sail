import torch
import triton
import triton.language as tl
import time

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - _, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - _, other=0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc.to(tl.int8)  # 模拟实际量化场景
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def benchmark_int8_perf(device=DEVICE):
    # 设置超参数
    matrix_size = 8192  # 需要根据显存容量调整
    a = torch.randint(-128, 127, (matrix_size, matrix_size), device=device, dtype=torch.int8)
    b = torch.randint(-128, 127, (matrix_size, matrix_size), device=device, dtype=torch.int8)
    # b = torch.arange(0, 32 * 32, device=device, dtype=torch.int8).reshape(32, 32)
    c = torch.zeros((matrix_size, matrix_size), device=device, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(matrix_size, META['BLOCK_M']) * triton.cdiv(matrix_size, META['BLOCK_N']), )

    # 预热
    int8_matmul_kernel[grid](a, b, c, matrix_size, matrix_size, matrix_size,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4)
    import numpy as np
    torch_output = np.matmul(a.cpu().numpy().astype(np.float32), b.cpu().numpy().astype(np.float32())).astype(np.int8)
    torch.set_printoptions(threshold=1000000)
    np.set_printoptions(threshold=1000000)
    print(f"triton_output_with_fp16_inputs={c}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 1e-2
    if torch.allclose(c, torch.tensor(torch_output).to(DEVICE), atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        np.testing.assert_allclose(c.cpu(), torch_output, atol=1e-2, rtol=rtol)
        print("❌ Triton and Torch differ")
    # 正式测试
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        int8_matmul_kernel[grid](a, b, c, matrix_size, matrix_size, matrix_size,
                                  a.stride(0), a.stride(1),
                                  b.stride(0), b.stride(1),
                                  c.stride(0), c.stride(1),
                                  BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # 计算算力
    total_ops = 2 * matrix_size**3 * 10  # 2*M*N*K per matrix multiply
    tflops = total_ops / elapsed / 1e12
    return tflops

if __name__ == "__main__":
    peak_tflops = benchmark_int8_perf()
    print(f"INT8 Peak Performance: {peak_tflops:.2f} TOPS")
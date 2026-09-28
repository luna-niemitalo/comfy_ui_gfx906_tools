#include <hip/hip_runtime.h>
#include <cstdint>

// Block-scaled int4 GEMM for gfx906: A[M,K] activation, B[N,K] weight, both
// nibble-packed (native two's-complement, 2 elements/byte), block=32
// elements/scale (matches quantize_model_int4.py's convention). Rescales
// and flushes to an fp32 accumulator every 32-element block instead of
// accumulating the whole K reduction in one int32 register -- required
// because the scale is not constant across K. Verified against a CPU
// fp64 reference in int4_gemm_scaled.hip (scratch correctness harness).
#define BM 64
#define BN 64
#define BLOCK 32
#define BK 32
#define BK_BYTES (BK / 2)
#define TM 4
#define TN 4

__global__ void int4_gemm_scaled_kernel(
    const int8_t* __restrict__ A, const float* __restrict__ A_scale,
    const int8_t* __restrict__ B, const float* __restrict__ B_scale,
    float* __restrict__ C, int M, int N, int K_elems) {

    __shared__ int8_t As[BM][BK_BYTES];
    __shared__ int8_t Bs[BN][BK_BYTES];
    __shared__ float As_scale[BM];
    __shared__ float Bs_scale[BN];

    int K_bytes = K_elems / 2;
    int n_blocks = K_elems / BLOCK;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;
    int tx = tid % (BN / TN);
    int ty = tid / (BN / TN);

    float fp32_acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) fp32_acc[i][j] = 0.0f;

    const int threads = (BM / TM) * (BN / TN);
    const int a_elems = BM * BK_BYTES, b_elems = BN * BK_BYTES;

    for (int blk = 0; blk < n_blocks; blk++) {
        int k0 = blk * BK_BYTES;
        for (int idx = tid; idx < a_elems; idx += threads) {
            int r = idx / BK_BYTES, c = idx % BK_BYTES;
            int m = block_row + r, kb = k0 + c;
            As[r][c] = (m < M && kb < K_bytes) ? A[m * K_bytes + kb] : 0;
        }
        for (int idx = tid; idx < b_elems; idx += threads) {
            int r = idx / BK_BYTES, c = idx % BK_BYTES;
            int n = block_col + r, kb = k0 + c;
            Bs[r][c] = (n < N && kb < K_bytes) ? B[n * K_bytes + kb] : 0;
        }
        if (tid < BM) {
            int m = block_row + tid;
            As_scale[tid] = (m < M) ? A_scale[m * n_blocks + blk] : 0.0f;
        }
        if (tid < BN) {
            int n = block_col + tid;
            Bs_scale[tid] = (n < N) ? B_scale[n * n_blocks + blk] : 0.0f;
        }
        __syncthreads();

        const int32_t* As32 = reinterpret_cast<const int32_t*>(As);
        const int32_t* Bs32 = reinterpret_cast<const int32_t*>(Bs);

        int32_t int_partial[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) int_partial[i][j] = 0;

#pragma unroll
        for (int kk = 0; kk < BK_BYTES / 4; kk++) {
#pragma unroll
            for (int i = 0; i < TM; i++) {
                int32_t a_pack = As32[(ty * TM + i) * (BK_BYTES / 4) + kk];
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    int32_t b_pack = Bs32[(tx * TN + j) * (BK_BYTES / 4) + kk];
                    int_partial[i][j] = __builtin_amdgcn_sdot8(a_pack, b_pack, int_partial[i][j], false);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < TM; i++) {
            float a_s = As_scale[ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++) {
                float b_s = Bs_scale[tx * TN + j];
                fp32_acc[i][j] += (float)int_partial[i][j] * a_s * b_s;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int m = block_row + ty * TM + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int n = block_col + tx * TN + j;
            if (n >= N) continue;
            C[m * N + n] = fp32_acc[i][j];
        }
    }
}

void int4_gemm_scaled_launch(const int8_t* A, const float* As, const int8_t* B, const float* Bs,
                              float* C, int M, int N, int K, hipStream_t stream) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
   hipLaunchKernelGGL(( int4_gemm_scaled_kernel), dim3(grid), dim3(block), 0, stream, A, As, B, Bs, C, M, N, K);
}

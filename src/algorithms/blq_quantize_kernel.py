__global__ void blq_quantize_kernel(
    const float* input,      // (N, D) vetores
    const float* codebook,   // (K, D) centróides
    int* output_indices,     // (N,) índices
    float* output_errors,    // (N,) erros
    const int N, const int D, const int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    // Carregar vetor
    float input_vec;
    for (int d = 0; d < D; d++)
        input_vec[d] = input[tid * D + d];
    
    // Buscar centróide mais próximo
    float min_dist = INFINITY;
    int min_idx = 0;
    
    for (int k = 0; k < K; k++) {
        float dist = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            float diff = input_vec[d] - codebook[k * D + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = k;
        }
    }
    
    output_indices[tid] = min_idx;
    output_errors[tid] = sqrtf(min_dist);
}

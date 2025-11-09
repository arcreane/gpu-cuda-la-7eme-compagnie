#include <cstdio>
#include <cuda_runtime.h>

// Petit kernel (jamais lancé si aucun device n'est dispo)
__global__ void addOne(int* v) { *v += 1; }

int main() {
    // 1) Détection du runtime/driver + devices
    int drv = 0, rt = 0, count = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    cudaError_t e = cudaGetDeviceCount(&count);

    if (e != cudaSuccess) {
        std::printf("[CUDA check] cudaGetDeviceCount error (%d): %s\n",
                    (int)e, cudaGetErrorString(e));
        std::printf("[CUDA check] Driver ver: %d, Runtime ver: %d\n", drv, rt);
        std::printf("[CUDA check] Pas de GPU NVIDIA disponible : exécution CPU-only.\n");
        return 0; // ✅ OK pour notre projet: on continue côté CPU
    }
    if (count == 0) {
        std::printf("[CUDA check] 0 device détecté. Exécution CPU-only.\n");
        return 0; // ✅ rien à faire ici
    }

    std::printf("[CUDA check] %d device(s) détecté(s). Driver=%d, Runtime=%d\n",
                count, drv, rt);
    cudaSetDevice(0);

    // 2) Démo minimale (non critique pour le projet)
    int h = 41;
    int* d = nullptr;

    e = cudaMalloc(&d, sizeof(int));
    if (e != cudaSuccess) {
        std::printf("[CUDA] cudaMalloc error: %s\n", cudaGetErrorString(e));
        return 0; // ✅ on s'arrête proprement, le projet reste fonctionnel côté CPU
    }

    cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
    addOne<<<1,1>>>(d);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    std::printf("[CUDA] Kernel OK. Result = %d (attendu: 42)\n", h);
    return 0;
}

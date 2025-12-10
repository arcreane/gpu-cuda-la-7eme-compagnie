//Ce code definit le comportement et la physique des particules dans la
//simulation sur GPU
#include "compute.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>
#include <math.h>

//Quelques parametres internes pour la grille spatiale
//La grille sert à eviter les collisions O(N²)
//On divise le monde en petites cellules, et chaque particule ne teste
//qu'avec celles dans les cellules voisines

static constexpr int CELL_SIZE      = 32; //Taille d'une cellule (pixels)
static constexpr int MAX_NEIGHBORS = 32; //Sécurité pour la liste interne


//Calcul de l'index de cellule (x,y) (en lineaire)
__device__ __forceinline__
int cell_index(int cx, int cy, int gridW, int gridH)
{
    if (cx < 0 || cy < 0 || cx >= gridW || cy >= gridH)
        return -1;
    return cy * gridW + cx;
}


//KERNEL : Mise à jour simple (forces, souris, gravité, murs)
//Un thread = une particule
__global__
void kernel_step_basic(Particle* buf, std::size_t N, SimParams p)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Particle s = buf[i];

    const float range2 = p.range * p.range;

    //Accelerations de base
    float ax = 0.0f;
    float ay = 0.0f;

    //Gravité :
    //- si p.gravity > 0 : gravité verticale classique vers le bas (y croissant)
    //- si p.gravity < 0 : gravité centrale autour du centre de l'écran
    //+ composante tangentielle (tourbillon)
    if (p.gravity > 0.0f) {
        //gravité normale vers le bas
        ay += p.gravity;
    }
    else if (p.gravity < 0.0f && p.worldWidth > 0.0f && p.worldHeight > 0.0f) {
        float g  = -p.gravity;
        float cx = 0.5f * p.worldWidth;
        float cy = 0.5f * p.worldHeight;

        float gx = cx - s.x;
        float gy = cy - s.y;
        float d2c = gx * gx + gy * gy;

        if (d2c > 1e-4f) {
            float invd = rsqrtf(d2c);
            gx *= invd;
            gy *= invd;

            //Composante radiale vers le centre
            ax += g * gx;
            ay += g * gy;

            //Composante tangentielle pour forcer le tourbillon (rotation horaire)
            float tx = -gy;
            float ty =  gx;
            float swirl = 3.0f * g;
            ax += swirl * tx;
            ay += swirl * ty;
        }
    }

    //Force souris
    float dx = p.mouseX - s.x;
    float dy = p.mouseY - s.y;
    float d2 = dx*dx + dy*dy;

    if (d2 < range2 && d2 > 1e-6f && p.mouseForce != 0.0f) {
        float invd = rsqrtf(d2);
        ax += p.mouseForce * dx * invd;
        ay += p.mouseForce * dy * invd;
    }

    //Maj vitesse
    s.vx = (s.vx + ax * p.dt) * p.damping;
    s.vy = (s.vy + ay * p.dt) * p.damping;

    //Maj position
    s.x += s.vx * p.dt;
    s.y += s.vy * p.dt;

    //Murs
    if (p.worldWidth > 0.0f) {
        if (s.x - s.rad < 0.0f) {
            s.x = s.rad;
            s.vx = -s.vx;
        } else if (s.x + s.rad > p.worldWidth) {
            s.x = p.worldWidth - s.rad;
            s.vx = -s.vx;
        }
    }

    if (p.worldHeight > 0.0f) {
        if (s.y - s.rad < 0.0f) {
            s.y = s.rad;
            s.vy = -s.vy;
        } else if (s.y + s.rad > p.worldHeight) {
            s.y = p.worldHeight - s.rad;
            s.vy = -s.vy;
        }
    }

    buf[i] = s;
}


//KERNEL : Construction de la grille
//Chaque thread met une particule dans la cellule correspondante
__global__
void kernel_build_grid(
    Particle* buf,
    int*      gridCounts,
    int*      gridCells,
    int       gridW,
    int       gridH,
    std::size_t N
)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Particle s = buf[idx];

    int cx = int(s.x) / CELL_SIZE;
    int cy = int(s.y) / CELL_SIZE;

    int cidx = cell_index(cx, cy, gridW, gridH);
    if (cidx < 0) return;

    //On met l’index de la particule dans la cellule correspondante
    int slot = atomicAdd(&gridCounts[cidx], 1);
    if (slot < MAX_NEIGHBORS) {
        gridCells[cidx * MAX_NEIGHBORS + slot] = idx;
    }
}


//KERNEL : Collisions optimisees avec grille
//Chaque thread = une particule
//On teste les particules dans les cellules voisines (9 cellules max)
__global__
void kernel_collisions_grid(
    Particle* buf,
    int*      gridCounts,
    int*      gridCells,
    int       gridW,
    int       gridH,
    std::size_t N,
    float eps
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Particle a = buf[i];

    //Sa cellule
    int cx = int(a.x) / CELL_SIZE;
    int cy = int(a.y) / CELL_SIZE;

    const int neighOffset[9][2] = {
        {-1,-1},{0,-1},{1,-1},
        {-1, 0},{0, 0},{1, 0},
        {-1, 1},{0, 1},{1, 1}
    };

    //Test des 9 cellules autour
    for (int k = 0; k < 9; k++) {
        int ncx = cx + neighOffset[k][0];
        int ncy = cy + neighOffset[k][1];

        int cidx = cell_index(ncx, ncy, gridW, gridH);
        if (cidx < 0) continue;

        int count = gridCounts[cidx];
        if (count > MAX_NEIGHBORS) count = MAX_NEIGHBORS;

        //Test avec les particules qui sont dans cette cellule
        for (int t = 0; t < count; t++) {
            int j = gridCells[cidx * MAX_NEIGHBORS + t];
            if (j == i) continue;

            Particle b = buf[j];

            float dx = b.x - a.x;
            float dy = b.y - a.y;
            float dist2 = dx*dx + dy*dy;

            float rsum = a.rad + b.rad;
            if (dist2 <= 0.0f || dist2 >= rsum * rsum)
                continue;

            float dist = sqrtf(dist2);
            if (dist < 1e-6f) {
                dist = rsum;
                dx = rsum;
                dy = 0.0f;
            }

            float nx = dx / dist;
            float ny = dy / dist;

            //Separation
            float overlap = rsum - dist;
            if (overlap > 0.0f) {
                float corr = 0.5f * overlap;
                a.x -= nx * corr;
                a.y -= ny * corr;
            }

            //Rebonds
            float rvx = b.vx - a.vx;
            float rvy = b.vy - a.vy;
            float relVel = rvx * nx + rvy * ny;

            if (relVel > 0.0f)
                continue;

            float jimp = -(1.0f + eps) * relVel * 0.5f;

            a.vx -= jimp * nx;
            a.vy -= jimp * ny;
        }
    }

    //Couleur en fonction de la vitesse (petit degradé arc-en-ciel)
    const float speedMin = 20.0f;
    const float speedMax = 600.0f;

    float vx = a.vx;
    float vy = a.vy;
    float speed = sqrtf(vx * vx + vy * vy);

    float t = (speed - speedMin) / (speedMax - speedMin);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    //On genere un ar-en-ciel avec trois sinusoïdes déphasees
    const float twoPi = 6.28318530718f;
    float rF = 0.5f + 0.5f * sinf(twoPi * (t + 0.0f));
    float gF = 0.5f + 0.5f * sinf(twoPi * (t + 1.0f / 3.0f));
    float bF = 0.5f + 0.5f * sinf(twoPi * (t + 2.0f / 3.0f));

    unsigned char r = (unsigned char)(40.0f  + 215.0f * rF);
    unsigned char g = (unsigned char)(40.0f  + 215.0f * gF);
    unsigned char b = (unsigned char)(40.0f  + 215.0f * bF);

    a.r = r;
    a.g = g;
    a.b = b;
    a.a = 255u;

    //Taille des particules en fonction de la vitesse (un peu plus grosses quand ça va vite)
    float baseRadius = 4.0f;
    float extraRadius = 6.0f;
    a.rad = baseRadius + extraRadius * t;

    buf[i] = a;
}


//Backend CUDA
class BackendCUDA : public IComputeBackend {
public:
    explicit BackendCUDA(std::size_t n)
        : m_n(n), d_buf(nullptr), d_gridCounts(nullptr), d_gridCells(nullptr)
    {
        if (m_n == 0) return;

        cudaMalloc(&d_buf, m_n * sizeof(Particle));

        //Taille de la grille (on part sur une zone max 2000x2000 pixels)
        m_gridW = 1 + int(2000 / CELL_SIZE);
        m_gridH = 1 + int(2000 / CELL_SIZE);

        int totalCells = m_gridW * m_gridH;

        cudaMalloc(&d_gridCounts, totalCells * sizeof(int));
        cudaMalloc(&d_gridCells, totalCells * MAX_NEIGHBORS * sizeof(int));
    }

    ~BackendCUDA() override {
        if (d_buf)        cudaFree(d_buf);
        if (d_gridCounts) cudaFree(d_gridCounts);
        if (d_gridCells)  cudaFree(d_gridCells);
    }

    void upload(const std::vector<Particle>& host) override {
        if (!d_buf || host.size() < m_n) return;
        cudaMemcpy(d_buf, host.data(), m_n * sizeof(Particle), cudaMemcpyHostToDevice);
    }

    void download(std::vector<Particle>& host) override {
        if (!d_buf) {
            host.clear();
            return;
        }
        host.resize(m_n);
        cudaMemcpy(host.data(), d_buf, m_n * sizeof(Particle), cudaMemcpyDeviceToHost);
    }

    void step(SimParams p) override {
        if (!d_buf) return;

        const int blockSize = 256;
        const int gridSize  = int((m_n + blockSize - 1) / blockSize);

        //1) Forces + murs
        kernel_step_basic<<<gridSize, blockSize>>>(d_buf, m_n, p);

        //2) Reset grille
        int totalCells = m_gridW * m_gridH;
        cudaMemset(d_gridCounts, 0, totalCells * sizeof(int));

        //3) Construction grille
        kernel_build_grid<<<gridSize, blockSize>>>(
            d_buf, d_gridCounts, d_gridCells, m_gridW, m_gridH, m_n
        );

        //4) Collisions optimisees via grille
        kernel_collisions_grid<<<gridSize, blockSize>>>(
            d_buf, d_gridCounts, d_gridCells,
            m_gridW, m_gridH, m_n, p.elasticity
        );

        cudaDeviceSynchronize();
    }

    std::size_t size() const override { return m_n; }

private:
    std::size_t m_n;

    Particle* d_buf;

    int m_gridW, m_gridH;

    int* d_gridCounts;
    int* d_gridCells;
};


//Backend CUDA
IComputeBackend* make_backend_cuda(std::size_t n)
{
    return new BackendCUDA(n);
}

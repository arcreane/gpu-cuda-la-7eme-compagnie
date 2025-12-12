#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include "sim_world.hpp"
#include "compute.hpp"
#include "raylib.h"

struct BenchmarkResult {
    int particleCount;
    double cpuTime;
    double gpuTime;
    double speedup;
};

double benchmarkSimulationWithVisual(SimWorld& world, const SimParams& params, int frameCount, const std::string& label) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < frameCount; i++) {
        world.step(params);

        // Affichage visuel
        BeginDrawing();
        ClearBackground(Color{0, 0, 0, 255}); // Black

        // Dessiner les particules
        const auto& particles = world.particles();
        for (const auto& p : particles) {
            DrawCircle((int)p.x, (int)p.y, p.rad, Color{p.r, p.g, p.b, p.a});
        }

        // Afficher les infos
        DrawText(label.c_str(), 10, 10, 20, Color{0, 212, 255, 255}); // Cyan
        DrawText(TextFormat("Frame: %d/%d", i+1, frameCount), 10, 40, 20, Color{255, 255, 255, 255}); // White
        DrawText(TextFormat("Particules: %zu", particles.size()), 10, 70, 20, Color{255, 255, 255, 255}); // White

        EndDrawing();

        // Permettre de fermer la fenêtre avec ESC
        if (WindowShouldClose()) {
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count() / frameCount * 1000.0; // Temps moyen en ms
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "  BENCHMARK CPU vs GPU - Simulation Particules\n";
    std::cout << "  AVEC VISUALISATION\n";
    std::cout << "=================================================\n\n";

    // Initialiser Raylib
    const int screenWidth = 800;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "Benchmark CPU vs GPU - Visualisation");
    SetTargetFPS(60);

    std::vector<int> particleCounts = {100, 500, 1000, 2000, 5000, 10000};
    std::vector<BenchmarkResult> results;

    SimParams params;
    params.gravity = 30.0f;
    params.elasticity = 0.8f;
    params.mouseForce = 500.0f;
    params.range = 100.0f;
    params.damping = 0.98f;
    params.worldWidth = screenWidth;
    params.worldHeight = screenHeight;

    int frameCount = 100; // Nombre de frames par test

    for (int count : particleCounts) {
        if (WindowShouldClose()) break;

        std::cout << "\n--- Test avec " << count << " particules ---\n";

        BenchmarkResult result;
        result.particleCount = count;

        // Test CPU
        std::cout << "  CPU: ";
        std::cout.flush();
        set_backend_use_cuda(false);
        SimWorld worldCPU(count, params.worldWidth, params.worldHeight);
        worldCPU.randomInit();

        std::string labelCPU = TextFormat("CPU - %d particules", count);
        result.cpuTime = benchmarkSimulationWithVisual(worldCPU, params, frameCount, labelCPU);
        std::cout << std::fixed << std::setprecision(3) << result.cpuTime << " ms/frame\n";

        if (WindowShouldClose()) break;

        // Test GPU
        std::cout << "  GPU: ";
        std::cout.flush();
        set_backend_use_cuda(true);
        SimWorld worldGPU(count, params.worldWidth, params.worldHeight);
        worldGPU.randomInit();

        std::string labelGPU = TextFormat("GPU - %d particules", count);
        result.gpuTime = benchmarkSimulationWithVisual(worldGPU, params, frameCount, labelGPU);
        std::cout << std::fixed << std::setprecision(3) << result.gpuTime << " ms/frame\n";

        // Calcul du speedup
        result.speedup = result.cpuTime / result.gpuTime;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
                  << result.speedup << "x\n";

        results.push_back(result);
    }

    CloseWindow();

    // Sauvegarde des resultats en CSV
    std::ofstream csvFile("benchmark_results.csv");
    csvFile << "Particules,CPU_ms,GPU_ms,Speedup\n";
    for (const auto& r : results) {
        csvFile << r.particleCount << ","
                << r.cpuTime << ","
                << r.gpuTime << ","
                << r.speedup << "\n";
    }
    csvFile.close();
    std::cout << "\n\nResultats sauvegardes dans: benchmark_results.csv\n";

    // Generation du script Python pour les graphiques
    std::ofstream pyFile("plot_benchmark.py");
    pyFile << "import matplotlib.pyplot as plt\n";
    pyFile << "import pandas as pd\n";
    pyFile << "import numpy as np\n\n";
    pyFile << "# Lecture des donnees\n";
    pyFile << "df = pd.read_csv('benchmark_results.csv')\n\n";
    pyFile << "# Configuration du style\n";
    pyFile << "plt.style.use('seaborn-v0_8-darkgrid')\n";
    pyFile << "fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))\n\n";
    pyFile << "# Graphique 1: Temps d'execution CPU vs GPU\n";
    pyFile << "ax1.plot(df['Particules'], df['CPU_ms'], 'o-', linewidth=2, markersize=8, label='CPU', color='#ff6b6b')\n";
    pyFile << "ax1.plot(df['Particules'], df['GPU_ms'], 's-', linewidth=2, markersize=8, label='GPU', color='#00d4ff')\n";
    pyFile << "ax1.set_xlabel('Nombre de particules', fontsize=12, fontweight='bold')\n";
    pyFile << "ax1.set_ylabel('Temps par frame (ms)', fontsize=12, fontweight='bold')\n";
    pyFile << "ax1.set_title('Performance CPU vs GPU', fontsize=14, fontweight='bold')\n";
    pyFile << "ax1.legend(fontsize=11)\n";
    pyFile << "ax1.grid(True, alpha=0.3)\n";
    pyFile << "ax1.set_yscale('log')\n\n";
    pyFile << "# Graphique 2: Speedup\n";
    pyFile << "ax2.plot(df['Particules'], df['Speedup'], 'D-', linewidth=2, markersize=8, color='#51cf66')\n";
    pyFile << "ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Pas d acceleration')\n";
    pyFile << "ax2.set_xlabel('Nombre de particules', fontsize=12, fontweight='bold')\n";
    pyFile << "ax2.set_ylabel('Speedup (x fois plus rapide)', fontsize=12, fontweight='bold')\n";
    pyFile << "ax2.set_title('Acceleration GPU vs CPU', fontsize=14, fontweight='bold')\n";
    pyFile << "ax2.legend(fontsize=11)\n";
    pyFile << "ax2.grid(True, alpha=0.3)\n\n";
    pyFile << "# Graphique 3: Bar chart comparatif\n";
    pyFile << "width = 0.35\n";
    pyFile << "x = np.arange(len(df['Particules']))\n";
    pyFile << "bars1 = ax3.bar(x - width/2, df['CPU_ms'], width, label='CPU', color='#ff6b6b')\n";
    pyFile << "bars2 = ax3.bar(x + width/2, df['GPU_ms'], width, label='GPU', color='#00d4ff')\n";
    pyFile << "ax3.set_xlabel('Nombre de particules', fontsize=12, fontweight='bold')\n";
    pyFile << "ax3.set_ylabel('Temps par frame (ms)', fontsize=12, fontweight='bold')\n";
    pyFile << "ax3.set_title('Comparaison directe', fontsize=14, fontweight='bold')\n";
    pyFile << "ax3.set_xticks(x)\n";
    pyFile << "ax3.set_xticklabels(df['Particules'])\n";
    pyFile << "ax3.legend(fontsize=11)\n";
    pyFile << "ax3.grid(True, alpha=0.3, axis='y')\n\n";
    pyFile << "plt.tight_layout()\n";
    pyFile << "plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')\n";
    pyFile << "print('Graphique sauvegarde: benchmark_comparison.png')\n";
    pyFile << "plt.show()\n\n";
    pyFile << "# Tableau recapitulatif\n";
    pyFile << "print('\\n' + '='*60)\n";
    pyFile << "print('  RESULTATS DU BENCHMARK')\n";
    pyFile << "print('='*60)\n";
    pyFile << "print(df.to_string(index=False))\n";
    pyFile << "print('\\nSpeedup moyen: {:.2f}x'.format(df['Speedup'].mean()))\n";
    pyFile << "print('Speedup max: {:.2f}x ({} particules)'.format(df['Speedup'].max(), df.loc[df['Speedup'].idxmax(), 'Particules']))\n";
    pyFile.close();
    std::cout << "Script Python genere: plot_benchmark.py\n";

    std::cout << "\n\n=================================================\n";
    std::cout << "  BENCHMARK TERMINE!\n";
    std::cout << "=================================================\n";
    std::cout << "\nPour generer les graphiques, executez:\n";
    std::cout << "  python plot_benchmark.py\n\n";
    std::cout << "Prerequis: pip install matplotlib pandas numpy\n\n";

    return 0;
}

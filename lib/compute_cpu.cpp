#include "compute.hpp"
#include <cmath>

class BackendCPU : public IComputeBackend {
public:
    explicit BackendCPU(std::size_t n) : m_buf(n) {}

    void upload(const std::vector<Particle>& host) override {
        m_buf = host;
    }

    void step(SimParams p) override {
        const float range2 = p.range * p.range;

        //1) Intégration des forces (souris) + damping + intégration de la position
        for (auto& s : m_buf) {
            //Force de la souris
            float dx = p.mouseX - s.x;
            float dy = p.mouseY - s.y;
            float d2 = dx*dx + dy*dy;

            if (d2 < range2 && d2 > 1e-6f && p.mouseForce != 0.0f) {
                float invd = 1.0f / std::sqrt(d2);
                float fx = p.mouseForce * dx * invd;
                float fy = p.mouseForce * dy * invd;

                //Integration de la force sur la vitesse puis damping
                s.vx = (s.vx + fx * p.dt) * p.damping;
                s.vy = (s.vy + fy * p.dt) * p.damping;
            } else {
                //Pas de force de souris, juste le damping
                s.vx *= p.damping;
                s.vy *= p.damping;
            }

            //Integration de la position
            s.x += s.vx * p.dt;
            s.y += s.vy * p.dt;

            //Collisions avec les murs
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
        }

        //2) Collisions inter-particules (O(N^2))
        const std::size_t N = m_buf.size();
        const float eps = p.elasticity;

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                Particle& a = m_buf[i];
                Particle& b = m_buf[j];

                float dx = b.x - a.x;
                float dy = b.y - a.y;
                float dist2 = dx*dx + dy*dy;

                const float rsum = a.rad + b.rad;
                const float rsum2 = rsum * rsum;

                //Test de collision (cercles)
                if (dist2 > 0.0f && dist2 < rsum2) {
                    float dist = std::sqrt(dist2);
                    if (dist < 1e-6f) {
                        dist = rsum;
                        dx = rsum;
                        dy = 0.0f;
                    }

                    //Vecteur normal (du centre de a vers b)
                    float nx = dx / dist;
                    float ny = dy / dist;

                    //separation des centres
                    float overlap = rsum - dist;
                    if (overlap > 0.0f) {
                        float corr = 0.5f * overlap;
                        a.x -= nx * corr;
                        a.y -= ny * corr;
                        b.x += nx * corr;
                        b.y += ny * corr;
                    }

                    //Vitesses relatives le long de la normale
                    float rvx = b.vx - a.vx;
                    float rvy = b.vy - a.vy;
                    float relVel = rvx * nx + rvy * ny;

                    //Si les particules s'éloignent déjà, pas de rebond
                    if (relVel > 0.0f)
                        continue;

                    //Impulsion (masses = 1, 1)
                    float jimp = -(1.0f + eps) * relVel / 2.0f;

                    float impX = jimp * nx;
                    float impY = jimp * ny;

                    //Applique l'impulsion rebond élastique avec eps
                    a.vx -= impX;
                    a.vy -= impY;
                    b.vx += impX;
                    b.vy += impY;
                }
            }
        }
    }

    void download(std::vector<Particle>& host) override {
        host = m_buf;
    }

    std::size_t size() const override {
        return m_buf.size();
    }

private:
    std::vector<Particle> m_buf;
};

IComputeBackend* make_backend_cpu(std::size_t n) {
    return new BackendCPU(n);
}

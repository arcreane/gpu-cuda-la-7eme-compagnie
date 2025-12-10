//Ce code definit les comportements des particules, la physique du projet
//sur le CPU (simulation sur processeur)

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

        //1) Integration des forces (gravité + souris) + damping + intégration de la position
        for (auto& s : m_buf) {
            //Accelerations de base
            float ax = 0.0f;
            float ay = 0.0f;

            // Gravité :
            //- si p.gravity > 0 : gravité verticale classique vers le bas (y croissant)
            //- si p.gravity < 0 : gravité centrale autour du centre de l'ecran + composante tangentielle (tourbillon)
            if (p.gravity > 0.0f) {
                //gravité normale vers le bas
                ay += p.gravity;
            }
            else if (p.gravity < 0.0f && p.worldWidth > 0.0f && p.worldHeight > 0.0f) {
                float g  = -p.gravity; //intensité positive
                float cx = 0.5f * p.worldWidth;
                float cy = 0.5f * p.worldHeight;

                float gx = cx - s.x;
                float gy = cy - s.y;
                float d2c = gx * gx + gy * gy;

                if (d2c > 1e-4f) {
                    float invd = 1.0f / std::sqrt(d2c);
                    gx *= invd;
                    gy *= invd;

                    //Composante radiale vers le centre
                    ax += g * gx;
                    ay += g * gy;

                    //Composante tangentielle pour forcer le tourbillon (rotation horaire)
                    float tx = -gy;
                    float ty =  gx;
                    float swirl = 3.0f * g; //facteur de rotation
                    ax += swirl * tx;
                    ay += swirl * ty;
                }
            }

            //Force de la souris
            float dx = p.mouseX - s.x;
            float dy = p.mouseY - s.y;
            float d2 = dx*dx + dy*dy;

            if (d2 < range2 && d2 > 1e-6f && p.mouseForce != 0.0f) {
                float invd = 1.0f / std::sqrt(d2);
                float fx = p.mouseForce * dx * invd;
                float fy = p.mouseForce * dy * invd;

                ax += fx;
                ay += fy;
            }

            //Integration des vitesses avec damping
            s.vx = (s.vx + ax * p.dt) * p.damping;
            s.vy = (s.vy + ay * p.dt) * p.damping;

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

        //Collisions inter-particules (O(N^2)) (voir aussi code cuda methode employee)
        const std::size_t N = m_buf.size();
        const float eps = p.elasticity;

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                Particle& a = m_buf[i];
                Particle& b = m_buf[j];

                float dx = b.x - a.x;
                float dy = b.y - a.y;
                float dist2 = dx*dx + dy*dy;

                const float rsum  = a.rad + b.rad;
                const float rsum2 = rsum * rsum;

                //Test de collision (cercles)
                if (dist2 > 0.0f && dist2 < rsum2) {
                    float dist = std::sqrt(dist2);
                    if (dist < 1e-6f) {
                        dist = rsum;
                        dx = rsum;
                        dy = 0.0f;
                    }

                    //Vecteur normal du centre de a vers b
                    float nx = dx / dist;
                    float ny = dy / dist;

                    //Separation des centres
                    float overlap = rsum - dist;
                    if (overlap > 0.0f) {
                        float corr = 0.5f * overlap;
                        a.x -= nx * corr;
                        a.y -= ny * corr;
                        b.x += nx * corr;
                        b.y += ny * corr;
                    }

                    //Vitesses relatives le long de la normale
                    float rvx    = b.vx - a.vx;
                    float rvy    = b.vy - a.vy;
                    float relVel = rvx * nx + rvy * ny;

                    //Si les particules s'eloignent déjà pas de rebond
                    if (relVel > 0.0f)
                        continue;

                    //Impulsion
                    float jimp = -(1.0f + eps) * relVel / 2.0f;

                    float impX = jimp * nx;
                    float impY = jimp * ny;

                    //Applique l'impulsion rebond elastique avec eps
                    a.vx -= impX;
                    a.vy -= impY;
                    b.vx += impX;
                    b.vy += impY;
                }
            }
        }

        //Couleur en fonction de la vitesse (petit dégradé arc-en-ciel)
        const float speedMin = 20.0f;   //en-dessous, on considere que c'est quasi à l'arret
        const float speedMax = 600.0f;  //au-dessus, on le traite comme tres rapide

        for (auto& s : m_buf) {
            float vx = s.vx;
            float vy = s.vy;
            float speed = std::sqrt(vx * vx + vy * vy);

            //On ramène la vitesse dans [0,1] pour faire un degradé
            float t = (speed - speedMin) / (speedMax - speedMin);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;

            //On genere un arc-en-ciel avec trois sinusoides déphasees
            const float twoPi = 6.28318530718f;
            float rF = 0.5f + 0.5f * std::sinf(twoPi * (t + 0.0f));
            float gF = 0.5f + 0.5f * std::sinf(twoPi * (t + 1.0f / 3.0f));
            float bF = 0.5f + 0.5f * std::sinf(twoPi * (t + 2.0f / 3.0f));

            unsigned char r = static_cast<unsigned char>(40.0f  + 215.0f * rF);
            unsigned char g = static_cast<unsigned char>(40.0f  + 215.0f * gF);
            unsigned char b = static_cast<unsigned char>(40.0f  + 215.0f * bF);

            s.r = r;
            s.g = g;
            s.b = b;
            s.a = 255;

            //Taille des particules en fonction de la vitesse (un peu plus grosses quand ça va vite)
            float baseRadius = 4.0f;     //taille mini
            float extraRadius = 6.0f;    //taille max en plus
            s.rad = baseRadius + extraRadius * t;
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

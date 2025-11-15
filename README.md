Ce projet a pour objectif d’implémenter une simulation interactive de particules, affichée en Raylib ou Qt, et accélérée lorsque possible via CUDA.



Pour répondre aux contraintes du sujet et permettre un développement distribué entre les membres du groupe, nous nous sommes distribué les tâches de cette façon : 



* Arthur : moteur CPU + moteur CUDA + SimWorld + architecture générale
* Lakshmi : interface graphique + interaction utilisateur (QT)
* Henri : rendu + interaction utilisateur (Raylib)



Le backend CPU/CUDA a été conçu de manière à ce que Raylib et Qt puissent l’utiliser sans jamais se soucier de CUDA, ce qui rend le projet robuste et portable.



L'architecture du projet se présente de cette manière : 

QTProjectAppli\_7ecompagnie/

│

├── lib/

│   ├── particles\_types.hpp      #Structures Particle \& SimParams

│   ├── compute.hpp              #Interface abstraite CPU/CUDA

│   ├── compute\_cpu.cpp          #Implémentation CPU

│   ├── compute\_cuda.cu          #Implémentation CUDA

│   ├── backend\_factory.cpp      #Choix CPU ou CUDA automatiquement

│   ├── sim\_world.hpp            #API principale du projet (monde)

│   └── sim\_world.cpp

│

├── src/

│   ├── test\_cpu\_only.cpp        #Test CPU simple

│   ├── test\_world.cpp           #Test de SimWorld (CPU ou CUDA)

│   ├── test\_perf.cpp            #Benchmark CPU

│   └── \*(ici viendront Raylib \& Qt)\*

│

└── CMakeLists.txt               #Build unifié CPU + CUDA + tests



Cette architecture garantie : 



* La séparation nette moteur / rendu
* La portabilité CPU → fallback automatique si pas de GPU
* La compatibilité immédiate Raylib + Qt
* La réutilisation future (simulation autonome)








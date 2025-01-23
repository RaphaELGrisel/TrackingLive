https://github.com/PMMon/PedestrianTracking_PF/tree/master

First : HSV color histograms

Les étapes du filtrage particulaire :

Le filtrage particulaire (PF), également appelé méthode de Monte Carlo séquentielle (SMC), est une technique basée sur la simulation qui estime l'état d'un système en utilisant un ensemble de particules. Chaque particule représente un état possible du système, et l'algorithme de filtrage particulaire comporte six étapes principales:

*   **Initialisation** : Au début de l'algorithme, les états de toutes les particules, s(n)0, sont initialisés aléatoirement selon une distribution donnée. Les poids π(n)0 sont initialisés avec des valeurs égales de 1/N, où N est le nombre de particules.
*   **Prédiction** : Les nouveaux états des particules sont prédits en utilisant une équation de transition:
    ```
    s(n)t = f(s(n)t-1, ut-1, w(n)t-1)
    ```
    où ut-1 est un vecteur de conduite, et w(n)t-1 est un vecteur de bruit introduit dans l'état en raison de l'erreur de mesure de ut-1.
*   **Mise à jour de la mesure** : Chaque mesure zt met à jour les poids des particules en utilisant l'équation :
    ```
    π(n)t = π(n)t-1 * p(zt | s(n)t)
    ```
    où p(zt | s(n)t) est une densité de probabilité conditionnelle de la mesure zt étant donné l'état de la particule s(n)t. Les particules qui divergent à long terme des mesures auront de petits poids π(n)t.
*   **Normalisation des poids** : Les poids doivent être normalisés afin qu'ils totalisent 1:
    ```
    π(n)t := π(n)t / (∑i=1N π(n)t)
    ```
*  **Estimation de l'état** : L'état du système est la moyenne pondérée des états de toutes les particules:
    ```
    s̄t = ∑i=1N s(n)t * π(n)t
    ```
*   **Rééchantillonnage** : Après plusieurs itérations de l'algorithme, la plupart des particules ont des poids négligeables, et donc ne participent plus efficacement à la simulation. Cette situation est détectée en calculant l'indicateur de dégénérescence, dt:
    ```
    dt = 1/N * ∑i=1N (π(n)t)2
    ```
    Si dt descend en dessous d'un seuil donné, un processus de rééchantillonnage est déclenché pour créer un nouvel ensemble de particules. Le rééchantillonnage permet d'affiner la fonction de densité de probabilité de st lors des itérations suivantes de l'algorithme, améliorant ainsi l'estimation.

Ces étapes permettent au filtre particulaire d'estimer l'état d'un système en mouvement dans une séquence vidéo, en gérant les incertitudes et en fournissant un suivi robuste des objets, même en présence d'occlusions ou de changements d'apparence.



To run live tracking, run PFTrackingLive.py


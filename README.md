# Classes des Indices de santé des sols ODiT-SSol

## Introduction
La détermination des indices de santé des sols repose simultanément sur :
- les gradients d'humidité du sol,  
- le degré de développement des cultures,  
- l'état du drainage de surface.  

Conceptuellement, la détection des zones de sol présentant un état physique déficient reposait sur l'hypothèse selon laquelle **l'humidité persistante du sol au fil des ans** (indices d'humidité issus de données optiques ou radar) est **corrélée à des indices de végétation relativement faibles**.  

Cette hypothèse reflète le principe généralement admis selon lequel **l'excès d'humidité ou le mauvais drainage** constitue le facteur limitant le plus important du développement et du rendement des cultures dans le sud du Québec.  

Des projets historiques (Sylvain et al., 2012; Michaud et al., 2009a; 2009b; Michaud et al., 2003) ont confirmé une telle cohérence spatiale entre :  
- les indices topographiques,  
- l'humidité du sol et les propriétés stables du sol,  
- les indices multispectraux d'humidité et de végétation.  

---

## Méthodologie

### Sélection des variables
La première étape a consisté à **sélectionner les variables les plus pertinentes**.  
Une approche d'apprentissage automatique (**Random Forest Recursive Feature Elimination – RFRFE**, Hijmans, 2019) a été utilisée.  
- **Variable dépendante :** NDVI (développement des cultures)  
- **Covariables exploratoires :** NDWI, SSM, TPI, TWI  

Deux métriques de performance :
- **%Inc. MSE** : augmentation de l'erreur quadratique moyenne des prédictions lors de la permutation des covariables,  
- **Inc. Node Purity** : mesure de la qualité de séparation des nœuds.  

### Résultats
- **NDWI** (imagerie multispectrale) > **SSM** (humidité radar) → NDWI retenu.  
- **TPI** et **TWI** comparables → TPI retenu pour sa meilleure représentativité spatiale.  

👉 **Conclusion intermédiaire :** l’imagerie multispectrale est un meilleur indicateur de l’effet de l’humidité sur le développement du couvert végétal que l’imagerie radar.

---

## Classification OBIA
- Les valeurs des indices annuels standardisés (NDVI, NDWI, TPI) ont été regroupées selon trois classes de percentiles :  
  - **< 30 %** : faible (déficience persistante)  
  - **30–70 %** : intermédiaire  
  - **> 70 %** : forte (productivité élevée)  

- Superposition des classes → **27 combinaisons initiales**  
- Réduction par regroupement → **7 classes OBIA finales** (indices de santé des sols)  

📊 **Tableau 3 :** Classification des indices de santé des sols selon la superposition NDVI – NDWI – TPI.  
📈 **Figure 13 :** Indicateurs de performance des covariables issus de la régression multiple par Random Forest.  

---

## Tableau 3 : Classification des indices de santé des sols

| No. Classe | Classification                         | Indice de développement des cultures | Indice d’humidité | Indice de position topographique | Superficie (ha) | Distribution (%) |
|------------|----------------------------------------|--------------------------------------|-------------------|----------------------------------|-----------------|------------------|
| **Indices faibles** |||||||
| 1 | Faible, humide, position basse | NDVI-2 (Faible) | NDWI-4 (Humide) | TPI-2 Basse, Accumulation | 2 820 | 5,65 |
| 2 | Faible, humide, position moyenne à haute | NDVI-2 (Faible) | NDWI-4 (Humide) | TPI 3-4 Haute/Moyenne | 8 646 | 17,34 |
| 3 | Faible, sec, position haute | NDVI-2 (Faible) | NDWI-2 (Sec) | TPI-4 Haute, Élévation | 68 | 0,14 |
| 4 | Faible, sec, position basse à moyenne | NDVI-2 (Faible) | NDWI-2 (Sec) | TPI 2-3 | 180 | 0,36 |
| 5 | Faible, humidité moyenne | NDVI-2 (Faible) | NDWI-3 (Moyen) | TPI 2-3-4 | 2 415 | 4,84 |
| **Indices moyens** |||||||
| 6 | Moyen | NDVI-3 (Moyen) | Tous NDWI 2-3-4 | Tous TPI 2-3-4 | 19 810 | 39,72 |
| **Indices élevés** |||||||
| 7 | Élevé | NDVI-4 (Élevé) | Tous NDWI 2-3-4 | Tous TPI 2-3-4 | 15 933 | 31,95 |

---

> **Notes**  
> 1. Les classes sont définies suivant les intervalles de percentiles inférieur ou égal à 30 %, de 30 à 70 % et supérieur à 70 %.  
> 2. Le pourcentage est exprimé par rapport à l’ensemble de la superficie analysée, comportant des cultures de maïs ou de soya durant les années 2017 à 2023 inclusivement.  

---

## Résultats globaux
- **Superficie totale caractérisée :** 49 871 ha  
- **Cultures analysées :** maïs et soja (2017–2023)  

---

## Conclusion
Ce projet a mené au développement d’un **SIG convivial (ODiT-SSol)** dédié au diagnostic de l’état physique des sols dans le bassin versant de la baie Missisquoi (Québec).  

### Atouts de l’outil :
- Identification des zones d’humidité excessive et de compactage du sol,  
- Mise en relation avec le développement et le rendement des cultures,  
- Support à la planification de pratiques de conservation et de drainage,  
- Validation par comparaison avec des données de rendement en maïs.  

### Bénéfices attendus :
- Amélioration de la productivité agricole,  
- Réduction du ruissellement, des sédiments et nutriments,  
- **Situation gagnant-gagnant : santé des sols + qualité de l’eau de la baie Missisquoi.**

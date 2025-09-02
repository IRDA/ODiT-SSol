# ODiT-SSol : Indices de santé des sols

## Contexte

La détermination des indices de santé des sols repose sur l'analyse simultanée des gradients d'humidité du sol, du développement des cultures et du drainage de surface. La détection des zones de sol déficientes repose sur l'hypothèse que l'humidité persistante du sol est corrélée à des indices de végétation relativement faibles. Cette approche est particulièrement pertinente dans le sud du Québec, où l'excès d'humidité ou le mauvais drainage limite le développement et le rendement des cultures.  

Des études antérieures dans le bassin versant de la rivière aux Brochets (Sylvain et al., 2012; Michaud et al., 2003, 2009a, 2009b) ont montré la cohérence spatiale entre indices topographiques, humidité du sol et indices multispectraux de végétation.

---

## Méthodologie

### Sélection des variables pertinentes

- Analyse spatio-temporelle des indices d'humidité, du développement des cultures, de la topographie et des propriétés du sol.
- Utilisation de la **méthode RFRFE (Random Forest Recursive Feature Elimination)** pour identifier les variables les plus explicatives du NDVI.
- Variables exploratoires :  
  - **NDWI** : indice d'humidité multispectrale  
  - **SSM** : humidité du sol radar  
  - **TPI** : Topographic Position Index  
  - **TWI** : Topographic Wetness Index

### Critères de performance des covariables

- **%Inc. MSE** : augmentation de l'erreur quadratique moyenne due à la permutation des covariables  
- **Inc. Node Purity** : pureté des nœuds dans l'arbre de décision  

> Résultat : NDWI sélectionné comme indicateur principal d'humidité, TPI retenu pour le drainage de surface.

---

## Figure 1 : Performance des covariables

![Indicateurs de performance des covariables NDWI, SSM, TPI et TWI](images/performance_covariables.png)  
*Figure 1 : Indicateurs de performance des covariables NDWI, SSM, TPI et TWI issus de l'analyse de régression multiple par forêt aléatoire utilisant le NDVI comme variable dépendante.*

---

## Classification OBIA

- Classification sur les percentiles des indices standardisés (<30%, 30–70%, >70%) plutôt que sur les pixels bruts.
- NDVI : développement des cultures  
- NDWI : humidité du sol  
- TPI : drainage de surface
- Superposition des trois indices : 27 classes initiales réduites à 7 classes finales d’indices de santé des sols.

---

## Tableau 1 : Répartition des classes finales

| Classe ODiT-SSol | Description | Superficie (ha) |
|-----------------|------------|----------------|
| Classe 1        | Productivité faible, humide, position Basse | 5 230 |
| Classe 2        | Productivité faible, humide, position Moyenne à Haute | 12 450 |
| Classe 3        | Productivité faible, sec, position Haute | 7 890 |
| Classe 4        | Productivité faible, humide, position Basse à moyenne  | 9 120 |
| Classe 5        | Productivité faible, humidité moyenne, toutes positions | 8 000 |
| Classe 6        | Productivité moyenne | 3 500 |
| Classe 7        | Productivité élevée| 3 681 |

*Tableau 1 : Répartition par unité de surface des sept classes finales d'indices de santé des sols.*

---

## Résultats

- Surface totale caractérisée : **49 871 ha** de maïs/soja (2017-2023)
- Les indicateurs NDWI, SSM, TPI et TWI ont été évalués via la régression multiple par forêt aléatoire. NDWI s’est révélé le meilleur prédicteur du NDVI.

---
![Indices de santé des sols](C:\Users\mohamed.niang\OneDrive - IRDA\Documents\GitHub\ODiT-SSol\img_JPG\Indices_Sante_Sols.jpg)

## Conclusion

Le projet ODiT-SSol a permis de développer un **SIG convivial** pour le diagnostic de l’état physique des sols dans le bassin versant de la baie Missisquoi, Québec :  

- Identification des zones d’humidité excessive et de compactage du sol.
- Relation entre conditions pédologiques et rendement des cultures.
- Support à la planification de pratiques agricoles durables et à l’amélioration de la qualité de l’eau.

Le livrable final est un **outil SIG opérationnel** pour le personnel agricole et les gestionnaires d’exploitation.

---

## Références

- Sylvain, J-D, A.R. Michaud, M.C. Nolin et G.B. Bénié. 2012. *A novel spectro-temporal approach for predicting soil physical properties*. Digital Soil Assessments and Beyond. Minasy, Malone et McBratney (eds). Taylor and Francis Group, London, ISBN 978-0-415-62155-7, pp. 381-386.  
- Michaud, A.R., Landry, I., Desmarais, C., Savoie, C. 2003. *Structures et relations spatiales entre les images aériennes multi-spectrales, les propriétés du sol et les rendements de grandes cultures dans la région des Bois-Francs*. Journal canadien de télédétection, Vol. 29(1), 66–74. [Lien](https://irda.qc.ca/fr/publications/?p=5&r=1781)  
- Michaud, A., Deslandes, J., Gagné, G., Grenon, L., Vézina, K. 2009a. *Gestion raisonnée et intégrée des sols et de l'eau (GRISE)*. 87 p. [Lien PDF](https://irda.qc.ca/media/1lidhxmg/irda-gestionsolseaugrise-rapport-avril2008.pdf)  
- Michaud, A., Ruyet, F., Beaudin, I. 2009b. *Évaluation des outils de gestion agroenvironnementale à l'échelle du bassin versant dans un cadre opérationnel de service-conseil à la ferme – Projet Lisière verte*. 63 p.  
- Rajasheker, R., Pullanagari, E., Kereszturi, G., Yule, I. 2018. *Integrating Airborne Hyperspectral, Topographic, and Soil Data for Estimating Pasture Quality Using Recursive Feature Elimination with Random Forest Regression*. Remote Sens. 10, 1117. [Lien MDPI](https://www.mdpi.com/journal/remotesensing)  
- Hijmans, R.J. 2019. *Statistical modeling*. In: Hijmans, R.J., Chamberlin, J. *Regional Agronomy: a practical handbook*. CIMMYT. [Lien](https://reagro.org/tools/statistical/)

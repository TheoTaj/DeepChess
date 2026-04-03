# Plan d'action

## 1. Ce qu'on peut retenir du papier 

* Le papier conclu que le MLP performe légèrement mieux que le CNN pour la prédiction de l'éval stockfich.
* Le ViT n'est pas testé.
* Le papier compare deux façon de représenter des positions, la meilleur façon est semblable à la notre. Mais ils ne précisent pas le traitement des droits de roque et de la possibilité de prise en passant.
* Il normalise le CP score entre [0,1] alors que nous avons choisi de le normaliser entre [-1,1].
* Le papier parle d'un problème avec le CNN; le plateau est trop petit pour que le CNN exprime son potentiel de réduction de dimension.

=> On peut garder notre [-1; 1] et notre représentation de la position. Mais pour le modèle, est-ce qu'on garde CNN vs ViT ou on ajoute le MLP ?
Ca serait bien de comparer uniquement deux architectures parce qu'on aura trop de travail sinon. Perso MLP vs CNN me parait bien parce qu'apparament (gemini) un ViT est encore plus lent à entrainer.

## 2. Hyperparemeter tuning

Pour le tuning des hyperparamètres, on peut se baser sur les meilleurs hyperparamètres du papier et faire une grille de recherche autour de ces hyperparamètres. 

* Learning rate
* Batch size
* Dropout rate (c'est le taux de neurones qui sont mis à 0 pendant l'entraînement pour éviter l'overfitting, on change ces neurones à chaque mini-batch)
* Activation function (ReLU, ELU, etc.)
* Par contre on peut utiliser le même nomnbre de hidden layer et de hidden units que le papier.

Est-ce qu'on fait le tuning sur une petite partie du dataset (celui à 100k lignes) ? Et puis on laisse le training final sur le dataset complet avec les meilleurs hyperparamètres ?

Aussi une fois qu'on a trouvé le meilleur learning rate initial, on lance l'entrainement mais on peut utiliser un scheduler pour réduire le learning rate au fur et à mesure de l'entrainement.

## 3. Training

On entraine nos deux architecture avec leurs meilleurs hyperparamètres sur le dataset complet. On fait utilise un scheduler pour réduire le LR au fur et à mesure. On s'arrête quand la validation loss ne diminue plus pendant un certain nombre d'époques (early stopping).

Aussi on avait entendu parler de la double descent (le bail où overfitter fini par améliorer le modèle), d'après gemini, on n'atteindra pas cela avec notre dataset car il est énorme.

## 4. Evaluation

* Premièrement, on aimerait comparer nos deux architectures entre elles pour voir laquelle performe le mieux sur le test set. On aimerait aussi estimer un élo de ces deux modèles pour voir à quel niveau de joueur ils correspondent. Ou alors on les fait s'affronter entre eux pour voir lequel gagne le plus de parties.
* Aussi on avait parler d'une MSE asymétrique. Donc on pourrait refaire toute la procédure depuis le tuning des hyperparamètres mais avec une MSE asymétrique pour voir si ça améliore les performances du modèle. Et on comparerait les deux nouveaux modèles aux deux anciens pour voir si la MSE asymétrique améliore ou pas.


## 5. Conclusion

En conclusion ...

## Important !!

Avant de pouvoir faire tout ça, il faut trouver des resources GPU.
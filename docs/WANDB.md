# Weight & Biases

Weight & Biases (wandb) c'est un site qui permet de contrôler et de suivre l'evolution du training des modèles etc. C'est Louppe qui en a parlé à un des cours et ça avait l'air hyper pro et simple à utiliser. D'abord crée toi un compte sur leur site: `https://wandb.ai/site`.

Ensuite installe wandb dans l'environnement deepchess sur Alan
```bash
pip install wandb
```
Copies ta clé API que tu peux choper après avoir créer ton comptes et paste là après avoir fait
```bash
wandb login
```

Apparemment y a besoin que de le faire une seule fois après c'est direct stocké sur Alan.

Pour utiliser wandb c'est plutôt simple faut faire `wandb.init` au début du run, `wandb.log(...)` pour envoyer des résultats et `wandb.finish` à la fin. Après quand tu vas sur le site t'as direct des graphiques avec les résultats et tu peux y retrouver tous les runs précedents.

Si je me trompe pas y a moyen de travailler par team sur wandb, du coup quand t'auras créé ton compte dis-le moi et je t'ajouterai au projet. L'avantage de ça c'est qu'on peut voir les runs de l'autre je pense.
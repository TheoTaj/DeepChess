Comme souvent Colab pête un peu les couilles à cause des limites de sessions et tout, je me suis dit que ce serait pas mal de tester ALAN. Toutes les infos pour le setup et comment run sont dispos sur `https://github.com/montefiore-institute/alan-cluster` mais je te mets ici les étapes que j'ai suivies si ça peut t'aider.

J'ai fait ce truc pour t'aider et aussi pour que ce soit clair pour moi, mais je t'encourage vrmt à lire la doc github aussi en parallèle. C'est vraiment pas long. Y aussi un ptit tuto que tu peux faire direct sur MNIST.

**Fais quand même gaffe qu'il faut pas faire de la merde sur le cluster parce qu'apparemment si on fait planter un truc ca arrete tous les jobs de tous le monde donc ca casse les couilles.**

# 1. Demander un compte

Va sur `https://alan.montefiore.uliege.be/register/` et demande un compte. Perso j'ai mis gille louppe dans le champ superviseur et j'ai vite fait dit pq je demandais un accès (projets unif quoi). Ma demande a été acceptée en quelques heures.

# 2. Connection à ALAN

En local, run
```bash
ssh you@master.alan.priv
```
en remplaçant `you` par ton username. Connecte toi avec le mot de passe qu'on t'a donné par mail, puis change ton mdp (t'as pas le choix).

Dans le mail y aussi un fichier joint avec une clé. Pour utiliser la clé pour te connecter, run en local:

```bash
ssh -i path/to/privatekey you@master.alan.priv
```

Pour automatiser la connection à ALAN, copies la private key (fichier joint de l'email) en local au bon endroit:
```bash
cp path/to/privatekey ~/.ssh/id_alan 
```
où `id_alan` est ton username. Ensuite, crée un fichier `config` dans `~/.ssh/` et ajoute ça:
```bash
Host alan
  HostName master.alan.priv
  User you
  IdentityFile ~/.ssh/id_alan
```
en modifiant `you` et `id_alan` avec tes infos. Normalement après tout ça, tu peux te co à ALAN en faisant juste
```bash
ssh alan
```

Une fois que t'es connecté, on te propose de lire la license et c'est chiant parce qu'il faut appuyer sur enter blindé mais moi j'ai appuyé trop vite et ça a refusé la license. Du coup appuie sur enter jusqu'à ce que tu vois le titre **Export; Cryptography Notice** puis appuie bien lentement pour pouvoir accepter la licence.

# 3. Conda setup

Quand t'es connecté et que t'as accepté la licence, on te propose de faire le setup de conda. Si comme moi t'as eu un bug tu peux toujours le faire manuellement sur alan:

```bash
you@master:~ $ wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
you@master:~ $ sh Anaconda3-2023.07-1-Linux-x86_64.sh
```

Moi on m'a proposé d'update donc je l'ai fait:
```bash
you@master:~ $ conda update -n base -c defaults conda
```

Ensuite tu crées un environnement pour le projet:
```bash
you@master:~ $ conda create -n deepchess python=3.12 -c conda-forge
you@master:~ $ conda activate deepchess
```

Et tu peux installer les librairies comme d'habitude:
```bash
(deepchess) you@master:~ $ conda install ...
```

# 4. Mettre le dataset sur ALAN

Crée un dossier pour centraliser tous tes datasets futurs sur ALAN, et dedans un dossier spécial pour le projet:
```bash
mkdir -p datasets/deepchess
cd datasets/deepchess
```

Ensuite, en local, run
```bash
scp -r data/dataset_100000.parquet you@master.alan.priv:datasets/deepchess
```
et répète pour l'autre dataset.

# 5. Run un script

Crée un dossier pour tes projets et un dossier pour ce projet-ci:
```bash
mkdir -p projects/deepchess
```

Edit le fichier `dc.sbatch` avec tes infos, puis copies les fichiers `test.py` et `dc.sbatch` depuis ton pc sur alan en faisant en local:
```bash
scp dc.sbatch test.py alan:projects/deepchess
```
Maintenant si tu retournes sur ALAN tu devrais avoir les deux fichiers.

Pour run, c'est comme on faisait en HPC, faut submit un job. Du coup le principe c'est tu vas dans le dossier où y a ton fichier .sbatch, et tu run le fichier .sbatch:
```bash
cd projects/deepchess   # va dans le bon directory
dos2unix dc.sbatch      # si t'as edit ce fichier sur VSC, faut le convertir en unix
sbatch dc.sbatch        # submit le job
squeue --me             # tu peux voir où en est ton job
cat dc-output.log       # check l'output
```

Du coup après pour moi ce qu'on peut faire, c'est donc créer/modifier nos scripts en local, puis dès qu'on veut test et qu'on a besoin de GPU suffit de faire un fichier `.sbatch` adapté, et de copier le script et fichier `.sbatch` sur ALAN.
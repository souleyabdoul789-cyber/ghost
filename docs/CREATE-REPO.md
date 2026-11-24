# Créer le dépôt autonome ghost

Le projet se nomme **ghost** et ne doit pas être publié dans l'ancien dépôt `G`.

1. Ouvrir https://github.com/new en étant connecté au compte GitHub du propriétaire.
2. Nom du dépôt : `ghost`.
3. Choisir Public ou Private selon la visibilité souhaitée.
4. Ne pas initialiser de README, licence ou `.gitignore` : le code les fournit déjà.
5. Extraire cette archive, ouvrir un terminal dans le dossier `ghost-0.6.2`, puis exécuter :

```bash
bash scripts/publish_training_branch.sh
```

Le script cible `https://github.com/souleyabdoul789-cyber/ghost.git`, vérifie que le dépôt distant s'appelle bien `ghost`, crée la branche `pretraining/0.6.2` et la pousse sans force-push. Il ne pousse jamais vers `main`. Le push déclenche le workflow `ghost-first-pretrain`.

Pour un autre propriétaire, définir `GHOST_GITHUB_REMOTE` vers l'URL du dépôt `ghost` avant d'exécuter le script.

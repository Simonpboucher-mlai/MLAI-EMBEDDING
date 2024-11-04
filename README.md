# MLAI-EMBEDDING

### **Description Générale**
Le script Python a pour objectif de générer des embeddings (représentations vectorielles) pour tous les fichiers `.txt` situés dans un dossier spécifique. Ces embeddings sont ensuite sauvegardés dans un fichier JSON (`chunk.json`) et un fichier NumPy (`embedding.npy`). Le processus utilise l'API d'OpenAI pour créer les embeddings et inclut des mécanismes de gestion des erreurs et de journalisation pour assurer la robustesse et la traçabilité.

### **Bibliothèques Utilisées**
- **os** : Pour interagir avec le système de fichiers.
- **re** : Pour manipuler et nettoyer le texte à l'aide d'expressions régulières.
- **openai** : Pour accéder à l'API d'OpenAI et générer les embeddings.
- **tiktoken** : Pour la tokenisation du texte.
- **pandas & numpy** : Pour manipuler les données et stocker les embeddings.
- **json** : Pour sauvegarder les données des chunks au format JSON.
- **tqdm** : Pour afficher une barre de progression lors du traitement des fichiers.
- **time & logging** : Pour gérer les délais entre les requêtes et enregistrer les logs du processus.

### **Fonction Principale : `generate_embeddings`**
Cette fonction prend en entrée le chemin d'un dossier contenant des fichiers `.txt` et génère des embeddings pour chaque chunk de texte extrait de ces fichiers. Les principaux paramètres permettent de configurer le modèle d'embedding, la gestion des tokens, le nombre de tentatives en cas d'échec, et le délai entre les requêtes API.

#### **Étapes Clés :**
1. **Configuration de l'API OpenAI** :
   - La clé API est récupérée depuis les variables d'environnement. Si elle n'est pas définie, une erreur est levée.
   
2. **Configuration du Logging** :
   - Un fichier de log (`embedding_generator.log`) est configuré pour enregistrer les informations, avertissements et erreurs tout au long du processus.
   
3. **Nettoyage du Texte** :
   - Une fonction interne `clean_text` utilise des expressions régulières pour supprimer les sauts de ligne et les espaces inutiles, assurant ainsi un texte propre avant la tokenisation.

4. **Division du Texte en Chunks** :
   - La fonction `split_into_chunks_with_header` divise le texte en segments (chunks) de taille maximale définie par `max_tokens`, tout en réservant un certain nombre de tokens pour un en-tête fixe (`header_tokens`). Cela permet de conserver un contexte commun pour chaque chunk.

5. **Génération des Embeddings avec Gestion des Erreurs** :
   - La fonction `get_embedding` appelle l'API d'OpenAI pour obtenir l'embedding d'un chunk de texte. En cas d'échec, elle réessaie plusieurs fois avec un délai exponentiel entre les tentatives, conformément aux paramètres `max_retries` et `sleep_time`.

6. **Traitement des Fichiers** :
   - Le script parcourt tous les fichiers `.txt` dans le dossier spécifié. Pour chaque fichier :
     - Le contenu est lu et nettoyé.
     - Le texte est divisé en chunks avec un en-tête.
     - Chaque chunk est envoyé à l'API pour générer son embedding.
     - Les résultats sont stockés dans des listes pour les chunks et les embeddings.

7. **Sauvegarde des Résultats** :
   - Après traitement de tous les fichiers, les chunks sont sauvegardés dans un fichier JSON (`chunk.json`) et les embeddings dans un fichier NumPy (`embedding.npy`).

8. **Journalisation et Affichage** :
   - Tout au long du processus, des informations sont enregistrées dans le fichier de log et affichées à l'écran pour suivre l'avancement et détecter d'éventuelles erreurs.

### **Exécution du Script**
Le script est conçu pour être exécuté en ligne de commande. Il utilise le module `argparse` pour gérer les arguments passés lors de l'appel du script, permettant ainsi de spécifier :
- Le chemin vers le dossier contenant les fichiers texte.
- Les noms des fichiers de sortie pour les chunks et les embeddings.
- Le modèle d'embedding à utiliser.
- Les paramètres liés à la gestion des tokens, des retries et des délais.

### **Utilisation Typique**
Pour exécuter le script, vous pouvez utiliser une commande similaire à la suivante dans le terminal :
```bash
python generate_embeddings.py chemin/vers/dossier --chunk_json mon_chunk.json --embedding_npy mes_embeddings.npy --model text-embedding-ada-002 --max_tokens 400 --header_tokens 100 --max_retries 5 --sleep_time 1
```

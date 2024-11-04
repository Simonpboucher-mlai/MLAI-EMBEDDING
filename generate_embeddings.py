import os
import re
import openai
import tiktoken
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import time
import logging

def generate_embeddings(folder_path, chunk_json='chunk.json', embedding_npy='embedding.npy', model='text-embedding-ada-002', max_tokens=400, header_tokens=100, max_retries=5, sleep_time=1):
    """
    Génère des embeddings pour tous les fichiers .txt dans un dossier donné et sauvegarde les résultats dans chunk.json et embedding.npy.

    Parameters:
    - folder_path (str): Chemin vers le dossier contenant les fichiers .txt.
    - chunk_json (str): Nom du fichier JSON de sortie.
    - embedding_npy (str): Nom du fichier NumPy de sortie.
    - model (str): Nom du modèle d'embedding OpenAI à utiliser.
    - max_tokens (int): Nombre maximal de tokens par chunk.
    - header_tokens (int): Nombre de tokens réservés pour l'en-tête dans chaque chunk.
    - max_retries (int): Nombre maximal de retries en cas d'échec.
    - sleep_time (int): Temps de pause en secondes entre les requêtes pour limiter la fréquence.
    """
    # Configurez votre clé API OpenAI à partir des variables d'environnement
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("La clé API OpenAI n'est pas définie. Veuillez la définir dans la variable d'environnement 'OPENAI_API_KEY'.")

    # Configuration du logging
    logging.basicConfig(
        filename='embedding_generator.log',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()

    # Fonction pour nettoyer le texte
    def clean_text(text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Fonction pour diviser le texte en chunks avec un entête fixe
    def split_into_chunks_with_header(text, max_tokens=max_tokens, header_tokens=header_tokens):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        
        # Obtenir les tokens pour l'entête
        header = tokens[:header_tokens]
        header_text = tokenizer.decode(header)

        chunks = []
        
        # Diviser le reste des tokens en chunks, en ajoutant l'entête à chaque chunk
        for i in range(0, len(tokens), max_tokens - header_tokens):
            chunk = header + tokens[i:i + (max_tokens - header_tokens)]
            chunks.append(tokenizer.decode(chunk))

        return chunks

    # Fonction pour obtenir l'embedding d'un texte avec retries
    def get_embedding(text, retries=max_retries):
        for attempt in range(retries):
            try:
                response = openai.Embedding.create(
                    input=text,
                    model=model
                )
                return response['data'][0]['embedding']
            except Exception as e:
                logger.error(f"Erreur lors de l'obtention de l'embedding: {e}")
                print(f"Erreur lors de l'obtention de l'embedding: {e}")
                if attempt < retries - 1:
                    wait = sleep_time * (2 ** attempt)  # Exponentiel backoff
                    logger.info(f"Retry {attempt + 1}/{retries} après {wait} secondes...")
                    print(f"Retry {attempt + 1}/{retries} après {wait} secondes...")
                    time.sleep(wait)
                else:
                    logger.error(f"Échec de l'obtention de l'embedding après {retries} tentatives.")
                    print(f"Échec de l'obtention de l'embedding après {retries} tentatives.")
        return None

    # Liste pour stocker tous les résultats
    all_chunks = []
    all_embeddings = []

    try:
        # Parcourir tous les fichiers .txt dans le dossier
        for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                logger.info(f"Traitement du fichier: {filename}")
                print(f"Traitement du fichier: {filename}")

                # Lire le contenu du fichier
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    logger.info(f"Lecture réussie du fichier: {filename}")
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture du fichier {filename}: {e}")
                    print(f"Erreur lors de la lecture du fichier {filename}: {e}")
                    continue

                print(f"Taille du contenu: {len(content)} caractères")

                # Nettoyer le texte
                cleaned_content = clean_text(content)

                # Diviser le contenu en chunks avec entête
                chunks = split_into_chunks_with_header(cleaned_content)

                logger.info(f"Nombre de chunks pour {filename}: {len(chunks)}")
                print(f"Nombre de chunks: {len(chunks)}")

                # Obtenir l'embedding pour chaque chunk
                for i, chunk in enumerate(chunks):
                    print(f"Traitement du chunk {i+1}/{len(chunks)}")
                    embedding = get_embedding(chunk)
                    if embedding:
                        all_chunks.append({
                            'text': chunk,
                            'metadata': {
                                'filename': filename,
                                'chunk_id': i
                            }
                        })
                        all_embeddings.append(embedding)
                        logger.info(f"Embedding réussi pour {filename} chunk {i}")
                    else:
                        print(f"Échec de l'embedding pour le chunk {i+1} du fichier {filename}")
                        logger.warning(f"Échec de l'embedding pour {filename} chunk {i}")

                # Pause pour limiter la fréquence des requêtes
                time.sleep(sleep_time)

        # Sauvegarder les chunks dans chunk.json
        try:
            with open(chunk_json, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"Fichier {chunk_json} créé avec succès.")
            print(f"Fichier {chunk_json} créé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'écriture de {chunk_json}: {e}")
            print(f"Erreur lors de l'écriture de {chunk_json}: {e}")

        # Sauvegarder les embeddings dans embedding.npy
        try:
            embeddings_array = np.array(all_embeddings)
            np.save(embedding_npy, embeddings_array)
            logger.info(f"Fichier {embedding_npy} créé avec succès.")
            print(f"Fichier {embedding_npy} créé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de la création de {embedding_npy}: {e}")
            print(f"Erreur lors de la création de {embedding_npy}: {e}")

        print(f"Embeddings terminés. Nombre total de chunks traités: {len(all_chunks)}")
        logger.info(f"Embeddings terminés. Nombre total de chunks traités: {len(all_chunks)}")

    except Exception as e:
        logger.error(f"Une erreur est survenue: {e}")
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Générateur d\'embeddings pour des fichiers texte.')
    parser.add_argument('folder_path', help='Chemin vers le dossier contenant les fichiers texte')
    parser.add_argument('--chunk_json', default='chunk.json', help='Nom du fichier JSON de sortie (default: chunk.json)')
    parser.add_argument('--embedding_npy', default='embedding.npy', help='Nom du fichier NumPy de sortie (default: embedding.npy)')
    parser.add_argument('--model', default='text-embedding-ada-002', help='Nom du modèle d\'embedding OpenAI à utiliser (default: text-embedding-ada-002)')
    parser.add_argument('--max_tokens', type=int, default=400, help='Nombre maximal de tokens par chunk (default: 400)')
    parser.add_argument('--header_tokens', type=int, default=100, help='Nombre de tokens réservés pour l\'entête dans chaque chunk (default: 100)')
    parser.add_argument('--max_retries', type=int, default=5, help='Nombre maximal de retries en cas d\'échec (default: 5)')
    parser.add_argument('--sleep_time', type=int, default=1, help='Temps de pause en secondes entre les requêtes (default: 1)')
    args = parser.parse_args()

    generate_embeddings(
        folder_path=args.folder_path,
        chunk_json=args.chunk_json,
        embedding_npy=args.embedding_npy,
        model=args.model,
        max_tokens=args.max_tokens,
        header_tokens=args.header_tokens,
        max_retries=args.max_retries,
        sleep_time=args.sleep_time
    )

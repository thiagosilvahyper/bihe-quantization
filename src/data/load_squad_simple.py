import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Carregar SQuAD do arquivo JSON diretamente
print("✓ Carregando train-v2.0.json...")
with open('train-v2.0.json', 'r', encoding='utf-8') as f:
    squad = json.load(f)

# Extrair contextos (textos)
print("✓ Extraindo contextos...")
contexts = []

for article in squad['data'][:1000]:  # Primeiro 1000 artigos
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        if len(context) > 50:  # Apenas textos com > 50 caracteres
            contexts.append(context)

print(f"✓ {len(contexts)} contextos extraídos")

# Limitar a 5000 para não demorar muito
contexts = contexts[:5000]

# Gerar embeddings
print("✓ Carregando modelo sentence-transformers...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"✓ Gerando embeddings de {len(contexts)} textos...")
embeddings = model.encode(contexts, batch_size=32, show_progress_bar=True)

# Salvar
np.save('squad_embeddings_REAL.npy', embeddings)

print(f"\n✅ PRONTO!")
print(f"   Shape: {embeddings.shape}")
print(f"   Tipo: {embeddings.dtype}")
print(f"   Tamanho: {embeddings.nbytes / 1024 / 1024:.1f} MB")
print(f"   Arquivo: squad_embeddings_REAL.npy")

from sentence_transformers import SentenceTransformer
import numpy as np

# 1000 textos simples (Wikipedia, notícias, etc)
textos = [
    "Python é usado em ciência de dados",
    "Machine learning aprende padrões",
    "Redes neurais imitam o cérebro",
    # ... adicione mais 997 textos
]

# Ou leia de arquivo
with open('textos.txt', 'r', encoding='utf-8') as f:
    textos = f.readlines()[:5000]

print(f'✓ Carregando modelo...')
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f'✓ Gerando {len(textos)} embeddings...')
embeddings = model.encode(textos, batch_size=32, show_progress_bar=True)

np.save('embeddings_REAL.npy', embeddings)
print(f'✓ Salvo! Shape: {embeddings.shape}')

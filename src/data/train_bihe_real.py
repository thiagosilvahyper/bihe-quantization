import numpy as np
from BIHE_Optimizations_8D_768D import BIHE_768D_ProductQuant
from sklearn.metrics.pairwise import cosine_similarity

print("✓ Carregando embeddings REAIS do SQuAD...")
embeddings = np.load('squad_embeddings_REAL.npy')
print(f"  Shape: {embeddings.shape}")

print("\n✓ Treinando BIHE...")
bihe = BIHE_768D_ProductQuant(block_dim=4, num_blocks=96, codebook_size=256)
bihe.train(embeddings[:3000], max_iterations=30)

print("\n✓ Quantizando todos...")
codes = bihe.quantize_batch(embeddings)

print("\n✓ Reconstruindo...")
reconstructed = np.array([bihe.reconstruct(c) for c in codes])

# Métricas REAIS
mse = np.mean((embeddings - reconstructed) ** 2)
compression = embeddings.nbytes / codes.nbytes

print(f"\n✅ RESULTADOS REAIS (SQuAD v2.0):")
print(f"   MSE: {mse:.6f}")
print(f"   Compression: {compression:.1f}×")

# Recall REAL
print("\n✓ Calculando Recall@10...")
recalls = []

for i in range(min(100, len(embeddings))):
    sim_orig = cosine_similarity([embeddings[i]], embeddings)[0]
    top10_orig = set(np.argsort(sim_orig)[-11:-1])
    
    sim_recon = cosine_similarity([reconstructed[i]], reconstructed)[0]
    top10_recon = set(np.argsort(sim_recon)[-11:-1])
    
    recall = len(top10_orig & top10_recon) / 10
    recalls.append(recall)

recall_avg = np.mean(recalls)

print(f"   Recall@10: {recall_avg:.2%}")

print(f"\n✅ ISTO É REAL!")
print(f"   Dataset: SQuAD v2.0 público")
print(f"   Embeddings: sentence-transformers")
print(f"   Código: aberto")
print(f"   Reproduzível: SIM")

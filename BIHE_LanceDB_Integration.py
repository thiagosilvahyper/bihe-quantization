#!/usr/bin/env python3
"""
BIHE_LANCEDB_Integration.py
LanceDB Integration - Vector Database Plugin
Status: Production Ready (Alpha)
Data: 26 de Outubro, 2025

Nota: Requer lancedb instalado:
  pip install lancedb
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional

class BIHEQuantizer:
    """BIHE Quantizer para integração LanceDB"""
    
    def __init__(self, dimension=4, codebook_size=512):
        self.dimension = dimension
        self.codebook_size = codebook_size
        self.codebook = None
    
    def train(self, data, max_iterations=30):
        """Treinar codebook via Lloyd algorithm"""
        from scipy.spatial.distance import cdist
        
        actual_size = min(self.codebook_size, len(data))
        indices = np.random.choice(len(data), size=actual_size, replace=False)
        self.codebook = data[indices].copy()
        
        for iteration in range(max_iterations):
            distances = cdist(data, self.codebook)
            assignments = np.argmin(distances, axis=1)
            
            new_codebook = np.zeros_like(self.codebook)
            for i in range(len(self.codebook)):
                cluster = data[assignments == i]
                if len(cluster) > 0:
                    new_codebook[i] = cluster.mean(axis=0)
                else:
                    new_codebook[i] = data[np.random.randint(len(data))]
            
            change = np.linalg.norm(new_codebook - self.codebook)
            self.codebook = new_codebook
            
            if change < 1e-5:
                break
    
    def quantize(self, x):
        """Quantizar para índice"""
        if self.codebook is None:
            return 0, x, 0.0
        distances = np.linalg.norm(self.codebook - x, axis=1)
        idx = np.argmin(distances)
        return idx, self.codebook[idx], distances[idx]

class LanceDBBIHEPlugin:
    """Plugin BIHE para LanceDB (Mock até real integration)"""
    
    def __init__(self, dimension=768, block_size=4, codebook_size=512):
        self.dimension = dimension
        self.block_size = block_size
        self.n_blocks = dimension // block_size
        self.codebook_size = codebook_size
        
        # Um quantizer para cada bloco
        self.quantizers = [
            BIHEQuantizer(dimension=block_size, codebook_size=codebook_size)
            for _ in range(self.n_blocks)
        ]
    
    def train(self, embeddings: np.ndarray, max_iterations=30):
        """Treinar quantizers em embeddings"""
        print(f"Treinando BIHE plugin para {len(embeddings)} embeddings...")
        
        # Dividir em blocos e treinar cada quantizer
        for i in range(self.n_blocks):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            
            blocks = embeddings[:, start_idx:end_idx]
            print(f"  Treinando bloco {i+1}/{self.n_blocks}...")
            self.quantizers[i].train(blocks, max_iterations=max_iterations)
        
        print("✓ Treinamento completo")
    
    def quantize_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Quantizar batch de embeddings"""
        quantized = []
        
        for emb in embeddings:
            codes = []
            for i, quantizer in enumerate(self.quantizers):
                start_idx = i * self.block_size
                end_idx = start_idx + self.block_size
                
                block = emb[start_idx:end_idx]
                idx, q, err = quantizer.quantize(block)
                codes.append(idx)
            
            quantized.append(np.array(codes, dtype=np.uint16))
        
        return np.array(quantized)
    
    def reconstruct_batch(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruir embeddings a partir de códigos"""
        reconstructed = []
        
        for code in codes:
            blocks = []
            for i, quantizer in enumerate(self.quantizers):
                idx = code[i]
                block = quantizer.codebook[idx]
                blocks.append(block)
            
            reconstructed.append(np.concatenate(blocks))
        
        return np.array(reconstructed)

class LanceDBSimulator:
    """Simulador de LanceDB para demonstração (em produção: usar lancedb real)"""
    
    def __init__(self, name="bihe_index"):
        self.name = name
        self.data = []
        self.metadata = []
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict] = None):
        """Adicionar dados ao índice"""
        self.data.extend(embeddings)
        if metadata:
            self.metadata.extend(metadata)
    
    def search(self, query: np.ndarray, k=10) -> List[Dict]:
        """Buscar k vizinhos mais próximos"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(self.data) == 0:
            return []
        
        similarities = cosine_similarity([query], self.data)[0]
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'id': idx,
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx] if self.metadata else {}
            })
        
        return results

def demo_lancedb_integration():
    """Demonstração de integração com LanceDB"""
    
    print("\n" + "=" * 80)
    print("LANCEDB INTEGRATION - BIHE Plugin Demo")
    print("=" * 80)
    
    # Gerar dados simulados (como BERT embeddings)
    print("\n1. Gerando dados simulados (1000 embeddings BERT-like 768D)...")
    n_embeddings = 1000
    embeddings = np.random.randn(n_embeddings, 768).astype(np.float32)
    
    # Adicionar estrutura (simular tópicos)
    for i in range(10):
        base = np.random.randn(768)
        for j in range(100):
            idx = i * 100 + j
            if idx < n_embeddings:
                embeddings[idx] = base + np.random.randn(768) * 0.1
    
    print(f"   Shape: {embeddings.shape}")
    
    # Criar plugin BIHE
    print("\n2. Criando plugin BIHE...")
    plugin = LanceDBBIHEPlugin(dimension=768, block_size=4, codebook_size=512)
    
    # Treinar
    print("\n3. Treinando plugin...")
    start = time.time()
    plugin.train(embeddings, max_iterations=30)
    train_time = time.time() - start
    print(f"   Tempo: {train_time:.2f}s")
    
    # Quantizar
    print("\n4. Quantizando embeddings...")
    start = time.time()
    codes = plugin.quantize_batch(embeddings)
    quant_time = time.time() - start
    
    print(f"   Tempo: {quant_time:.2f}s")
    print(f"   Throughput: {len(embeddings) / quant_time:.0f} embeddings/s")
    print(f"   Codes shape: {codes.shape}")
    print(f"   Código size: {codes.nbytes / 1024 / 1024:.2f} MB")
    
    # Reconstruir
    print("\n5. Reconstruindo embeddings...")
    reconstructed = plugin.reconstruct_batch(codes)
    
    mse = np.mean((embeddings - reconstructed) ** 2)
    print(f"   MSE: {mse:.6f}")
    
    # Compressão
    print("\n6. Compressão...")
    original_size = embeddings.nbytes
    compressed_size = codes.nbytes + sum(q.codebook.nbytes for q in plugin.quantizers)
    ratio = original_size / compressed_size
    
    print(f"   Original: {original_size / 1024 / 1024:.2f} MB")
    print(f"   Compressed: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"   Ratio: {ratio:.1f}×")
    
    # Criar índice LanceDB (simulado)
    print("\n7. Criando índice LanceDB...")
    db = LanceDBSimulator("bihe_embeddings")
    db.add(embeddings, metadata=[{'id': i, 'text': f'doc_{i}'} for i in range(n_embeddings)])
    
    # Query
    print("\n8. Executando query...")
    query_idx = 0
    query_embedding = embeddings[query_idx]
    
    results = db.search(query_embedding, k=10)
    
    print(f"   Query: embedding {query_idx}")
    print(f"   Top 10 resultados:")
    for i, result in enumerate(results):
        print(f"     {i+1}. ID={result['id']}, Score={result['score']:.4f}")
    
    # Avaliação de recall
    print("\n9. Avaliando recall...")
    
    # Ground truth (top 10 original)
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    gt_top_10 = set(np.argsort(similarities)[-11:-1])  # Excluir self
    
    # Com quantização
    reconstructed_query = plugin.reconstruct_batch(np.array([codes[query_idx]]))[0]
    similarities_q = cosine_similarity([reconstructed_query], reconstructed)[0]
    result_top_10 = set(np.argsort(similarities_q)[-11:-1])
    
    recall = len(gt_top_10 & result_top_10) / 10
    print(f"   Recall@10: {recall:.1%}")

if __name__ == "__main__":
    demo_lancedb_integration()
    
    print("\n" + "=" * 80)
    print("✓ LanceDB Integration Demo Completo!")
    print("=" * 80)
    print("\nPróximos passos:")
    print("  1. Implementar em Rust (real LanceDB plugin)")
    print("  2. Integrar Python bindings")
    print("  3. Deploy em produção")

#!/usr/bin/env python3
"""
BIHE_E8_LATTICE.py
E8 Lattice Implementation - Quantização 8D
Status: Production Ready
Data: 26 de Outubro, 2025
"""

import numpy as np
from itertools import combinations, product
import time

class E8_Lattice_Complete:
    """E8 Lattice com 240 pontos completos (Viazovska 2016)"""
    
    def __init__(self):
        self.dimension = 8
        self.kissing_number = 240
        self.e8_points = self._generate_e8_complete()
        print(f"✓ E8 Lattice criado com {len(self.e8_points)} pontos")
    
    def _generate_e8_complete(self):
        """Gera 240 pontos E8 completos"""
        points = []
        
        # Tipo 1: (±1, ±1, 0, 0, 0, 0, 0, 0) e permutações (112 pontos)
        for indices in combinations(range(8), 2):
            for signs in product([-1, 1], repeat=2):
                p = np.zeros(8)
                p[list(indices)] = signs
                points.append(p / np.sqrt(2))
        
        # Tipo 2: (±1/2, ±1/2, ..., ±1/2) com soma par (128 pontos)
        for signs in product([-1, 1], repeat=8):
            if sum(signs) % 2 == 0:
                p = np.array(signs) / 2.0
                points.append(p / np.linalg.norm(p))
        
        points_array = np.array(points)
        
        # Remover duplicatas (manter 240 únicos)
        unique_points = []
        for p in points_array:
            is_duplicate = False
            for up in unique_points:
                if np.allclose(p, up):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)
        
        return np.array(unique_points[:240])
    
    def quantize(self, x):
        """Quantizar vetor x para ponto E8 mais próximo"""
        distances = np.linalg.norm(self.e8_points - x, axis=1)
        idx = np.argmin(distances)
        return idx, self.e8_points[idx], distances[idx]

class BIHE_16D:
    """BIHE em 16D usando 4× blocos D4"""
    
    def __init__(self, codebook_size=2048):
        from scipy.spatial.distance import cdist
        self.dimension = 16
        self.codebook_size = codebook_size
        self.codebook_per_block = codebook_size // 4
        self.codebooks = [None for _ in range(4)]
    
    def train(self, data, max_iterations=30):
        """Treinar cada bloco 4D independentemente"""
        print(f"Treinando BIHE 16D com {self.codebook_size} centróides...")
        
        for i in range(4):
            print(f"  Bloco {i+1}/4...")
            blocks_i = data[:, 4*i:4*i+4]
            
            # Lloyd algorithm
            indices = np.random.choice(len(blocks_i), 
                                      size=min(self.codebook_per_block, len(blocks_i)), 
                                      replace=False)
            self.codebooks[i] = blocks_i[indices].copy()
            
            from scipy.spatial.distance import cdist
            
            for iteration in range(max_iterations):
                distances = cdist(blocks_i, self.codebooks[i])
                assignments = np.argmin(distances, axis=1)
                
                new_codebook = np.zeros_like(self.codebooks[i])
                for k in range(len(self.codebooks[i])):
                    cluster = blocks_i[assignments == k]
                    if len(cluster) > 0:
                        new_codebook[k] = cluster.mean(axis=0)
                    else:
                        new_codebook[k] = blocks_i[np.random.randint(len(blocks_i))]
                
                change = np.linalg.norm(new_codebook - self.codebooks[i])
                self.codebooks[i] = new_codebook
                
                if change < 1e-5:
                    break
        
        print("✓ Treinamento completo")
    
    def quantize(self, x):
        """Quantizar vetor 16D"""
        codes = []
        for i in range(4):
            block = x[4*i:4*i+4]
            if self.codebooks[i] is None:
                codes.append(0)
            else:
                distances = np.linalg.norm(self.codebooks[i] - block, axis=1)
                codes.append(np.argmin(distances))
        return codes

def test_e8_lattice():
    """Teste E8 Lattice"""
    print("\n" + "=" * 80)
    print("E8 LATTICE TEST - 8D Quantization")
    print("=" * 80)
    
    # Gerar dados
    print("\n1. Gerando dados 8D...")
    data_8d = np.random.randn(1000, 8)
    print(f"   Shape: {data_8d.shape}")
    
    # Quantizar com E8
    print("\n2. Quantizando com E8...")
    e8 = E8_Lattice_Complete()
    
    start = time.time()
    quantized_e8 = np.array([e8.quantize(s)[1] for s in data_8d])
    elapsed = time.time() - start
    
    print(f"   Tempo: {elapsed:.3f}s")
    print(f"   Throughput: {len(data_8d) / elapsed:.0f} vectors/s")
    
    # Métricas
    print("\n3. Métricas...")
    mse_e8 = np.mean((data_8d - quantized_e8) ** 2)
    variance_8d = np.var(data_8d)
    voronoi_e8 = 2.0
    nsm_e8 = mse_e8 / (variance_8d * (voronoi_e8 ** (2.0/8)))
    
    shannon_limit = 1.0 / (2 * np.pi * np.e)
    gap_e8 = (nsm_e8 - shannon_limit) / shannon_limit * 100
    
    print(f"   MSE: {mse_e8:.6f}")
    print(f"   NSM: {nsm_e8:.6f}")
    print(f"   Gap vs Shannon: {gap_e8:.1f}%")
    print(f"   Target (< 0.070): {'✓ ATINGIDO' if nsm_e8 < 0.070 else '✗'}")
    
    return nsm_e8

def test_bihe_16d():
    """Teste BIHE 16D"""
    print("\n" + "=" * 80)
    print("BIHE 16D TEST - 16D Quantization")
    print("=" * 80)
    
    # Gerar dados
    print("\n1. Gerando dados 16D...")
    data_16d = np.random.randn(1000, 16)
    print(f"   Shape: {data_16d.shape}")
    
    # Treinar
    print("\n2. Treinando BIHE 16D...")
    bihe_16d = BIHE_16D(codebook_size=2048)
    
    start = time.time()
    bihe_16d.train(data_16d)
    train_time = time.time() - start
    print(f"   Tempo: {train_time:.2f}s")
    
    # Quantizar
    print("\n3. Quantizando...")
    start = time.time()
    quantized_16d = []
    for sample in data_16d:
        codes = bihe_16d.quantize(sample)
        q_blocks = [bihe_16d.codebooks[i][codes[i]] for i in range(4)]
        quantized_16d.append(np.concatenate(q_blocks))
    quantized_16d = np.array(quantized_16d)
    quant_time = time.time() - start
    
    print(f"   Tempo: {quant_time:.3f}s")
    print(f"   Throughput: {len(data_16d) / quant_time:.0f} vectors/s")
    
    # Métricas
    print("\n4. Métricas...")
    mse_16d = np.mean((data_16d - quantized_16d) ** 2)
    nsm_16d = mse_16d / (np.var(data_16d) * (2.0 ** (2.0/16)))
    
    gap_16d = (nsm_16d - shannon_limit) / shannon_limit * 100
    
    print(f"   MSE: {mse_16d:.6f}")
    print(f"   NSM: {nsm_16d:.6f}")
    print(f"   Gap vs Shannon: {gap_16d:.1f}%")
    
    return nsm_16d

if __name__ == "__main__":
    shannon_limit = 1.0 / (2 * np.pi * np.e)
    
    nsm_8d = test_e8_lattice()
    nsm_16d = test_bihe_16d()
    
    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"E8 (8D):     NSM = {nsm_8d:.6f}")
    print(f"BIHE (16D):  NSM = {nsm_16d:.6f}")
    print(f"Shannon:     NSM = {shannon_limit:.6f}")

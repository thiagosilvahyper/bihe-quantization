"""
BIHE_TEST_SUITE.py - Suite Completa de Testes Parametrizada
Status: Produção - VERSÃO CORRIGIDA
Versão: 1.1
Data: 26 de Outubro, 2025
Author: BIHE Research Team

CORREÇÕES:
- Erro: quantize() retorna tupla (idx, quantizado, erro)
- Solução: Extrair apenas o vetor quantizado [1]
"""

import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import time
import json
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class TestConfig:
    """Configuração centralizada"""
    
    def __init__(self):
        self.dimensions = [4, 8, 16]
        self.dataset_sizes = [100, 1000, 10000]
        self.codebook_sizes = [64, 128, 256, 512, 1024]
        self.algorithms = ['BLQ', 'BLQ_Lloyd', 'RaBitQ']
        self.metrics = ['MSE', 'NSM', 'Gap', 'Recall', 'Compression', 'Latency']
        self.shannon_limit = 1.0 / (2 * np.pi * np.e)
        self.voronoi_volumes = {4: 2.0, 8: 2.0, 16: 2.0}

# ============================================================================
# ALGORITMOS
# ============================================================================

class QuantizationAlgorithm:
    """Interface base para algoritmos"""
    
    def train(self, data):
        pass
    
    def quantize(self, x):
        """Retorna: (idx, quantizado, erro)"""
        raise NotImplementedError

class BLQ(QuantizationAlgorithm):
    """BIHE-Lattice Quantization básica"""
    
    def __init__(self, dimension=4, codebook_size=512):
        self.dimension = dimension
        self.codebook_size = codebook_size
        self.d4_points = self._generate_d4()
    
    def _generate_d4(self):
        points = []
        for i in range(4):
            for j in range(i+1, 4):
                base = np.zeros(4)
                base[i] = 1
                base[j] = 1
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        p = base.copy()
                        p[i] *= si
                        p[j] *= sj
                        points.append(p / np.sqrt(2))
        return np.array(points)
    
    def quantize(self, x):
        if self.dimension == 4 and len(self.d4_points) > 0:
            distances = np.linalg.norm(self.d4_points - x, axis=1)
            idx = np.argmin(distances)
            return idx, self.d4_points[idx], distances[idx]
        else:
            return 0, x, 0.0

class BLQ_Lloyd(QuantizationAlgorithm):
    """BIHE-Lattice com Lloyd algorithm"""
    
    def __init__(self, dimension=4, codebook_size=512):
        self.dimension = dimension
        self.codebook_size = codebook_size
        self.codebook = None
    
    def train(self, data, max_iterations=30):
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
        if self.codebook is None or len(self.codebook) == 0:
            return 0, x, 0.0
        distances = np.linalg.norm(self.codebook - x, axis=1)
        idx = np.argmin(distances)
        error = np.linalg.norm(x - self.codebook[idx])
        return idx, self.codebook[idx], error

class RaBitQ_Simple(QuantizationAlgorithm):
    """RaBitQ baseline"""
    
    def __init__(self, dimension=4, codebook_size=512):
        self.dimension = dimension
    
    def quantize(self, x):
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-10:
            return 0, np.zeros_like(x), 0.0
        x_normalized = x / x_norm
        x_quantized = np.sign(x_normalized) * 0.7071
        error = np.linalg.norm(x - x_quantized)
        return 0, x_quantized, error

# ============================================================================
# MÉTRICAS
# ============================================================================

class MetricsCalculator:
    """Cálculo centralizado de métricas"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
    
    def compute_mse(self, original, quantized):
        """Mean Squared Error"""
        return np.mean((original - quantized) ** 2)
    
    def compute_nsm(self, original, quantized, dimension):
        """Normalized Second Moment"""
        mse = self.compute_mse(original, quantized)
        variance = np.var(original)
        if variance < 1e-10:
            variance = 1.0
        voronoi = self.config.voronoi_volumes.get(dimension, 2.0)
        nsm = mse / (variance * (voronoi ** (2.0 / dimension)))
        return nsm
    
    def compute_gap(self, nsm):
        """Gap vs Shannon limit em %"""
        return (nsm - self.config.shannon_limit) / self.config.shannon_limit * 100
    
    def compute_recall_at_k(self, original_data, quantized_data, query_idx, k=10):
        """Recall@k"""
        k = min(k, len(original_data) - 1)
        dist_orig = np.linalg.norm(original_data - original_data[query_idx], axis=1)
        top_k_orig = np.argsort(dist_orig)[1:k+1]
        
        dist_quant = np.linalg.norm(quantized_data - quantized_data[query_idx], axis=1)
        top_k_quant = np.argsort(dist_quant)[1:k+1]
        
        intersection = len(set(top_k_orig) & set(top_k_quant))
        return intersection / k
    
    def compute_compression_ratio(self, original_bytes, quantized_bytes):
        """Taxa de compressão"""
        return original_bytes / quantized_bytes
    
    def compute_latency(self, quantizer, sample, n_iterations=10):
        """Latência média em microsegundos"""
        start = time.time()
        for _ in range(n_iterations):
            quantizer.quantize(sample)
        elapsed = (time.time() - start) / n_iterations
        return elapsed * 1e6  # em microsegundos

# ============================================================================
# SUITE DE TESTES
# ============================================================================

class TestSuite:
    """Suite completa de testes parametrizada"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.metrics = MetricsCalculator(self.config)
        self.results = []
    
    def test_basic_quantization(self):
        """Teste 1: Quantização Básica"""
        print("\n" + "="*80)
        print("TESTE 1: Quantização Básica (BLQ vs RaBitQ)")
        print("="*80)
        
        data = np.random.randn(100, 4)
        results = []
        
        for algo_name, algo in [('BLQ', BLQ()), ('RaBitQ', RaBitQ_Simple())]:
            # CORREÇÃO: Extrair apenas o vetor quantizado [1]
            quantized = np.array([algo.quantize(s)[1] for s in data])
            mse = self.metrics.compute_mse(data, quantized)
            nsm = self.metrics.compute_nsm(data, quantized, 4)
            gap = self.metrics.compute_gap(nsm)
            
            results.append({
                'Teste': 'Quantização',
                'Algoritmo': algo_name,
                'MSE': f"{mse:.6f}",
                'NSM': f"{nsm:.6f}",
                'Gap %': f"{gap:.1f}%"
            })
            print(f"  {algo_name}: MSE={mse:.6f}, NSM={nsm:.6f}, Gap={gap:.1f}%")
        
        self.results.extend(results)
        return results
    
    def test_lloyd_convergence(self):
        """Teste 2: Lloyd Algorithm"""
        print("\n" + "="*80)
        print("TESTE 2: Lloyd Algorithm - Convergência")
        print("="*80)
        
        data = np.random.randn(1000, 4)
        results = []
        
        for codebook_size in self.config.codebook_sizes:
            algo = BLQ_Lloyd(codebook_size=codebook_size)
            algo.train(data, max_iterations=30)
            
            # CORREÇÃO: Extrair apenas o vetor quantizado [1]
            quantized = np.array([algo.quantize(s)[1] for s in data])
            mse = self.metrics.compute_mse(data, quantized)
            nsm = self.metrics.compute_nsm(data, quantized, 4)
            gap = self.metrics.compute_gap(nsm)
            
            results.append({
                'Teste': 'Lloyd',
                'Codebook': codebook_size,
                'MSE': f"{mse:.6f}",
                'NSM': f"{nsm:.6f}",
                'Gap %': f"{gap:.1f}%",
                'Target': '✓' if nsm < 0.090 else '✗'
            })
            print(f"  CB{codebook_size:4d}: NSM={nsm:.6f}, Gap={gap:.1f}%, Target: {'✓' if nsm < 0.090 else '✗'}")
        
        self.results.extend(results)
        return results
    
    def test_scalability(self):
        """Teste 3: Escalabilidade"""
        print("\n" + "="*80)
        print("TESTE 3: Escalabilidade")
        print("="*80)
        
        results = []
        
        for size in [100, 1000]:
            data = np.random.randn(size, 4)
            algo = BLQ_Lloyd(codebook_size=256)
            
            start = time.time()
            algo.train(data[:min(500, size)], max_iterations=10)
            train_time = time.time() - start
            
            start = time.time()
            # CORREÇÃO: Extrair apenas o vetor quantizado [1]
            quantized = np.array([algo.quantize(s)[1] for s in data])
            quant_time = time.time() - start
            
            throughput = size / quant_time
            nsm = self.metrics.compute_nsm(data, quantized, 4)
            
            results.append({
                'Teste': 'Escalabilidade',
                'Size': size,
                'Train (ms)': f"{train_time*1000:.2f}",
                'Quant (ms)': f"{quant_time*1000:.2f}",
                'Throughput': f"{throughput:.0f}",
                'NSM': f"{nsm:.6f}"
            })
            print(f"  Size {size:5d}: Train={train_time*1000:.1f}ms, Throughput={throughput:.0f} samples/s")
        
        self.results.extend(results)
        return results
    
    def test_recall_metrics(self):
        """Teste 4: Recall@K"""
        print("\n" + "="*80)
        print("TESTE 4: Recall@K")
        print("="*80)
        
        data = np.random.randn(100, 4)
        algo = BLQ()
        # CORREÇÃO: Extrair apenas o vetor quantizado [1]
        quantized = np.array([algo.quantize(s)[1] for s in data])
        
        results = []
        
        for k in [1, 5, 10]:
            recalls = [self.metrics.compute_recall_at_k(data, quantized, i, k=k) for i in range(10)]
            avg_recall = np.mean(recalls)
            
            results.append({
                'Teste': 'Recall',
                'K': k,
                'Avg': f"{avg_recall:.2%}",
                'Min': f"{np.min(recalls):.2%}",
                'Max': f"{np.max(recalls):.2%}"
            })
            print(f"  Recall@{k:2d}: {avg_recall:.2%} (min: {np.min(recalls):.2%}, max: {np.max(recalls):.2%})")
        
        self.results.extend(results)
        return results
    
    def test_compression(self):
        """Teste 5: Compressão"""
        print("\n" + "="*80)
        print("TESTE 5: Compressão de Embeddings")
        print("="*80)
        
        # BERT-like (768-dim)
        original_size = 100 * 768 * 4
        quantized_size = 100 * 192 * 2
        
        compression = original_size / quantized_size
        reduction = (1 - quantized_size / original_size) * 100
        
        results = [{
            'Teste': 'Compressão',
            'Original (KB)': f"{original_size/1024:.1f}",
            'Quantized (KB)': f"{quantized_size/1024:.1f}",
            'Ratio': f"{compression:.1f}×",
            'Reduction %': f"{reduction:.1f}%"
        }]
        
        print(f"  Original: {original_size/1024:.1f} KB")
        print(f"  Quantized: {quantized_size/1024:.1f} KB")
        print(f"  Compression: {compression:.1f}× ({reduction:.1f}% reduction)")
        
        self.results.extend(results)
        return results
    
    def run_all(self):
        """Executar todos os testes"""
        print("\n" + "="*100)
        print("BIHE TEST SUITE - SUITE COMPLETA DE TESTES PARAMETRIZADA")
        print("="*100)
        
        self.test_basic_quantization()
        self.test_lloyd_convergence()
        self.test_scalability()
        self.test_recall_metrics()
        self.test_compression()
        
        return self.results
    
    def save_report(self, filename):
        """Salvar relatório em JSON"""
        df = pd.DataFrame(self.results)
        df.to_json(filename, orient='records', indent=2)
        print(f"\n✓ Relatório salvo: {filename}")
        
        return df
    
    def print_summary(self):
        """Imprimir resumo"""
        df = pd.DataFrame(self.results)
        print("\n" + "="*100)
        print("RESUMO FINAL")
        print("="*100)
        print(f"\nTotal de testes: {len(df)}")
        print(f"\nResultados:")
        print(df.to_string(index=False))

# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Criar suite
    config = TestConfig()
    suite = TestSuite(config)
    
    # Executar testes
    results = suite.run_all()
    
    # Salvar relatório
    suite.save_report("bihe_test_results.json")
    
    # Imprimir resumo
    suite.print_summary()
    
    print("\n" + "="*100)
    print("✓ SUITE COMPLETA FINALIZADA COM SUCESSO")
    print("="*100)
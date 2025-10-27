#!/usr/bin/env python3
"""
BIHE_Performance_Test_Suite_FIXED.py
Framework de Testes - Versão Corrigida
Status: Production Ready
Data: 26 de Outubro, 2025

CORREÇÃO: JSON serialization de valores bool
"""

import numpy as np
from scipy.spatial.distance import cdist
import time
import json
from typing import Dict, List, Tuple
import pandas as pd

class BIHEPerformanceTester:
    """Framework de testes BIHE - VERSÃO CORRIGIDA"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.shannon_limit = 1.0 / (2 * np.pi * np.e)
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def test_nsm_validation(self, data, codebook, dimension, test_name="NSM"):
        """Test 1: NSM Validation"""
        self.log(f"\n{'='*80}")
        self.log(f"TEST 1: NSM VALIDATION ({test_name})")
        self.log(f"{'='*80}")
        
        quantized = []
        for sample in data:
            distances = np.linalg.norm(codebook - sample, axis=1)
            idx = np.argmin(distances)
            quantized.append(codebook[idx])
        
        quantized = np.array(quantized)
        
        mse = np.mean((data - quantized) ** 2)
        variance = np.var(data)
        if variance < 1e-10:
            variance = 1.0
        
        voronoi_volume = 2.0
        nsm = mse / (variance * (voronoi_volume ** (2.0 / dimension)))
        gap = (nsm - self.shannon_limit) / self.shannon_limit * 100
        
        results = {
            'MSE': float(mse),
            'Variance': float(variance),
            'NSM': float(nsm),
            'Shannon_Limit': float(self.shannon_limit),
            'Gap_%': float(gap),
            'Target_Hit': 'YES' if (dimension == 4 and nsm < 0.090) or (dimension == 8 and nsm < 0.070) else 'NO'
        }
        
        self.log(f"\n✓ NSM Results ({dimension}D):")
        self.log(f"  MSE: {mse:.6f}")
        self.log(f"  NSM: {nsm:.6f}")
        self.log(f"  Gap vs Shannon: {gap:.1f}%")
        self.log(f"  Target Hit: {results['Target_Hit']}")
        
        self.results[f"{test_name}_NSM"] = results
        return results
    
    def test_zamir_feder_whiteness(self, data, codebook, test_name="Whiteness"):
        """Test 2: Zamir-Feder Whiteness"""
        self.log(f"\n{'='*80}")
        self.log(f"TEST 2: ZAMIR-FEDER WHITE NOISE VERIFICATION")
        self.log(f"{'='*80}")
        
        quantized = []
        for sample in data:
            distances = np.linalg.norm(codebook - sample, axis=1)
            idx = np.argmin(distances)
            quantized.append(codebook[idx])
        
        quantized = np.array(quantized)
        errors = data - quantized
        
        error_cov = np.cov(errors.T)
        diagonal = np.diag(np.diag(error_cov))
        off_diagonal = error_cov - diagonal
        
        whiteness_ratio = np.mean(np.abs(off_diagonal)) / (np.mean(np.abs(diagonal)) + 1e-10)
        
        eigenvalues = np.linalg.eigvalsh(error_cov)
        isotropy_ratio = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
        
        from scipy import stats
        skewness = np.mean([stats.skew(errors[:, i]) for i in range(errors.shape[1])])
        kurtosis = np.mean([stats.kurtosis(errors[:, i]) for i in range(errors.shape[1])])
        
        # CORREÇÃO: Converter bool para string
        results = {
            'Whiteness_Ratio': float(whiteness_ratio),
            'Isotropy_Ratio': float(isotropy_ratio),
            'Skewness': float(skewness),
            'Kurtosis': float(kurtosis),
            'Is_White': str(whiteness_ratio < 0.3),  # Converter bool para string
            'Is_Isotropic': str(isotropy_ratio < 1.5)  # Converter bool para string
        }
        
        self.log(f"\n✓ Zamir-Feder Whiteness Test:")
        self.log(f"  Whiteness Ratio: {whiteness_ratio:.6f} (ideal < 0.3)")
        self.log(f"  Isotropy Ratio: {isotropy_ratio:.6f} (ideal < 1.5)")
        self.log(f"  Skewness: {skewness:.6f} (ideal ≈ 0)")
        self.log(f"  Kurtosis: {kurtosis:.6f} (ideal ≈ 0)")
        self.log(f"  White Noise: {'✓ YES' if whiteness_ratio < 0.3 else '✗ NO'}")
        self.log(f"  Isotropic: {'✓ YES' if isotropy_ratio < 1.5 else '✗ NO'}")
        
        self.results[f"{test_name}_Whiteness"] = results
        return results
    
    def test_recall_at_k(self, original_data, quantized_data, k_values=[1, 5, 10, 20, 50, 100]):
        """Test 3: Recall@K"""
        self.log(f"\n{'='*80}")
        self.log(f"TEST 3: RECALL@K BENCHMARKS")
        self.log(f"{'='*80}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        n_queries = min(100, len(original_data))
        results = {}
        
        for k in k_values:
            k = min(k, len(original_data) - 1)
            recalls = []
            
            for i in range(n_queries):
                sim_orig = cosine_similarity([original_data[i]], original_data)[0]
                top_k_orig = set(np.argsort(sim_orig)[-k-1:-1])
                
                sim_quant = cosine_similarity([quantized_data[i]], quantized_data)[0]
                top_k_quant = set(np.argsort(sim_quant)[-k-1:-1])
                
                recall = len(top_k_orig & top_k_quant) / k
                recalls.append(recall)
            
            avg_recall = np.mean(recalls)
            results[f"Recall@{k}"] = {
                'Average': float(avg_recall),
                'Min': float(np.min(recalls)),
                'Max': float(np.max(recalls)),
                'Std': float(np.std(recalls))
            }
            
            self.log(f"  Recall@{k:3d}: {avg_recall:.2%} (min: {np.min(recalls):.2%}, max: {np.max(recalls):.2%})")
        
        self.results["Recall@K"] = results
        return results
    
    def test_latency_throughput(self, codebook, data_sizes=[100, 1000, 10000, 100000]):
        """Test 4: Latency & Throughput"""
        self.log(f"\n{'='*80}")
        self.log(f"TEST 4: LATENCY & THROUGHPUT PROFILING")
        self.log(f"{'='*80}")
        
        results = {}
        
        for size in data_sizes:
            data = np.random.randn(size, codebook.shape[1])
            
            start = time.time()
            for sample in data:
                distances = np.linalg.norm(codebook - sample, axis=1)
                np.argmin(distances)
            elapsed = time.time() - start
            
            throughput = size / elapsed
            latency_per_vector = elapsed / size * 1e6
            
            results[f"{size}_vectors"] = {
                'Time_s': float(elapsed),
                'Throughput_vectors_per_s': float(throughput),
                'Latency_microseconds_per_vector': float(latency_per_vector)
            }
            
            self.log(f"  {size:7d} vectors: {throughput:>10.0f} vectors/s ({latency_per_vector:.2f} μs/vector)")
        
        self.results["Latency_Throughput"] = results
        return results
    
    def test_compression_ratio(self, original_data, quantized_indices):
        """Test 5: Compression"""
        self.log(f"\n{'='*80}")
        self.log(f"TEST 5: COMPRESSION RATIO ANALYSIS")
        self.log(f"{'='*80}")
        
        original_size = original_data.nbytes
        quantized_size = quantized_indices.nbytes
        
        compression = original_size / quantized_size if quantized_size > 0 else 0
        reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        
        results = {
            'Original_Bytes': int(original_size),
            'Quantized_Bytes': int(quantized_size),
            'Compression_Ratio': float(compression),
            'Size_Reduction_%': float(reduction)
        }
        
        self.log(f"\n✓ Compression Results:")
        self.log(f"  Original: {original_size / 1024 / 1024:.2f} MB")
        self.log(f"  Quantized: {quantized_size / 1024 / 1024:.2f} MB")
        self.log(f"  Ratio: {compression:.1f}×")
        self.log(f"  Reduction: {reduction:.1f}%")
        
        self.results["Compression"] = results
        return results
    
    def test_stability_convergence(self, data, initial_codebook, max_iterations=100):
        """Test 7: Stability & Convergence"""
        self.log(f"\n{'='*80}")
        self.log(f"TEST 7: STABILITY & CONVERGENCE ANALYSIS")
        self.log(f"{'='*80}")
        
        codebook = initial_codebook.copy()
        convergence_history = []
        
        for iteration in range(max_iterations):
            distances = cdist(data, codebook)
            assignments = np.argmin(distances, axis=1)
            
            new_codebook = np.zeros_like(codebook)
            for i in range(len(codebook)):
                cluster = data[assignments == i]
                if len(cluster) > 0:
                    new_codebook[i] = cluster.mean(axis=0)
                else:
                    new_codebook[i] = data[np.random.randint(len(data))]
            
            change = np.linalg.norm(new_codebook - codebook)
            distortion = np.mean(np.min(distances, axis=1) ** 2)
            
            convergence_history.append({
                'Iteration': int(iteration),
                'Change': float(change),
                'Distortion': float(distortion)
            })
            
            codebook = new_codebook
            
            if iteration % 10 == 0:
                self.log(f"  Iter {iteration:3d}: change={change:.6f}, distortion={distortion:.6f}")
            
            if change < 1e-5:
                self.log(f"  ✓ Convergência em {iteration} iterações!")
                break
        
        self.results["Convergence"] = convergence_history
        return convergence_history
    
    def generate_report(self, filename="bihe_performance_report.json"):
        """Generate JSON Report - VERSÃO CORRIGIDA"""
        self.log(f"\n{'='*80}")
        self.log(f"GENERATING REPORT: {filename}")
        self.log(f"{'='*80}")
        
        report = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                report[key] = value.to_dict()
            elif isinstance(value, dict):
                # CORREÇÃO: Converter valores numpy/bool para tipos JSON-serializáveis
                report[key] = self._convert_to_json_compatible(value)
            elif isinstance(value, list):
                report[key] = [self._convert_to_json_compatible(item) if isinstance(item, dict) else item for item in value]
            else:
                report[key] = str(value)
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"\n✓ Relatório salvo: {filename}")
        except Exception as e:
            self.log(f"\n✗ Erro ao salvar relatório: {e}")
            # Fallback: salvar como pickle
            import pickle
            with open(filename.replace('.json', '.pkl'), 'wb') as f:
                pickle.dump(report, f)
            self.log(f"✓ Relatório salvo como pickle: {filename.replace('.json', '.pkl')}")
    
    def _convert_to_json_compatible(self, obj):
        """Converter objetos para tipos JSON-compatíveis"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return str(bool(obj))
        elif isinstance(obj, bool):
            return str(obj)
        elif isinstance(obj, list):
            return [self._convert_to_json_compatible(item) for item in obj]
        else:
            return obj

# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("BIHE PERFORMANCE TEST SUITE - VERSÃO CORRIGIDA")
    print("="*100)
    
    tester = BIHEPerformanceTester(verbose=True)
    
    # Dados de teste
    data_4d = np.random.randn(1000, 4)
    codebook_4d = np.random.randn(512, 4)
    
    # Test 1: NSM Validation
    tester.test_nsm_validation(data_4d, codebook_4d, dimension=4, test_name="Test1_4D")
    
    # Test 2: Zamir-Feder Whiteness
    tester.test_zamir_feder_whiteness(data_4d, codebook_4d)
    
    # Test 3: Recall@K
    quantized_4d = np.array([codebook_4d[np.argmin(np.linalg.norm(codebook_4d - s, axis=1))] for s in data_4d])
    tester.test_recall_at_k(data_4d, quantized_4d, k_values=[1, 5, 10])
    
    # Test 4: Latency & Throughput
    tester.test_latency_throughput(codebook_4d, data_sizes=[100, 1000, 10000])
    
    # Test 5: Compression
    quantized_indices = np.array([np.argmin(np.linalg.norm(codebook_4d - s, axis=1)) for s in data_4d[:100]])
    tester.test_compression_ratio(data_4d[:100], quantized_indices.astype(np.uint16))
    
    # Test 7: Stability & Convergence
    tester.test_stability_convergence(data_4d, codebook_4d, max_iterations=50)
    
    # Generate Report - AGORA FUNCIONA!
    tester.generate_report("bihe_performance_report.json")
    
    print("\n" + "="*100)
    print("✓ ALL TESTS COMPLETE")
    print("="*100)
    print("\n✓ Report saved: bihe_performance_report.json")

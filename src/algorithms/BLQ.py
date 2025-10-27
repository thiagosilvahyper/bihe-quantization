def BLQ(x):  # x ∈ ℝⁿ, n divisível por 4
    # 1. Dividir em blocos 4D
    blocks = reshape(x, (n//4, 4))
    
    # 2. Decompor: magnitude + direção
    norms = [||b|| for b in blocks]
    dirs = [b/||b|| for b in blocks]
    
    # 3. Quantizar direção → D4 lattice (24 vizinhos)
    idx_lattice = [D4_quantize(d) for d in dirs]  # 0-23
    
    # 4. Quantizar magnitude → Γ(3)=2 níveis
    idx_magnitude = [log_quantize(n, levels=2) for n in norms]
    
    # 5. Codebook BIHE: 24 × 2 = 48 estados (ou 512 full)
    codes = [idx_lat + 24*idx_mag 
             for idx_lat, idx_mag in zip(idx_lattice, idx_magnitude)]
    
    return codes

# Complexidade: O(n) encoding, O(n) decoding
# Memória: 512-entry lookup table (4KB)

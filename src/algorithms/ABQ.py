def ABQ(x):
    # 1. Whitening transform (Zamir-Feder)
    μ, Σ = mean(x), cov(x)
    x_white = Σ^(-1/2) × (x - μ)
    
    # 2. Quantizar distribuição branca
    codes = BLQ(x_white)
    
    # 3. Codificar transformação
    transform_code = encode(μ, Σ)
    
    return (codes, transform_code)

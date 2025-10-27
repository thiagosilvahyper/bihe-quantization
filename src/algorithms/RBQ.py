def RBQ(x, stages=2):
    residual = x
    codes = []
    
    for stage in range(stages):
        c = BLQ(residual, codebook_size=512//2^stage)
        x_rec = decode(c)
        residual = residual - x_rec
        codes.append(c)
    
    return codes

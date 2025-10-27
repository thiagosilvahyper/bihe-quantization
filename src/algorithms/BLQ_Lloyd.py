class BLQ_Lloyd:
    def __init__(self, dimension=4, codebook_size=512):
        self.dimension = dimension
        self.codebook_size = codebook_size
        
    def train(self, data, max_iterations=50):
        # K-means++ initialization
        self.initialize_codebook(data)
        
        # Lloyd iterations
        for i in range(max_iterations):
            change, distortion = self.lloyd_iteration(data)
            if change < tolerance:
                break

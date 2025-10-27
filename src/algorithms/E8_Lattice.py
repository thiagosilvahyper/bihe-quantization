class E8_Lattice:
    def __init__(self):
        self.dimension = 8
        self.e8_points = self._generate_e8_points()  # 128-240 pontos
    
    def quantize(self, x):
        # Encontrar ponto E8 mais próximo
        distances = np.linalg.norm(self.e8_points - x, axis=1)
        return np.argmin(distances)

import lancedb
from bihe_quantizer import BLQ_Lloyd

# 1. Conectar LanceDB
db = lancedb.connect("./data/vectors")

# 2. Carregar embeddings BERT (768-dim)
embeddings = load_bert_embeddings("data/bert_768.npy")

# 3. Treinar BIHE quantizer
blq = BLQ_Lloyd(dimension=4, codebook_size=512)
sample = embeddings[:10000].reshape(-1, 4)
blq.train(sample, max_iterations=50)

# 4. Quantizar embeddings
quantized = []
for emb in embeddings:
    blocks = emb.reshape(-1, 4)  # 192 blocos
    codes = [blq.quantize(block) for block in blocks]
    quantized.append(np.array(codes, dtype=np.uint16))

# 5. Armazenar no LanceDB
data = [{"id": i, "embedding": emb, "quantized": q} 
        for i, (emb, q) in enumerate(zip(embeddings, quantized))]
table = db.create_table("embeddings_bihe", data=data)

# 6. Query
query_embedding = encode_bert("What is machine learning?")
results = table.search(query_embedding).limit(10).to_list()

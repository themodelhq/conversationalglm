from memory.embeddings import cosine,embed_text
def test_embedding_similarity():
 a=embed_text("The user prefers dark mode")
 b=embed_text("User prefers dark interface")
 c=embed_text("Bananas grow in tropical regions")
 assert cosine(a,b)>cosine(a,c)

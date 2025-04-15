from sentence_transformers import SentenceTransformer

def get_embeddings(texts):
    """
    Generate embeddings for a list of texts using SentenceTransformer.
    
    Args:
        texts (list): List of strings to generate embeddings for
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return embeddings

if __name__ == "__main__":
    # Example usage
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = get_embeddings(sentences)
    print("Embeddings shape:", embeddings.shape)
    print("First embedding:", embeddings[0]) 
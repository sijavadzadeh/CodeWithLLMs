from datasets import load_dataset
from sentence_transformers import SentenceTransformer,util
import torch

dataset = load_dataset("multi_news", split="test", trust_remote_code=True)
df = dataset.to_pandas().sample(2000,random_state=22)

model = SentenceTransformer("all-MiniLM-L6-v2")

passage_embeddings = list(model.encode(df['summary'].tolist()))
passage_embeddings[0].shape

query = "Find me some articles about technology and artificial inteligence"

query_embedding = model.encode(query)

similarities = util.cos_sim(query_embedding,passage_embeddings)

top_indices = torch.topk(similarities.flatten(),k=3).indices

top_relevant_passages=[df.iloc[x.item()]['summary'][:200] + "..." for x in top_indices]







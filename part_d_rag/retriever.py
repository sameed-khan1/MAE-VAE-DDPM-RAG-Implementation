
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from config import EMBEDDING_MODEL, DEVICE


class Retriever:
    def __init__(self):
        self.device = DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)

            embeddings = summed / counts
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


if __name__ == "__main__":
    retriever = Retriever()

    embeddings = retriever.encode([
        "What is histone lysine methylation?",
        "What are the effects of cancer?"
    ])

    print("Embedding shape:", embeddings.shape)
    print("Device:", retriever.device)
    print("Embedding norms:", torch.norm(embeddings, dim=1))

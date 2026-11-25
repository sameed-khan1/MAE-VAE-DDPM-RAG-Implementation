
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import GENERATOR_MODEL, DEVICE, MAX_NEW_TOKENS


class BioASQGenerator:
    def __init__(self):
        self.device = DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(
            GENERATOR_MODEL
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

    def generate(self, question, retrieved_chunks):
        passages = ""

        for i, chunk in enumerate(retrieved_chunks, 1):
            passages += (
                f"\n[{i}] Abstract ID: {chunk['abstract_id']}\n"
                f"{chunk['text']}\n"
            )

        prompt = f"""You are a biomedical question-answering assistant.

Answer the question using ONLY the provided passages.

Rules:
1. Use only information supported by the passages.
2. Cite supporting passages using [1], [2], etc.
3. If the passages do not contain enough information, say:
   "I don't know based on the provided passages."
4. Do not invent medical facts.

Passages:
{passages}

Question: {question}

Answer:"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        return answer.strip()


if __name__ == "__main__":
    from search import BioASQRetriever

    retriever = BioASQRetriever()
    generator = BioASQGenerator()

    question = "What is the implication of histone lysine methylation in medulloblastoma?"

    results = retriever.search(question, top_k=5)
    answer = generator.generate(question, results)

    print("\nQuestion:", question)
    print("\nGenerated Answer:")
    print(answer)
    print("\nFor educational use only. Not medical advice.")

import math

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer


def numerical_score(predicted, actual, alpha=23):
    return math.exp(-alpha * abs(predicted - actual) / actual)


class BertBleuCalculator:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print("Loading BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, device_map=self.device)
        self.model.eval()
        print("Model loaded.")

    def get_embeddings(self, sentence):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=False,  # Avoiding [CLS] [SEP] for pure n-gram matching
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Last hidden state contains token embeddings
        # Shape: (1, seq_len, hidden_size) -> (seq_len, hidden_size)
        embeddings = outputs.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return tokens, embeddings

    def get_ngram_embeddings(self, embeddings, n):
        num_tokens = embeddings.shape[0]
        if num_tokens < n:
            return []

        ngram_embeddings = []
        for i in range(num_tokens - n + 1):
            segment = embeddings[i : i + n]
            phrase_embedding = torch.mean(segment, dim=0)
            ngram_embeddings.append(phrase_embedding.cpu().numpy())

        return ngram_embeddings

    def calculate_Pn(self, candidate_ngrams, reference_ngrams):
        if not reference_ngrams:
            return 0.0
        if not candidate_ngrams:
            return 0.0

        C_matrix = np.array(candidate_ngrams)
        R_matrix = np.array(reference_ngrams)

        similarity_matrix = cosine_similarity(R_matrix, C_matrix)

        max_similarities = np.max(similarity_matrix, axis=1)

        Pn = np.mean(max_similarities)

        return Pn


bert_blue_calculator = None


def bert_bleu_score(candidate_sentence, reference_sentence, N=4, alpha=0.5):
    global bert_blue_calculator
    if bert_blue_calculator is None:
        bert_blue_calculator = BertBleuCalculator()

    c_tokens, c_emb = bert_blue_calculator.get_embeddings(candidate_sentence)
    r_tokens, r_emb = bert_blue_calculator.get_embeddings(reference_sentence)

    Lc = len(candidate_sentence.split(" "))
    Lr = len(reference_sentence.split(" "))

    if Lc == 0 or Lr == 0:
        return 0.0

    Pn_values = []
    for n in range(1, N + 1):
        c_ngrams = bert_blue_calculator.get_ngram_embeddings(c_emb, n)
        r_ngrams = bert_blue_calculator.get_ngram_embeddings(r_emb, n)

        pn = bert_blue_calculator.calculate_Pn(c_ngrams, r_ngrams)
        Pn_values.append(pn)

    max_Pn = max(Pn_values) if Pn_values else 0.0

    length_diff = abs(Lc - Lr)
    lp = np.exp(-alpha * (length_diff / Lr))

    bert_bleu = lp * max_Pn

    return bert_bleu

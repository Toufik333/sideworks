"""
SBERT Architecture — Core Module
=================================
Implements the Sentence-BERT architecture as described in:
  Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks" (EMNLP 2019)

Key components (Section 3 of the paper):
  1. MeanPooling — Default pooling strategy (MEAN > MAX > CLS per ablation)
  2. SBERTModel — Pretrained transformer + pooling → fixed-size embeddings
  3. SoftmaxClassificationHead — (u, v, |u-v|) → softmax (Figure 1)
  4. RegressionHead — cosine-sim(u, v) → MSE loss (Figure 2)
  5. Cosine similarity matrix computation for evaluation

Reference: https://github.com/UKPLab/sentence-transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Dict, Union
import numpy as np


# ============================================================================
# Device Detection — Apple Silicon MPS with CPU fallback
# ============================================================================

def get_device() -> torch.device:
    """
    Detect the best available compute device for Apple Silicon.
    Priority: MPS (Apple GPU) → CUDA → CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ============================================================================
# 1. Pooling Strategies (Section 3)
# ============================================================================

class MeanPooling(nn.Module):
    """
    Mean Pooling — Paper's default and best-performing strategy.
    
    From Section 3: "computing the mean of all output vectors (MEAN-strategy)"
    From Section 6 (Ablation): MEAN pooling outperforms CLS and MAX for both
    classification and regression objectives.
    
    Computes the mean of token embeddings, weighted by the attention mask
    to exclude padding tokens from the average.
    """

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask:   [batch_size, seq_len]
        Returns:
            Pooled embeddings: [batch_size, hidden_dim]
        """
        # Expand mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        
        # Sum embeddings where mask is 1, divide by count of non-padded tokens
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask


class MaxPooling(nn.Module):
    """
    Max Pooling — MAX-over-time strategy.
    
    From Section 6: "When trained with the regression objective function,
    the MAX strategy performs significantly worse than MEAN or CLS-token strategy."
    """

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        
        # Set padded positions to very negative so they are never selected
        token_embeddings[mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, dim=1)[0]


class CLSPooling(nn.Module):
    """CLS-token output pooling (index 0)."""

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return token_embeddings[:, 0]


# ============================================================================
# 2. SBERT Model — Transformer + Pooling (Section 3)
# ============================================================================

class SBERTModel(nn.Module):
    """
    Core SBERT model: Pretrained Transformer + Pooling Layer.
    
    From Section 3: "SBERT adds a pooling operation to the output of
    BERT / RoBERTa to derive a fixed sized sentence embedding."
    
    This module produces fixed-size dense sentence embeddings from
    variable-length text inputs.
    """

    POOLING_STRATEGIES = {
        "mean": MeanPooling,
        "max": MaxPooling,
        "cls": CLSPooling,
    }

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling_strategy: str = "mean",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.model_name = model_name
        
        # Load pretrained transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pooling strategy (paper default: MEAN)
        if pooling_strategy not in self.POOLING_STRATEGIES:
            raise ValueError(
                f"Unknown pooling: {pooling_strategy}. "
                f"Choose from: {list(self.POOLING_STRATEGIES.keys())}"
            )
        self.pooling = self.POOLING_STRATEGIES[pooling_strategy]()
        self.embedding_dim = self.transformer.config.hidden_size
        
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass: Transformer → Pooling → Sentence Embedding
        
        Returns:
            sentence_embedding: [batch_size, hidden_dim]
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        # Use last hidden state: [batch_size, seq_len, hidden_dim]
        token_embeddings = outputs.last_hidden_state
        
        # Pool to fixed-size vector
        sentence_embedding = self.pooling(token_embeddings, attention_mask)
        
        return sentence_embedding

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode sentences to dense embeddings (inference mode).
        
        Optimized for M1: uses torch.no_grad(), processes in batches,
        and moves results to CPU for numpy conversion.
        
        Args:
            sentences: Single sentence or list of sentences.
            batch_size: Batch size (32 is safe for M1 8GB).
            normalize: L2-normalize embeddings (for cosine similarity).
            show_progress: Show tqdm progress bar.
        
        Returns:
            np.ndarray of shape [num_sentences, hidden_dim]
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        self.eval()
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding", unit="batch")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx:start_idx + batch_size]
                
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                embeddings = self.forward(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0).numpy()


# ============================================================================
# 3. Siamese Network for Inference (Figure 2)
# ============================================================================

class SiameseSBERT(nn.Module):
    """
    Siamese SBERT for pairwise similarity scoring.
    
    From Section 3, Figure 2: "SBERT architecture at inference, for example,
    to compute similarity scores. This architecture is also used with the
    regression objective function."
    
    Both sentences pass through the SAME BERT network (tied weights),
    then cosine similarity is computed between their embeddings.
    """

    def __init__(self, sbert_model: SBERTModel):
        super().__init__()
        self.sbert = sbert_model

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a pair of sentences.
        
        Returns dict with:
            - 'embedding_a': sentence A embedding
            - 'embedding_b': sentence B embedding
            - 'cosine_similarity': cosine sim between A and B
        """
        # Both sentences through the SAME network (siamese / tied weights)
        u = self.sbert(input_ids_a, attention_mask_a)
        v = self.sbert(input_ids_b, attention_mask_b)
        
        cosine_sim = F.cosine_similarity(u, v, dim=1)
        
        return {
            "embedding_a": u,
            "embedding_b": v,
            "cosine_similarity": cosine_sim,
        }


# ============================================================================
# 4. Classification Head — Softmax (Figure 1, Section 3)
# ============================================================================

class SoftmaxClassificationHead(nn.Module):
    """
    Classification objective from Section 3, Figure 1:
    
    "We concatenate the sentence embeddings u and v with the element-wise
     difference |u−v| and multiply it with the trainable weight W_t ∈ R^{3n×k}:
         o = softmax(W_t(u, v, |u − v|))
     where n is the dimension of the sentence embeddings and k the number
     of labels. We optimize cross-entropy loss."
    
    Used for fine-tuning on NLI datasets (SNLI + MultiNLI) with 3 labels:
    entailment, contradiction, neutral.
    """

    def __init__(self, embedding_dim: int, num_labels: int = 3):
        super().__init__()
        # Input: (u, v, |u-v|) → 3 * embedding_dim
        # Paper: W_t ∈ R^{3n×k}
        self.classifier = nn.Linear(3 * embedding_dim, num_labels)

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            u: Sentence A embedding [batch_size, embedding_dim]
            v: Sentence B embedding [batch_size, embedding_dim]
        Returns:
            logits: [batch_size, num_labels]
        """
        # Paper concatenation: (u, v, |u-v|)
        diff = torch.abs(u - v)
        combined = torch.cat([u, v, diff], dim=1)
        
        logits = self.classifier(combined)
        return logits


# ============================================================================
# 5. Regression Head — Cosine Similarity + MSE (Figure 2, Section 3)
# ============================================================================

class RegressionHead(nn.Module):
    """
    Regression objective from Section 3, Figure 2:
    
    "The cosine-similarity between the two sentence embeddings u and v
     is computed. We use mean-squared-error loss as the objective function."
    
    Used for fine-tuning on STS benchmark dataset.
    """

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between sentence embeddings.
        
        Returns:
            cosine_similarity: [batch_size] values in [-1, 1]
        """
        return F.cosine_similarity(u, v, dim=1)


# ============================================================================
# 6. Full Training Models
# ============================================================================

class SBERTForClassification(nn.Module):
    """
    Full SBERT model for NLI classification training (Figure 1).
    Combines: SBERT encoder + Softmax classification head.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 3,
        pooling_strategy: str = "mean",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.sbert = SBERTModel(model_name, pooling_strategy, device)
        self.head = SoftmaxClassificationHead(
            self.sbert.embedding_dim, num_labels
        )
        self.device = self.sbert.device
        self.head.to(self.device)

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        u = self.sbert(input_ids_a, attention_mask_a)
        v = self.sbert(input_ids_b, attention_mask_b)
        logits = self.head(u, v)
        return logits


class SBERTForRegression(nn.Module):
    """
    Full SBERT model for STS regression training (Figure 2).
    Combines: SBERT encoder + Cosine similarity regression.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling_strategy: str = "mean",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.sbert = SBERTModel(model_name, pooling_strategy, device)
        self.head = RegressionHead()
        self.device = self.sbert.device

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        u = self.sbert(input_ids_a, attention_mask_a)
        v = self.sbert(input_ids_b, attention_mask_b)
        cosine_sim = self.head(u, v)
        return cosine_sim


# ============================================================================
# 7. Utility Functions
# ============================================================================

def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for a set of embeddings.
    
    This is the core operation for SBERT evaluation (Section 4):
    "we always use cosine-similarity to compare the similarity between
     two sentence embeddings"
    
    Args:
        embeddings: np.ndarray of shape [N, dim]
    Returns:
        similarity_matrix: np.ndarray of shape [N, N] with values in [-1, 1]
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normalized = embeddings / norms
    
    # Cosine similarity = dot product of normalized vectors
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


def pairwise_cosine_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity for paired embeddings (element-wise).
    
    Args:
        embeddings_a: [N, dim]
        embeddings_b: [N, dim]
    Returns:
        similarities: [N] cosine similarity for each pair
    """
    # Normalize
    a_norm = embeddings_a / np.maximum(
        np.linalg.norm(embeddings_a, axis=1, keepdims=True), 1e-9
    )
    b_norm = embeddings_b / np.maximum(
        np.linalg.norm(embeddings_b, axis=1, keepdims=True), 1e-9
    )
    
    return np.sum(a_norm * b_norm, axis=1)


# ============================================================================
# Module info
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SBERT Architecture Module")
    print("  Based on: Reimers & Gurevych (EMNLP 2019)")
    print("=" * 65)
    
    device = get_device()
    print(f"\n  Device: {device}")
    
    print("\n  Components:")
    print("    • MeanPooling       — Default pooling (Section 3)")
    print("    • MaxPooling        — MAX-over-time (Section 3)")
    print("    • CLSPooling        — CLS token output (Section 3)")
    print("    • SBERTModel        — Transformer + Pooling")
    print("    • SiameseSBERT      — Siamese inference (Figure 2)")
    print("    • SoftmaxClassHead  — (u,v,|u-v|) → softmax (Figure 1)")
    print("    • RegressionHead    — cosine-sim → MSE (Figure 2)")
    print("    • SBERTForClassification — Full NLI training model")
    print("    • SBERTForRegression     — Full STS training model")
    
    print("\n  Quick test with all-MiniLM-L6-v2...")
    model = SBERTModel("sentence-transformers/all-MiniLM-L6-v2", device=device)
    embs = model.encode(["Hello world", "Hi there"])
    sim = compute_cosine_similarity_matrix(embs)
    print(f"    Embedding shape: {embs.shape}")
    print(f"    Cosine similarity: {sim[0, 1]:.4f}")
    print("\n  ✅ Architecture module loaded successfully!")

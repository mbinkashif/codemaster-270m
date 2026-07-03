"""Evaluation utilities for CodeMaster."""

import torch
import numpy as np
from typing import Dict, List, Tuple
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss."""
    return np.exp(loss)


def calculate_bleu_score(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Calculate BLEU score."""
    # Tokenize
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]
    
    # Calculate BLEU-4
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate BLEU-1, 2, 3
    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0))
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
    }


def calculate_rouge_scores(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
    }


def exact_match_score(prediction: str, reference: str) -> float:
    """Calculate exact match score."""
    return float(prediction.strip() == reference.strip())


def f1_score(prediction: str, reference: str) -> float:
    """Calculate F1 score between prediction and reference."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    common = pred_tokens & ref_tokens
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

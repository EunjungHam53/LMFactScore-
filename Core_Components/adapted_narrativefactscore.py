"""
File thay thế cho narrativefactscore.py
Sử dụng local NLI models thay vì GPT API
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
import logging
from typing import List, Tuple
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import util

from local_model_replacements import AtomicFactDecomposer, LocalFactVerifier
from fact.utils import break_down2scenes

logger = logging.getLogger(__name__)


class NarrativeFactScore:
    """
    Thay thế class gốc - sử dụng local models
    """
    def __init__(
        self, 
        device: str = "cuda:0",
        model: str = "microsoft/deberta-v3-large-mnli",
        split_type: str = "local",
        checkpoint: str = None
    ):
        """
        Args:
            device: CUDA device
            model: NLI model name từ HuggingFace
            split_type: 'local' (dùng local model) hoặc 'fast' (sentence split)
            checkpoint: Không dùng nữa (để tương thích API)
        """
        self.device = device
        self.model = model
        self.split_type = split_type
        
        # Load sentence embedding model (giữ nguyên từ code gốc)
        self.sent_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info("Loaded sentence embedding model")
        
        # Load local models
        self.decomposer = AtomicFactDecomposer(
            model_name="google/flan-t5-base",
            device=device
        )
        self.verifier = LocalFactVerifier(
            model_name=model,
            device=device
        )
        logger.info("Loaded local fact decomposer and verifier")
    
    def get_surrounding_sentences(self, sentence_array: List[str], ii: int) -> str:
        """Giữ nguyên từ code gốc"""
        if ii > 0 and ii < len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 : ii + 1])
        elif ii == 0:
            sents = " ".join(np.array(sentence_array)[:2])
        elif ii == len(sentence_array) - 1:
            sents = " ".join(np.array(sentence_array)[ii - 1 :])
        return sents
    
    def split_sent(self, text: str) -> List[str]:
        """
        Split summary thành atomic facts
        Thay thế GPT API bằng local model
        """
        if self.split_type == "fast":
            # Fast mode: simple sentence splitting
            text_list = []
            for t in text.split('.'):
                t = t.strip()
                if len(t) > 0:
                    text_list.append(t)
            return text_list
        
        elif self.split_type == "local":
            # Local model mode: sử dụng decomposer
            try:
                text_list = self.decomposer.decompose(text)
                return text_list
            except Exception as e:
                logger.error(f"Error in decomposition: {e}, falling back to fast mode")
                return self.split_sent_fast(text)
        
        else:
            logger.warning(f"Unknown split_type: {self.split_type}, using fast mode")
            return self.split_sent_fast(text)
    
    def split_sent_fast(self, text: str) -> List[str]:
        """Fallback fast splitting"""
        return [t.strip() for t in text.split('.') if t.strip()]
    
    def score_src_hyp_long(
        self,
        srcs: List[str],
        hyps: List[str],
        kgs: List[str]
    ) -> Tuple[List[float], List[List[float]], List[List[str]], List[List[str]], List[List[str]]]:
        """
        Score summaries với local models
        
        Args:
            srcs: List of source scripts
            hyps: List of generated summaries
            kgs: List of KG triplets (formatted strings)
            
        Returns:
            all_scores: Document-level scores
            all_scores_per_sent: Sentence-level scores
            all_relevant_scenes: Most relevant source scenes
            all_summary_chunks: Decomposed summary facts
            all_feedback_list: Feedback for incorrect facts
        """
        all_scores = []
        all_scores_per_sent = []
        all_relevant_scenes = []
        all_summary_chunks = []
        all_feedback_list = []
        
        total_score = 0
        
        for global_idx, (src, hyp) in enumerate(zip(tqdm(srcs), hyps)):
            # Break down source into scenes
            src_sents = break_down2scenes(src)
            
            # Encode source sentences
            sentence_embeddings_src = self.sent_model.encode(
                src_sents, 
                batch_size=12, 
                max_length=8192
            )['dense_vecs']
            
            # Encode KG triplets
            sentence_embeddings_kg = self.sent_model.encode(
                kgs, 
                batch_size=12, 
                max_length=8192
            )['dense_vecs']
            
            # Decompose summary into atomic facts
            hyp_array = self.split_sent(hyp)
            
            doc_scores = []
            relevant_scenes = []
            feedbacks = []
            
            for idx, hyp_sentence in enumerate(hyp_array):
                # Encode hypothesis sentence
                sentence_embeddings_hyp = self.sent_model.encode(
                    hyp_sentence, 
                    max_length=8192
                )['dense_vecs']
                
                # Find most similar source sentences
                scores = util.cos_sim(sentence_embeddings_hyp, sentence_embeddings_src)[0]
                scores_kg = util.cos_sim(sentence_embeddings_hyp, sentence_embeddings_kg)[0]
                
                # Sort by similarity
                sorted_idxs = np.argsort(-1 * scores)  # Descending order
                sorted_idxs_kg = np.argsort(-1 * scores_kg)
                
                # Get most relevant KG triplets
                triple = ''
                for sorted_idx, ii in enumerate(sorted_idxs_kg[0:1]):
                    if sorted_idx == 0:
                        triple += f'{kgs[ii]}'
                    else:
                        triple += f', {kgs[ii]}'
                
                # Get most similar source scenes
                similar_src_sentences = []
                for ii in sorted_idxs[0:1]:  # Top 1 most similar
                    similar_sents = src_sents[ii]
                    similar_src_sentences.append(similar_sents)
                
                # Verify fact using local NLI model
                fact_scores = []
                fact_feedbacks = []
                
                for similar_sent in similar_src_sentences:
                    score, feedback = self.verifier.verify_fact(
                        premise=similar_sent,
                        hypothesis=hyp_sentence,
                        kg_context=triple
                    )
                    fact_scores.append(score)
                    fact_feedbacks.append(feedback)
                
                # Take max score
                max_score = np.max(fact_scores)
                max_idx = np.argmax(fact_scores)
                max_scene = similar_src_sentences[max_idx]
                max_feedback = fact_feedbacks[max_idx]
                
                doc_scores.append(max_score)
                relevant_scenes.append(max_scene)
                feedbacks.append(max_feedback)
            
            # Calculate document-level score
            doc_score = np.mean(doc_scores) if doc_scores else 0.0
            
            all_scores.append(doc_score)
            all_scores_per_sent.append(doc_scores)
            all_relevant_scenes.append(relevant_scenes)
            all_summary_chunks.append(hyp_array)
            all_feedback_list.append(feedbacks)
            
            total_score += doc_score
            
            if global_idx % 100 == 99:
                print(f"Document {global_idx+1} Mean Score: {total_score/(global_idx+1):.4f}")
        
        return (
            all_scores,
            all_scores_per_sent,
            all_relevant_scenes,
            all_summary_chunks,
            all_feedback_list
        )
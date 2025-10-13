"""
Vietnamese-specific models cho news summarization
Thay thế OpenAI API với Vietnamese models
"""

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from typing import List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class VietnameseSummarizer:
    """
    Summarizer cho tiếng Việt
    Sử dụng ViT5 hoặc mBART models
    """
    def __init__(self, model_name="VietAI/vit5-large-vietnews-summarization", device="cuda"):
        """
        Args:
            model_name: 
                - "VietAI/vit5-large-vietnews-summarization" (Recommended for news)
                - "VietAI/vit5-base-vietnews-summarization"
                - "VietAI/vit5-large"
                - "google/mt5-base" (multilingual)
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading Vietnamese summarizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info(f"Loaded Vietnamese summarizer successfully")
    
    def generate_summary(self, text: str, max_length: int = 256) -> str:
        """
        Tạo summary từ Vietnamese text
        
        Args:
            text: Vietnamese news text
            max_length: Max length của summary
            
        Returns:
            summary: Vietnamese summary
        """
        # ViT5 format: "vietnews: <text>"
        if "vit5" in self.model_name.lower():
            input_text = f"vietnews: {text}"
        else:
            input_text = text
        
        inputs = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()


class VietnameseAtomicFactDecomposer:
    """
    Phân tách summary thành atomic facts cho tiếng Việt
    """
    def __init__(self, model_name="VietAI/vit5-base", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info(f"Loaded Vietnamese fact decomposer: {model_name}")
    
    def decompose(self, summary: str) -> List[str]:
        """
        Phân tách Vietnamese summary thành các atomic facts
        """
        # Prompt tiếng Việt
        prompt = f"""Hãy chia đoạn tóm tắt sau thành các câu sự kiện đơn lẻ (atomic facts). Mỗi sự kiện phải là một câu độc lập, đầy đủ nghĩa.

Tóm tắt: {summary}

Các sự kiện:"""
        
        inputs = self.tokenizer(
            prompt,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.3,
                do_sample=False
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse kết quả thành list các facts
        facts = [f.strip() for f in result.split('\n') if f.strip()]
        
        # Fallback: Vietnamese sentence splitting
        if len(facts) <= 1:
            # Simple Vietnamese sentence splitting
            facts = self._split_vietnamese_sentences(result)
        
        return facts
    
    def _split_vietnamese_sentences(self, text: str) -> List[str]:
        """Simple Vietnamese sentence splitter"""
        # Split by common Vietnamese sentence endings
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]


class VietnameseKGExtractor:
    """
    Knowledge Graph extraction cho tiếng Việt
    Sử dụng NER + Relation Extraction
    """
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load Vietnamese NER model
        logger.info("Loading Vietnamese NER model...")
        self.ner_pipeline = pipeline(
            "ner",
            model="NlpHUST/ner-vietnamese-electra-base",
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1
        )
        
        # Load multilingual relation extraction (REBEL hỗ trợ Vietnamese)
        logger.info("Loading relation extraction model...")
        self.tokenizer_rel = AutoTokenizer.from_pretrained("Babelscape/mrebel-large")
        self.model_rel = AutoModelForSeq2SeqLM.from_pretrained(
            "Babelscape/mrebel-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        logger.info("Vietnamese KG extractor loaded")
    
    def extract_named_entities_and_relations(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Extract entities và relations từ Vietnamese text
        
        Returns:
            entities: List of entity names
            relations: List of (subject, predicate, object) tuples
        """
        # Step 1: Extract named entities
        entities = self._extract_entities(text)
        
        # Step 2: Extract relations
        relations = self._extract_relations(text, entities)
        
        return entities, relations
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities using Vietnamese NER"""
        try:
            ner_results = self.ner_pipeline(text)
            entities = list(set([ent['word'] for ent in ner_results]))
            logger.info(f"Extracted {len(entities)} entities")
            return entities
        except Exception as e:
            logger.error(f"Error in NER: {e}")
            return []
    
    def _extract_relations(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relations using mREBEL (multilingual REBEL)"""
        try:
            # Truncate text nếu quá dài
            inputs = self.tokenizer_rel(
                text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_rel.generate(
                    **inputs,
                    max_length=256,
                    num_beams=3,
                    num_return_sequences=1
                )
            
            decoded = self.tokenizer_rel.decode(outputs[0], skip_special_tokens=False)
            
            # Parse REBEL output
            relations = self._parse_rebel_output(decoded, entities)
            logger.info(f"Extracted {len(relations)} relations")
            return relations
            
        except Exception as e:
            logger.error(f"Error in relation extraction: {e}")
            return []
    
    def _parse_rebel_output(self, decoded: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Parse REBEL output format"""
        relations = []
        
        # REBEL format: <triplet> subject <subj> predicate <obj> object
        for triplet_str in decoded.split('<triplet>')[1:]:
            try:
                # Extract subject
                if '<subj>' not in triplet_str:
                    continue
                subject = triplet_str.split('<subj>')[0].strip()
                
                # Extract object and predicate
                remaining = triplet_str.split('<subj>')[1]
                if '<obj>' not in remaining:
                    continue
                
                predicate = remaining.split('<obj>')[0].strip()
                obj = remaining.split('<obj>')[1].split('</s>')[0].strip()
                
                if subject and predicate and obj:
                    relations.append((subject, predicate, obj))
            except Exception as e:
                logger.debug(f"Error parsing triplet: {e}")
                continue
        
        return relations


class VietnameseFactVerifier:
    """
    Verify factuality cho tiếng Việt sử dụng NLI
    """
    def __init__(self, model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device="cuda"):
        """
        Args:
            model_name: Multilingual NLI model
                - "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" (Recommended)
                - "NlpHUST/vi-nlp-deberta-base"
        """
        self.device = device
        self.nli_pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        logger.info(f"Loaded Vietnamese fact verifier: {model_name}")
    
    def verify_fact(self, premise: str, hypothesis: str, kg_context: str = "") -> Tuple[float, str]:
        """
        Verify xem hypothesis có được support bởi premise không
        
        Args:
            premise: Vietnamese source text
            hypothesis: Vietnamese statement to verify
            kg_context: KG context (optional)
            
        Returns:
            score: 1.0 nếu supported, 0.0 nếu contradicted
            feedback: Feedback tiếng Việt nếu không supported
        """
        # Combine premise với KG context
        full_premise = premise
        if kg_context:
            full_premise = f"{premise}\nThông tin liên quan: {kg_context}"
        
        # Run NLI
        try:
            result = self.nli_pipeline({
                "text": full_premise,
                "text_pair": hypothesis
            })[0]
            
            label = result['label'].lower()
            confidence = result['score']
            
            if 'entail' in label or 'support' in label:
                return 1.0, ""
            elif 'contradict' in label:
                feedback = f"Phát biểu mâu thuẫn với nguồn (độ tin cậy: {confidence:.2f})"
                return 0.0, feedback
            else:  # neutral
                if confidence > 0.7:
                    feedback = f"Phát biểu không được hỗ trợ bởi nguồn (độ tin cậy: {confidence:.2f})"
                    return 0.0, feedback
                else:
                    return 0.5, f"Phát biểu không chắc chắn (độ tin cậy: {confidence:.2f})"
        
        except Exception as e:
            logger.error(f"Error in fact verification: {e}")
            return 0.5, "Lỗi khi xác minh"


class VietnameseModelManager:
    """
    Manager cho tất cả Vietnamese models
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.summarizer = None
        self.decomposer = None
        self.kg_extractor = None
        self.verifier = None
        
        logger.info(f"Initializing Vietnamese Model Manager on {device}")
    
    def load_summarizer(self, model_name="VietAI/vit5-large-vietnews-summarization"):
        """Load Vietnamese summarization model"""
        self.summarizer = VietnameseSummarizer(model_name, self.device)
        return self.summarizer
    
    def load_decomposer(self, model_name="VietAI/vit5-base"):
        """Load atomic fact decomposer"""
        self.decomposer = VietnameseAtomicFactDecomposer(model_name, self.device)
        return self.decomposer
    
    def load_kg_extractor(self):
        """Load knowledge graph extractor"""
        self.kg_extractor = VietnameseKGExtractor(self.device)
        return self.kg_extractor
    
    def load_verifier(self, model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        """Load fact verifier"""
        self.verifier = VietnameseFactVerifier(model_name, self.device)
        return self.verifier
    
    def load_all(self):
        """Load tất cả models"""
        self.load_summarizer()
        self.load_decomposer()
        self.load_kg_extractor()
        self.load_verifier()
        logger.info("All Vietnamese models loaded successfully")


# Example usage
if __name__ == "__main__":
    # Test Vietnamese models
    manager = VietnameseModelManager(device="cuda")
    manager.load_all()
    
    # Test text
    text = """Chuyến bay của hãng Viva Aerobus mang số hiệu VB518, có sức chứa 186 hành khách, 
    khởi hành từ Guadalajara, Mexico, tối 24/8. Khoảng 10 phút sau khi cất cánh, 
    hành khách phát hiện tia lửa bắn ra từ động cơ bên phải của máy bay."""
    
    # Test summarization
    summary = manager.summarizer.generate_summary(text)
    print(f"Summary: {summary}")
    
    # Test decomposition
    facts = manager.decomposer.decompose(summary)
    print(f"Facts: {facts}")
    
    # Test KG extraction
    entities, relations = manager.kg_extractor.extract_named_entities_and_relations(text)
    print(f"Entities: {entities}")
    print(f"Relations: {relations}")
    
    # Test verification
    for fact in facts:
        score, feedback = manager.verifier.verify_fact(text, fact)
        print(f"Fact: {fact}, Score: {score}, Feedback: {feedback}")
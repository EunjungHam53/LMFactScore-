"""
Vietnamese-specific models cho news summarization
Thay thế OpenAI API với Vietnamese models
Đã sửa: Thay spaCy bằng PhoBERT NER model
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
    Sử dụng PhoBERT NER + ViT5 relation generation
    Độ chính xác cao, không giới hạn relation types
    """
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load PhoBERT NER model thay vì spaCy
        logger.info("Loading PhoBERT NER model...")
        try:
            # Sử dụng PhoBERT fine-tuned cho NER
            # Có thể dùng các model sau:
            # 1. "NlpHUST/ner-vietnamese-electra-base" - ELECTRA-based NER
            # 2. "uitnlp/vihealthbert-base-finetuned-ner" - Healthcare NER
            # 3. "FPTAI/velectra-base-discriminator-finetuned-ner" - VELEcTRA NER
            # 4. "PhoNER_COVID19" - COVID-19 domain NER
            
            self.ner_pipeline = pipeline(
                "ner",
                model="NlpHUST/ner-vietnamese-electra-base",
                aggregation_strategy="simple",  # Gộp các sub-tokens lại
                device=0 if device == "cuda" else -1
            )
            logger.info("PhoBERT NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load PhoBERT NER model: {e}")
            logger.info("Falling back to regex-based entity extraction")
            self.ner_pipeline = None
        
        # Load ViT5 tokenizer
        logger.info("Loading ViT5 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
        
        # Load ViT5 model for relation generation
        logger.info("Loading ViT5 model for relation generation...")
        self.relation_model = AutoModelForSeq2SeqLM.from_pretrained(
            "VietAI/vit5-base",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        logger.info("Vietnamese KG extractor loaded successfully")
    
    def extract_named_entities_and_relations(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Extract entities và relations từ Vietnamese text
        
        Returns:
            entities: List of entity names
            relations: List of (subject, predicate, object) tuples
        """
        # Step 1: Extract named entities
        entities = self._extract_entities(text)
        
        if not entities or len(entities) < 2:
            logger.warning("Less than 2 entities found, cannot extract relations")
            return entities, []
        
        # Step 2: Extract relations using ViT5
        relations = self._extract_relations_vit5(text, entities)
        
        return entities, relations
    
    def _safe_truncate(self, text: str, max_chars: int = 2000) -> str:
        """
        Truncate text an toàn tại sentence boundary
        Tránh cắt giữa câu làm mất entities
        
        Args:
            text: Input text
            max_chars: Max characters to keep
        
        Returns:
            Truncated text at sentence boundary
        """
        if len(text) <= max_chars:
            return text
        
        # Truncate tại max_chars
        truncated = text[:max_chars]
        
        # Tìm sentence boundary gần nhất (dấu câu + space)
        # Vietnamese sentence endings: . ! ? ; \n
        last_sentence_end = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? '),
            truncated.rfind('.\n'),
            truncated.rfind('!\n'),
            truncated.rfind('?\n')
        )
        
        if last_sentence_end > max_chars * 0.8:  # Chỉ truncate nếu không mất quá 20%
            return truncated[:last_sentence_end + 1]
        else:
            # Nếu không tìm thấy sentence boundary gần, truncate tại word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_chars * 0.9:
                return truncated[:last_space]
            else:
                # Worst case: truncate hard tại max_chars
                logger.warning("Hard truncation applied - may cut entities")
                return truncated

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities using PhoBERT NER pipeline"""
        try:
            if self.ner_pipeline is not None:
                # CRITICAL: Truncate text để tránh tensor size mismatch
                # PhoBERT NER max_length = 512 tokens (~2000 chars Vietnamese)
                # Truncate an toàn tại sentence boundary để không mất entities
                text_truncated = self._safe_truncate(text, max_chars=2000)
                
                # Sử dụng PhoBERT NER pipeline
                ner_results = self.ner_pipeline(text_truncated)
                
                # Extract entity text và loại bỏ duplicates
                entities = []
                seen = set()
                
                for entity in ner_results:
                    entity_text = entity['word'].strip()
                    # Filter: chỉ lấy entities có ít nhất 2 ký tự
                    if len(entity_text) >= 2 and entity_text not in seen:
                        entities.append(entity_text)
                        seen.add(entity_text)
                
                logger.info(f"Extracted {len(entities)} entities using PhoBERT NER")
                return entities
            else:
                # Fallback: Regex-based extraction
                return self._extract_entities_regex(text)
                
        except Exception as e:
            logger.error(f"Error in PhoBERT NER: {e}")
            logger.info("Falling back to regex-based extraction")
            return self._extract_entities_regex(text)
    
    def _extract_entities_regex(self, text: str) -> List[str]:
        """
        Fallback: Extract entities bằng regex patterns
        Tìm proper nouns, tên riêng, địa danh, tổ chức
        """
        entities = []
        seen = set()
        
        # Pattern 1: Capitalized words (tên riêng)
        # Trong tiếng Việt, tên riêng thường viết hoa chữ cái đầu
        capitalized_pattern = r'\b[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+)*'
        matches = re.findall(capitalized_pattern, text)
        
        for match in matches:
            match = match.strip()
            # Filter: ít nhất 2 từ hoặc 1 từ dài hơn 3 ký tự
            words = match.split()
            if (len(words) >= 2 or len(match) > 3) and match not in seen:
                entities.append(match)
                seen.add(match)
        
        # Pattern 2: Các từ khóa location/organization
        # Tìm các từ theo sau các marker như "tại", "ở", "công ty", "tổ chức"
        location_pattern = r'(?:tại|ở|đến|từ)\s+([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]+?)(?=[,.]|\s+của|\s+với|\s+và)'
        org_pattern = r'(?:công ty|tổ chức|hãng|trường|bệnh viện|đài)\s+([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ][^\n,.]{2,50}?)(?=[,.])'
        
        for pattern in [location_pattern, org_pattern]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if len(match) > 2 and match not in seen:
                    entities.append(match)
                    seen.add(match)
        
        logger.info(f"Extracted {len(entities)} entities using regex fallback")
        return entities
    
    def _extract_relations_vit5(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract relations using ViT5 generation
        Không giới hạn số loại relations, generate natural descriptions
        """
        relations = []
        
        if len(entities) < 2:
            return relations
        
        # Truncate text để input vừa vặn
        text_truncated = text[:512 * 4]  # ~2000 chars
        
        # Filter: chỉ lấy top entity pairs có relevance cao
        # (tránh generate cho quá nhiều cặp, gây chậm)
        entity_pairs = self._select_relevant_pairs(text_truncated, entities)
        
        logger.info(f"Processing {len(entity_pairs)} entity pairs for relation extraction")
        
        # Batch generate relations
        for ent1, ent2 in entity_pairs:
            try:
                relation = self._generate_relation(text_truncated, ent1, ent2)
                
                if relation and relation.lower() != "không có quan hệ":
                    # Clean relation
                    relation = relation.strip()
                    if len(relation) > 3:  # Filter out very short/meaningless relations
                        relations.append((ent1, relation, ent2))
                        logger.debug(f"Relation: {ent1} - {relation} - {ent2}")
            
            except Exception as e:
                logger.debug(f"Error extracting relation ({ent1}, {ent2}): {e}")
                continue
        
        logger.info(f"Extracted {len(relations)} relations")
        return relations
    
    def _select_relevant_pairs(self, text: str, entities: List[str], max_pairs: int = 15) -> List[Tuple[str, str]]:
        """
        Chọn các cặp entities có relevance cao
        Tránh generate cho quá nhiều cặp (gây chậm)
        """
        pairs = []
        
        # Co-occurrence score: 2 entities gần nhau trong text
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i >= j:  # Tránh duplicate (ent1-ent2 và ent2-ent1)
                    continue
                
                # Tìm vị trí của 2 entities trong text
                pos1 = text.find(ent1)
                pos2 = text.find(ent2)
                
                if pos1 == -1 or pos2 == -1:
                    continue
                
                # Distance score: entities càng gần càng cao
                distance = abs(pos1 - pos2)
                if distance < 500:  # Chỉ lấy entities trong vòng 500 chars
                    score = 1.0 - (distance / 500.0)
                    pairs.append((score, ent1, ent2))
        
        # Sort by relevance score, lấy top max_pairs
        pairs.sort(reverse=True)
        selected = [(ent1, ent2) for _, ent1, ent2 in pairs[:max_pairs]]
        
        return selected
    
    def _generate_relation(self, text: str, ent1: str, ent2: str) -> str:
        """
        Generate relation description giữa 2 entities using ViT5
        Prompt tiếng Việt để model hiểu rõ task
        """
        try:
            # Tìm context xung quanh 2 entities
            context = self._extract_context(text, ent1, ent2, window=150)
            
            # Prompt tiếng Việt
            prompt = f"""Hãy mô tả quan hệ giữa '{ent1}' và '{ent2}' dựa trên đoạn văn bản:

    Văn bản: {context}

    Mô tả quan hệ (một cụm từ ngắn gọn):"""
            
            # Tokenize - FIX: Set max_length và truncation explicitly
            inputs = self.tokenizer(
                prompt,
                max_length=512,  # ViT5 base max_length
                truncation=True,  # CRITICAL: Bật truncation
                padding=False,    # Không cần padding cho generation
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.relation_model.generate(
                    **inputs,
                    max_length=25,  # Giới hạn độ dài output
                    num_beams=3,
                    temperature=0.5,
                    do_sample=False,
                    early_stopping=True
                )
            
            # Decode
            relation = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            return relation
        
        except Exception as e:
            logger.debug(f"Error in relation generation: {e}")
            return None
    
    def _extract_context(self, text: str, ent1: str, ent2: str, window: int = 150) -> str:
        """
        Extract context xung quanh 2 entities để generation tốt hơn
        """
        try:
            pos1 = text.find(ent1)
            pos2 = text.find(ent2)
            
            if pos1 == -1 or pos2 == -1:
                return text[:window]
            
            # Find common context window
            start = min(pos1, pos2) - window // 2
            start = max(0, start)
            
            end = max(pos1 + len(ent1), pos2 + len(ent2)) + window // 2
            end = min(len(text), end)
            
            context = text[start:end]
            return context
        
        except Exception as e:
            logger.debug(f"Error extracting context: {e}")
            return text[:window]


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
        FIX: Truncate premise để fit vào 512 token limit của NLI model
        
        Args:
            premise: Vietnamese source text (có thể rất dài)
            hypothesis: Vietnamese statement to verify (ngắn)
            kg_context: KG context (optional)
            
        Returns:
            score: 1.0 nếu supported, 0.0 nếu contradicted
            feedback: Feedback tiếng Việt nếu không supported
        """
        # FIX 1: Truncate premise thông minh
        # Thay vì lấy premise full, chỉ lấy phần liên quan nhất
        premise_truncated = self._truncate_premise_for_nli(premise, hypothesis, max_length=400)
        
        # Combine với KG context nếu có
        full_premise = premise_truncated
        if kg_context:
            full_premise = f"{premise_truncated}\nThông tin liên quan: {kg_context}"
        
        try:
            # Run NLI
            result = self.nli_pipeline({
                "text": full_premise,
                "text_pair": hypothesis
            })[0]
            
            label = result['label'].lower()
            confidence = result['score']
            
            logger.debug(f"NLI result - Label: {label}, Confidence: {confidence:.3f}")
            
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
            # Fallback: trả về neutral score thay vì error
            return 0.5, f"Lỗi khi xác minh: {str(e)[:50]}"


    def _truncate_premise_for_nli(self, premise: str, hypothesis: str, max_length: int = 400) -> str:
        """
        Truncate premise thông minh dựa trên hypothesis
        Tìm keywords từ hypothesis trong premise, lấy context xung quanh
        
        Args:
            premise: Toàn bộ source text
            hypothesis: Fact cần verify
            max_length: Max characters cho truncated premise
            
        Returns:
            Truncated premise (khoảng 400 chars)
        """
        if len(premise) <= max_length:
            return premise
        
        # Strategy 1: Tìm keywords từ hypothesis trong premise
        hypothesis_words = hypothesis.split()
        
        best_match_pos = -1
        best_match_count = 0
        
        # Tìm vị trí có nhiều keywords của hypothesis nhất
        for i in range(0, len(premise) - 100, 50):  # Slide window
            window = premise[i:i+200]
            match_count = sum(1 for word in hypothesis_words if word.lower() in window.lower())
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_match_pos = i
        
        # Strategy 2: Nếu tìm được keywords, lấy context xung quanh
        if best_match_pos >= 0 and best_match_count > 0:
            # Lấy 200 chars trước + 200 chars sau
            start = max(0, best_match_pos - 100)
            end = min(len(premise), best_match_pos + 300)
            truncated = premise[start:end]
            
            logger.debug(f"Found {best_match_count} keywords at position {best_match_pos}")
        else:
            # Strategy 3: Nếu không tìm được, lấy phần đầu của premise
            truncated = premise[:max_length]
            logger.debug("No keywords found, using beginning of premise")
        
        # Truncate tại sentence boundary để tránh cắt giữa câu
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:
            truncated = truncated[:last_period + 1]
        
        return truncated.strip()


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
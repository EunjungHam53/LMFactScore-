"""
Thay thế OpenAI API bằng các model open-source local
Sử dụng:
- Flan-T5/T0 cho summarization và atomic fact decomposition
- OpenIE hoặc Rebel cho knowledge graph extraction
- NLI models cho fact verification
"""

import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from typing import List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class LocalSummarizer:
    """
    Thay thế GPT API cho việc tạo summary
    Sử dụng Flan-T5-large hoặc BART-large-cnn
    """
    def __init__(self, model_name="google/flan-t5-large", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info(f"Loaded summarizer: {model_name}")
    
    def generate_summary(self, script_text: str, max_length: int = 512) -> str:
        """
        Tạo summary từ script
        """
        prompt = f"""Summarize the following movie script. Include key character actions, emotions, and outcomes. Keep it 2-5 sentences.

Script: {script_text}

Summary:"""
        
        inputs = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=50,
                num_beams=4,
                temperature=0.7,
                do_sample=False,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()


class AtomicFactDecomposer:
    """
    Thay thế GPT API cho việc phân tách atomic facts
    Sử dụng Flan-T5 với prompt engineering
    """
    def __init__(self, model_name="google/flan-t5-base", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info(f"Loaded atomic fact decomposer: {model_name}")
    
    def decompose(self, summary: str) -> List[str]:
        """
        Phân tách summary thành các atomic facts
        """
        prompt = f"""Break down the following summary into atomic facts. Each fact should be a single, independent statement in third-person format. List each fact on a new line.

Summary: {summary}

Atomic facts:"""
        
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
        
        # Fallback: nếu model không split đúng, dùng sentence tokenization
        if len(facts) <= 1:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            facts = nltk.sent_tokenize(result)
        
        return facts


class KnowledgeGraphExtractor:
    """
    Thay thế GPT API cho việc trích xuất Knowledge Graph
    Sử dụng OpenIE hoặc REBEL model
    """
    def __init__(self, method="rebel", device="cuda"):
        self.device = device
        self.method = method
        
        if method == "rebel":
            # REBEL: Relation Extraction By End-to-end Language generation
            model_name = "Babelscape/rebel-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            logger.info(f"Loaded KG extractor: REBEL")
        
        elif method == "openie":
            # Sử dụng Stanford OpenIE wrapper hoặc alternative
            try:
                from openie import StanfordOpenIE
                self.openie = StanfordOpenIE()
                logger.info("Loaded KG extractor: OpenIE")
            except ImportError:
                logger.warning("OpenIE not available, falling back to REBEL")
                self.method = "rebel"
                self.__init__(method="rebel", device=device)
    
    def extract_triplets_rebel(self, text: str) -> List[Dict[str, str]]:
        """
        Trích xuất triplets sử dụng REBEL model
        """
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=3,
                num_return_sequences=1
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Parse REBEL output format: <triplet> subj <subj> pred <obj>
        triplets = []
        for triplet_str in decoded.split('<triplet>')[1:]:
            parts = triplet_str.split('<subj>')
            if len(parts) < 2:
                continue
            
            subj_parts = parts[1].split('<obj>')
            if len(subj_parts) < 2:
                continue
            
            subject = subj_parts[0].strip()
            obj_parts = subj_parts[1].split('</s>')
            obj = obj_parts[0].strip() if obj_parts else ""
            
            # Extract predicate (between subject and object)
            pred_match = re.search(r'<subj>.*?<obj>', parts[1])
            if pred_match:
                pred_text = pred_match.group(0)
                predicate = pred_text.replace('<subj>', '').replace('<obj>', '').strip()
            else:
                predicate = ""
            
            if subject and predicate and obj:
                triplets.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj
                })
        
        return triplets
    
    def extract_triplets_openie(self, text: str) -> List[Dict[str, str]]:
        """
        Trích xuất triplets sử dụng OpenIE
        """
        if not hasattr(self, 'openie'):
            return self.extract_triplets_rebel(text)
        
        try:
            results = self.openie.annotate(text)
            triplets = []
            for triple in results:
                triplets.append({
                    'subject': triple['subject'],
                    'predicate': triple['relation'],
                    'object': triple['object']
                })
            return triplets
        except Exception as e:
            logger.error(f"OpenIE extraction failed: {e}")
            return []
    
    def extract_named_entities_and_relations(self, text: str) -> Tuple[List[str], List[Dict]]:
        """
        Trích xuất entities và relations cho Knowledge Graph
        Output format tương thích với code gốc
        """
        if self.method == "rebel":
            triplets = self.extract_triplets_rebel(text)
        else:
            triplets = self.extract_triplets_openie(text)
        
        # Extract unique entities
        entities = set()
        for t in triplets:
            entities.add(t['subject'])
            entities.add(t['object'])
        
        # Format edges: (subject, predicate, object)
        edges = []
        for t in triplets:
            edges.append((t['subject'], t['predicate'], t['object']))
        
        return list(entities), edges


class LocalFactVerifier:
    """
    Thay thế GPT API cho việc verify factuality
    Sử dụng NLI models (Natural Language Inference)
    """
    def __init__(self, model_name="microsoft/deberta-v3-large-mnli", device="cuda"):
        self.device = device
        self.nli_pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        logger.info(f"Loaded fact verifier: {model_name}")
    
    def verify_fact(self, premise: str, hypothesis: str, kg_context: str = "") -> Tuple[float, str]:
        """
        Verify xem hypothesis có được support bởi premise không
        
        Returns:
            score (float): 1.0 nếu supported, 0.0 nếu contradicted
            feedback (str): lý do nếu không supported
        """
        # Combine premise với KG context
        full_premise = premise
        if kg_context:
            full_premise = f"{premise}\nRelated information: {kg_context}"
        
        # Run NLI
        result = self.nli_pipeline({
            "text": full_premise,
            "text_pair": hypothesis
        })[0]
        
        label = result['label'].lower()
        confidence = result['score']
        
        if 'entail' in label or 'support' in label:
            return 1.0, ""
        elif 'contradict' in label:
            feedback = f"Statement contradicts the source (confidence: {confidence:.2f})"
            return 0.0, feedback
        else:  # neutral
            if confidence > 0.7:
                feedback = f"Statement is not supported by the source (confidence: {confidence:.2f})"
                return 0.0, feedback
            else:
                # Uncertain case - give partial credit
                return 0.5, f"Statement support is uncertain (confidence: {confidence:.2f})"


class LocalModelManager:
    """
    Manager class để quản lý tất cả local models
    Thay thế hoàn toàn OpenAI API
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.summarizer = None
        self.decomposer = None
        self.kg_extractor = None
        self.verifier = None
        
        logger.info(f"Initializing LocalModelManager on {device}")
    
    def load_summarizer(self, model_name="google/flan-t5-large"):
        """Load summarization model"""
        self.summarizer = LocalSummarizer(model_name, self.device)
        return self.summarizer
    
    def load_decomposer(self, model_name="google/flan-t5-base"):
        """Load atomic fact decomposer"""
        self.decomposer = AtomicFactDecomposer(model_name, self.device)
        return self.decomposer
    
    def load_kg_extractor(self, method="rebel"):
        """Load knowledge graph extractor"""
        self.kg_extractor = KnowledgeGraphExtractor(method, self.device)
        return self.kg_extractor
    
    def load_verifier(self, model_name="microsoft/deberta-v3-large-mnli"):
        """Load fact verifier"""
        self.verifier = LocalFactVerifier(model_name, self.device)
        return self.verifier
    
    def load_all(self):
        """Load tất cả models"""
        self.load_summarizer()
        self.load_decomposer()
        self.load_kg_extractor()
        self.load_verifier()
        logger.info("All models loaded successfully")


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = LocalModelManager(device="cuda")
    manager.load_all()
    
    # Test summarization
    script = "John enters the room. Mary is sitting by the window. John says hello to Mary."
    summary = manager.summarizer.generate_summary(script)
    print(f"Summary: {summary}")
    
    # Test atomic fact decomposition
    facts = manager.decomposer.decompose(summary)
    print(f"Atomic facts: {facts}")
    
    # Test KG extraction
    entities, relations = manager.kg_extractor.extract_named_entities_and_relations(script)
    print(f"Entities: {entities}")
    print(f"Relations: {relations}")
    
    # Test fact verification
    for fact in facts:
        score, feedback = manager.verifier.verify_fact(script, fact)
        print(f"Fact: {fact}")
        print(f"Score: {score}, Feedback: {feedback}")
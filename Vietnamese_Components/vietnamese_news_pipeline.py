"""
Pipeline hoàn chỉnh cho Vietnamese News Summarization
Xử lý JSONL format với single_documents
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm

from vietnamese_model_replacements import VietnameseModelManager
from adapted_knowledge_graph import generate_knowledge_graph_for_scripts_local
from ..kg.generate_kg import save_knowledge_graph, refine_kg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VietnameseNewsPipeline:
    """
    Pipeline đầy đủ cho Vietnamese News
    
    Workflow:
    1. Load JSONL data
    2. Generate initial summary  
    3. Decompose into atomic facts
    4. Build knowledge graph
    5. Verify factuality
    6. Self-correction
    7. Iterate
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        logger.info("Initializing Vietnamese News Pipeline...")
        
        # Load Vietnamese models
        self.model_manager = VietnameseModelManager(device=device)
        self.model_manager.load_all()
        
        logger.info("Vietnamese News Pipeline initialized")
    
    def load_jsonl_data(self, jsonl_path: str) -> List[Dict]:
        """
        Load data từ JSONL file
        
        Format mỗi dòng:
        {
            "single_documents": [
                {"title": "...", "anchor_text": "...", "raw_text": "..."},
                ...
            ],
            "summary": "...",  # Reference summary (optional)
            "category": "..."
        }
        """
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    data['id'] = f'news_{line_num:04d}'
                    data_list.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
        
        logger.info(f"Loaded {len(data_list)} news items from {jsonl_path}")
        return data_list
    
    def prepare_text_from_documents(self, single_documents: List[Dict]) -> str:
        """
        Combine multiple documents thành một text
        """
        combined_text = ""
        for doc in single_documents:
            title = doc.get('title', '')
            raw_text = doc.get('raw_text', '')
            if title:
                combined_text += f"{title}\n\n"
            combined_text += f"{raw_text}\n\n"
        
        return combined_text.strip()
    
    def generate_summary(self, text: str) -> str:
        """
        Step 1: Generate summary từ news text
        """
        logger.info("Generating summary...")
        summary = self.model_manager.summarizer.generate_summary(text)
        return summary
    
    def decompose_to_facts(self, summary: str) -> List[str]:
        """
        Step 2: Decompose summary thành atomic facts
        """
        logger.info("Decomposing into atomic facts...")
        facts = self.model_manager.decomposer.decompose(summary)
        logger.info(f"Extracted {len(facts)} atomic facts")
        return facts
    
    def build_knowledge_graph(self, text: str, idx: str, save_path: Path) -> Tuple[any, List[str]]:
        """
        Step 3: Build Knowledge Graph
        
        Returns:
            kg: Knowledge graph object
            kg_triplets: List of KG triplets as strings
        """
        logger.info("Building knowledge graph...")
        
        # Prepare data format for KG extraction
        book = {
            'id': idx,
            'chapters': [
                {'index': 1, 'text': text}
            ]
        }
        
        # Generate KG
        kg = generate_knowledge_graph_for_scripts_local(
            book=book,
            idx=idx,
            method="rebel",
            device=self.device,
            language="vi"
        )
        
        # Save KG
        kg_save_path = save_path / idx / '3_knowledge_graphs'
        save_knowledge_graph(kg, idx, kg_save_path)
        
        # Refine and extract triplets
        refine_kg(idx, save_path / idx, topk=10, refine='ner')
        
        # Load triplets
        kg_triplets = []
        kg_file = kg_save_path / 'final_kg.jsonl'
        if kg_file.exists():
            with open(kg_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    triplet = f"{data['subject']} - {data['predicate']} - {data['object']}"
                    kg_triplets.append(triplet)
        
        logger.info(f"Built KG with {len(kg_triplets)} triplets")
        return kg, kg_triplets
    
    def calculate_factuality(self, text: str, summary: str, facts: List[str], kg_triplets: List[str]) -> Dict:
        """
        Step 4: Calculate factuality score
        """
        logger.info("Calculating factuality scores...")
        
        scores = []
        feedback_list = []
        
        for fact in facts:
            # Find relevant context (simplified - trong thực tế dùng BM25 + BGE)
            # Ở đây đơn giản hóa: verify với toàn bộ text
            
            # Get most relevant KG triplet
            kg_context = kg_triplets[0] if kg_triplets else ""
            
            # Verify fact
            score, feedback = self.model_manager.verifier.verify_fact(
                premise=text,
                hypothesis=fact,
                kg_context=kg_context
            )
            
            scores.append(score)
            feedback_list.append(feedback)
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        result = {
            'overall_score': overall_score,
            'sentence_scores': scores,
            'atomic_facts': facts,
            'feedback': feedback_list
        }
        
        logger.info(f"Factuality score: {overall_score:.4f}")
        return result
    
    def self_correction(self, text: str, summary: str, feedback_list: List[str], facts: List[str]) -> str:
        """
        Step 5: Self-correction based on feedback
        """
        logger.info("Performing self-correction...")
        
        # Collect errors
        errors = []
        for fact, feedback in zip(facts, feedback_list):
            if feedback:
                errors.append(f"- {fact}\n  Vấn đề: {feedback}")
        
        if not errors:
            logger.info("No errors found")
            return summary
        
        error_text = '\n'.join(errors)
        
        # Generate correction prompt (tiếng Việt)
        prompt = f"""Dưới đây là văn bản tin tức gốc và bản tóm tắt ban đầu. Hãy sửa lại bản tóm tắt dựa trên các vấn đề được chỉ ra.

Văn bản gốc: {text}

Bản tóm tắt ban đầu: {summary}

Các vấn đề cần sửa:
{error_text}

Bản tóm tắt đã sửa (giữ độ dài tương tự, không copy trực tiếp văn bản gốc):"""
        
        # Use summarizer model for correction
        corrected = self.model_manager.summarizer.generate_summary(prompt)
        
        logger.info("Self-correction completed")
        return corrected
    
    def process_single_news(self, news_item: Dict, save_root: Path, num_iterations: int = 2) -> Dict:
        """
        Process một news item qua full pipeline
        """
        idx = news_item['id']
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing news: {idx}")
        logger.info(f"{'='*60}")
        
        # Prepare text
        text = self.prepare_text_from_documents(news_item['single_documents'])
        reference_summary = news_item.get('summary', '')
        
        # Step 1: Generate initial summary
        initial_summary = self.generate_summary(text)
        
        # Step 2: Build KG
        kg, kg_triplets = self.build_knowledge_graph(text, idx, save_root)
        
        # Iterative refinement
        current_summary = initial_summary
        history = []
        
        for iteration in range(num_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Step 3: Decompose
            facts = self.decompose_to_facts(current_summary)
            
            # Step 4: Verify
            fact_results = self.calculate_factuality(text, current_summary, facts, kg_triplets)
            
            history.append({
                'iteration': iteration + 1,
                'summary': current_summary,
                'score': fact_results['overall_score'],
                'feedback': fact_results['feedback']
            })
            
            # Step 5: Correct (except last iteration)
            if iteration < num_iterations - 1:
                current_summary = self.self_correction(
                    text, current_summary, fact_results['feedback'], facts
                )
        
        # Prepare results
        results = {
            'id': idx,
            'category': news_item.get('category', ''),
            'source_text': text[:500] + '...',  # Truncate for saving
            'reference_summary': reference_summary,
            'initial_summary': initial_summary,
            'final_summary': current_summary,
            'num_kg_triplets': len(kg_triplets),
            'iteration_history': history
        }
        
        # Save results
        result_path = save_root / idx / 'results.json'
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nCompleted {idx}")
        logger.info(f"Initial score: {history[0]['score']:.4f}")
        logger.info(f"Final score: {history[-1]['score']:.4f}")
        logger.info(f"Improvement: {history[-1]['score'] - history[0]['score']:+.4f}")
        
        return results
    
    def process_jsonl_file(self, jsonl_path: str, output_dir: str, num_iterations: int = 2, max_items: int = None):
        """
        Process toàn bộ JSONL file
        """
        # Load data
        data_list = self.load_jsonl_data(jsonl_path)
        
        if max_items:
            data_list = data_list[:max_items]
        
        save_root = Path(output_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        
        # Process each news item
        all_results = []
        for news_item in tqdm(data_list, desc="Processing news"):
            try:
                results = self.process_single_news(news_item, save_root, num_iterations)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error processing {news_item['id']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary of all results
        summary_path = save_root / 'all_results_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        self._print_statistics(all_results)
        
        return all_results
    
    def _print_statistics(self, all_results: List[Dict]):
        """Print statistics of processing"""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        
        total = len(all_results)
        avg_initial = sum(r['iteration_history'][0]['score'] for r in all_results) / total
        avg_final = sum(r['iteration_history'][-1]['score'] for r in all_results) / total
        avg_improvement = avg_final - avg_initial
        
        print(f"Total news processed: {total}")
        print(f"Average initial score: {avg_initial:.4f}")
        print(f"Average final score: {avg_final:.4f}")
        print(f"Average improvement: {avg_improvement:+.4f}")
        
        # Category breakdown
        from collections import defaultdict
        category_scores = defaultdict(list)
        for r in all_results:
            category_scores[r['category']].append(r['iteration_history'][-1]['score'])
        
        print("\nScores by category:")
        for category, scores in category_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"  {category}: {avg_score:.4f} ({len(scores)} items)")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Vietnamese News Summarization Pipeline')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSONL file')
    parser.add_argument('--output', type=str, default='./output_vietnamese',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of self-correction iterations')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Maximum number of items to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VietnameseNewsPipeline(device=args.device)
    
    # Process JSONL file
    results = pipeline.process_jsonl_file(
        jsonl_path=args.input,
        output_dir=args.output,
        num_iterations=args.iterations,
        max_items=args.max_items
    )
    
    print("\n✅ Processing completed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
"""
Main pipeline tích hợp tất cả local models
Thay thế hoàn toàn OpenAI API
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import argparse

# Import adapted modules
from adapted_scripty_summarizer import ScriptySummarizer
from adapted_knowledge_graph import generate_knowledge_graph_for_scripts_local
from adapted_narrativefactscore import NarrativeFactScore
from local_model_replacements import LocalModelManager

# Import utilities từ code gốc
from kg.generate_kg import save_knowledge_graph, refine_kg
from kg.preprocess import preprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalPipeline:
    """
    Pipeline hoàn chỉnh sử dụng local models
    Workflow:
    1. Generate initial summary
    2. Decompose into atomic facts
    3. Build knowledge graph
    4. Calculate factuality score
    5. Self-correction with feedback
    """
    
    def __init__(
        self,
        device: str = "cuda",
        kg_method: str = "rebel",
        summarizer_model: str = "google/flan-t5-large",
        nli_model: str = "microsoft/deberta-v3-large-mnli"
    ):
        """
        Args:
            device: 'cuda' hoặc 'cpu'
            kg_method: 'rebel' hoặc 'openie'
            summarizer_model: HuggingFace model cho summarization
            nli_model: HuggingFace model cho NLI verification
        """
        self.device = device
        self.kg_method = kg_method
        
        logger.info("Initializing Local Pipeline...")
        
        # Initialize summarizer
        self.summarizer = ScriptySummarizer(
            model=summarizer_model,
            device=device
        )
        
        # Initialize fact scorer
        self.fact_scorer = NarrativeFactScore(
            device=device,
            model=nli_model,
            split_type="local"
        )
        
        logger.info("Local Pipeline initialized successfully")
    
    def generate_initial_summary(
        self,
        script_text: str,
        save_path: str = None
    ) -> str:
        """
        Step 1: Tạo summary ban đầu từ script
        
        Args:
            script_text: Raw script text
            save_path: Path để save summary
            
        Returns:
            summary: Generated summary
        """
        logger.info("Generating initial summary...")
        
        # Format prompt cho summarizer
        prompt = f"""This is a part of a script from a Movie. Read the following content carefully, then answer my question:
{script_text}
The script has ended now.

Please summarize the content:
- Provide a detailed summary of the key characters' actions, emotions, and situations as reflected in the dialogue or context.
- Clearly state the outcome of the events.
- The summary should be between 2 to 5 sentences long.

Summary:"""
        
        summary = self.summarizer.inference_with_gpt(prompt)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({'summary': summary}, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved summary to {save_path}")
        
        return summary
    
    def build_knowledge_graph(
        self,
        scene_list: List[str],
        idx: str,
        save_root: Path,
        name: str
    ) -> Path:
        """
        Step 2: Build Knowledge Graph từ script
        
        Args:
            scene_list: List of scenes
            idx: Script ID
            save_root: Root directory để save
            name: Script name
            
        Returns:
            kg_path: Path to saved KG
        """
        logger.info(f"Building Knowledge Graph for script {idx}...")
        
        # Preprocess script
        preprocessed_script = preprocess(scene_list, idx)
        
        preprocess_path = save_root / f'{idx}_{name}' / '1_preprocessed' / 'script.json'
        os.makedirs(preprocess_path.parent, exist_ok=True)
        
        with open(preprocess_path, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_script, f, indent=4, ensure_ascii=False)
        
        # Build KG using local model
        knowledge_graph = generate_knowledge_graph_for_scripts_local(
            book=preprocessed_script,
            idx=idx,
            method=self.kg_method,
            device=self.device
        )
        
        # Save KG
        kg_save_path = save_root / f'{idx}_{name}' / '3_knowledge_graphs'
        save_knowledge_graph(knowledge_graph, preprocessed_script['id'], kg_save_path)
        
        logger.info(f"Knowledge Graph saved to {kg_save_path}")
        
        # Refine KG
        refine_kg(idx, save_root / f'{idx}_{name}', topk=10, refine='ner')
        
        return kg_save_path
    
    def calculate_factuality(
        self,
        script_text: str,
        summary: str,
        kg_triplets: List[str]
    ) -> Dict:
        """
        Step 3: Calculate factuality score
        
        Args:
            script_text: Original script
            summary: Generated summary
            kg_triplets: List of KG triplets as strings
            
        Returns:
            results: Dict with scores and feedback
        """
        logger.info("Calculating factuality scores...")
        
        # Run NarrativeFactScore
        scores, scores_per_sent, relevant_scenes, summary_chunks, feedback_list = \
            self.fact_scorer.score_src_hyp_long(
                srcs=[script_text],
                hyps=[summary],
                kgs=kg_triplets
            )
        
        results = {
            'overall_score': scores[0],
            'sentence_scores': scores_per_sent[0],
            'relevant_scenes': relevant_scenes[0],
            'atomic_facts': summary_chunks[0],
            'feedback': feedback_list[0]
        }
        
        logger.info(f"Factuality score: {results['overall_score']:.4f}")
        
        return results
    
    def self_correction(
        self,
        script_text: str,
        original_summary: str,
        feedback_list: List[str],
        atomic_facts: List[str]
    ) -> str:
        """
        Step 4: Self-correction dựa trên feedback
        
        Args:
            script_text: Original script
            original_summary: Summary cần sửa
            feedback_list: List feedback cho từng atomic fact
            atomic_facts: List atomic facts
            
        Returns:
            revised_summary: Summary sau khi sửa
        """
        logger.info("Performing self-correction...")
        
        # Tạo prompt cho correction
        errors = []
        for fact, feedback in zip(atomic_facts, feedback_list):
            if feedback:  # Có lỗi
                errors.append(f"Statement: {fact}\nIssue: {feedback}")
        
        if not errors:
            logger.info("No errors found, returning original summary")
            return original_summary
        
        error_text = '\n\n'.join(errors)
        
        prompt = f"""Below is a part of the script from the titled movie.
Script: {script_text}

Based on the following issues, revise the summary to fix the errors.
Keep the revised summary concise and similar in length to the original summary.
Do not directly copy any part of the Script.

Original Summary: {original_summary}

Issues to fix:
{error_text}

Revised Summary:"""
        
        revised_summary = self.summarizer.inference_with_gpt(prompt)
        
        logger.info("Self-correction completed")
        
        return revised_summary
    
    def run_full_pipeline(
        self,
        script_text: str,
        scene_list: List[str],
        idx: str,
        name: str,
        save_root: Path,
        num_iterations: int = 2
    ) -> Dict:
        """
        Chạy full pipeline với multiple iterations
        
        Args:
            script_text: Raw script text
            scene_list: List of scenes
            idx: Script ID
            name: Script name
            save_root: Root directory
            num_iterations: Số lần iteration cho self-correction
            
        Returns:
            final_results: Dict chứa tất cả kết quả
        """
        logger.info(f"Starting full pipeline for script {idx}...")
        
        # Step 1: Generate initial summary
        summary = self.generate_initial_summary(
            script_text,
            save_path=save_root / f'{idx}_{name}' / 'initial_summary.json'
        )
        
        # Step 2: Build Knowledge Graph
        kg_path = self.build_knowledge_graph(scene_list, idx, save_root, name)
        
        # Load KG triplets
        kg_file = kg_path / 'final_kg.jsonl'
        kg_triplets = []
        if kg_file.exists():
            with open(kg_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    triplet = f"{data['subject']} - {data['predicate']} - {data['object']}"
                    kg_triplets.append(triplet)
        
        # Iterative refinement
        current_summary = summary
        history = []
        
        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # Step 3: Calculate factuality
            fact_results = self.calculate_factuality(
                script_text,
                current_summary,
                kg_triplets
            )
            
            history.append({
                'iteration': iteration + 1,
                'summary': current_summary,
                'score': fact_results['overall_score'],
                'feedback': fact_results['feedback']
            })
            
            # Step 4: Self-correction
            if iteration < num_iterations - 1:  # Không correct ở iteration cuối
                current_summary = self.self_correction(
                    script_text,
                    current_summary,
                    fact_results['feedback'],
                    fact_results['atomic_facts']
                )
        
        # Save final results
        final_results = {
            'script_id': idx,
            'script_name': name,
            'initial_summary': summary,
            'final_summary': current_summary,
            'kg_path': str(kg_path),
            'num_kg_triplets': len(kg_triplets),
            'iteration_history': history
        }
        
        results_path = save_root / f'{idx}_{name}' / 'final_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nPipeline completed!")
        logger.info(f"Initial score: {history[0]['score']:.4f}")
        logger.info(f"Final score: {history[-1]['score']:.4f}")
        logger.info(f"Improvement: {history[-1]['score'] - history[0]['score']:.4f}")
        
        return final_results


def main():
    """
    Main function để chạy pipeline
    """
    parser = argparse.ArgumentParser(description='Run local model pipeline')
    parser.add_argument('--script_path', type=str, required=True,
                        help='Path to script file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--kg_method', type=str, default='rebel',
                        help='KG extraction method: rebel or openie')
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of self-correction iterations')
    
    args = parser.parse_args()
    
    # Load script
    with open(args.script_path, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    script_text = script_data.get('text', '')
    scene_list = script_data.get('scenes', [script_text])
    idx = script_data.get('id', '0')
    name = script_data.get('name', 'script')
    
    # Initialize pipeline
    pipeline = LocalPipeline(
        device=args.device,
        kg_method=args.kg_method
    )
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        script_text=script_text,
        scene_list=scene_list,
        idx=idx,
        name=name,
        save_root=Path(args.output_dir),
        num_iterations=args.iterations
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Script: {results['script_name']}")
    print(f"Initial Summary: {results['initial_summary'][:200]}...")
    print(f"Final Summary: {results['final_summary'][:200]}...")
    print(f"Score Improvement: {results['iteration_history'][-1]['score'] - results['iteration_history'][0]['score']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
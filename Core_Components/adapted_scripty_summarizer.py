"""
File thay thế cho scripty_summarizer.py
Sử dụng local models thay vì OpenAI API
"""

import os
import logging
from typing import Union, Optional
from local_model_replacements import LocalSummarizer

logger = logging.getLogger(__name__)


class ScriptySummarizer:
    """
    Thay thế OpenAI API bằng local summarization model
    Interface giống y hệt class gốc để dễ dàng thay thế
    """
    def __init__(self,
                 model: str = "google/flan-t5-large",
                 seed: int = 42,
                 device: str = "cuda") -> None:
        """
        Args:
            model: Tên model từ HuggingFace (thay vì OpenAI model)
            seed: Random seed
            device: 'cuda' hoặc 'cpu'
        """
        self.model = model
        self.seed = seed
        self.device = device
        
        # Initialize local summarizer
        self.summarizer = LocalSummarizer(model_name=model, device=device)
        logger.info(f"Initialized ScriptySummarizer with {model}")
    
    def inference_with_gpt(self, prompt: str) -> str:
        """
        Thay thế method gốc - giữ nguyên tên để tương thích
        Nhưng thực tế sử dụng local model
        
        Args:
            prompt: Prompt chứa script cần summarize
            
        Returns:
            summary: Bản tóm tắt được tạo ra
        """
        try:
            # Extract script từ prompt nếu có template
            script_text = self._extract_script_from_prompt(prompt)
            
            # Generate summary using local model
            response = self.summarizer.generate_summary(script_text)
            
            logger.info(f"Generated summary (length: {len(response)})")
            return response
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return ''
    
    def _extract_script_from_prompt(self, prompt: str) -> str:
        """
        Extract script content từ prompt template
        """
        # Nếu prompt có format đặc biệt, parse ra script
        # Giả sử format: "Summarize this: {script}"
        if "Script:" in prompt:
            parts = prompt.split("Script:")
            if len(parts) > 1:
                script = parts[1].split("Summary:")[0].strip()
                return script
        
        # Fallback: return toàn bộ prompt
        return prompt
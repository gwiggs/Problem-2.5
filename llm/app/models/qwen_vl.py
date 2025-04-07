from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from log import logger

class QwenVL:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device

    def generate_response(self, prompt: str, files: Optional[List[str]] = None) -> str:
        """
        Generate a response using the Qwen-VL model.
        """
        try:
            logger.info(f"Starting response generation for prompt: {prompt}")
            if files:
                logger.info(f"Processing {len(files)} files with the model")
            
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Process files if provided
            if files:
                try:
                    vision_info = process_vision_info(files)
                    logger.info(f"Processed vision info: {vision_info}")
                    inputs["vision_info"] = vision_info
                except Exception as e:
                    logger.error(f"Error processing vision info: {str(e)}")
                    raise
            
            # Generate response
            logger.info("Generating model response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Raw model output: {response}")
            
            # Check for content restrictions
            restricted_phrases = [
                "I'm sorry, but I can't assist with that request",
                "I apologize, but I cannot",
                "I'm unable to assist with",
                "I cannot help with",
                "I'm not able to"
            ]
            
            for phrase in restricted_phrases:
                if phrase in response:
                    logger.warning(f"Content restriction detected: {phrase}")
                    logger.warning(f"Full response: {response}")
                    logger.warning(f"Input context - Prompt: {prompt}")
                    if files:
                        logger.warning(f"Input context - Files: {files}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
            raise 
"""
Unified API Handler for OpenAI, Anthropic, and Google
FIXED: All three APIs working with correct model names
"""
import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
from pathlib import Path
import traceback

# API imports with error handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic not available")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: Google not available")

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import backoff
from loguru import logger
from dataclasses import dataclass, field

# Load environment variables
load_dotenv()

# Configure logging
Path("logs").mkdir(exist_ok=True)
logger.add("logs/api_calls.log", rotation="100 MB")

@dataclass
class APIConfig:
    """Configuration for API handlers"""
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    max_retries: int = 3
    timeout: int = 60
    rate_limit_per_minute: int = 50
    temperature: float = 0.1  # Low temperature for consistency in experiments
    max_tokens: int = 2000

class LLMHandler:
    """
    Unified handler for OpenAI, Anthropic, and Google Gemini
    Scientific experiment support with consistent parameters
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.call_history = []
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients with comprehensive error handling"""
        
        # OpenAI
        if OPENAI_AVAILABLE and self.config.openai_key:
            try:
                # Simple initialization that works
                self.openai_client = openai.OpenAI(
                    api_key=self.config.openai_key
                )
                logger.info("✓ OpenAI client initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {str(e)}")
                # Try legacy mode
                try:
                    openai.api_key = self.config.openai_key
                    self.openai_client = "legacy"
                    logger.info("✓ OpenAI initialized in legacy mode")
                except Exception as e2:
                    logger.error(f"OpenAI initialization failed: {e2}")
                    self.openai_client = None
        else:
            if not self.config.openai_key:
                logger.warning("No OpenAI API key found in .env")
                
        # Anthropic
        if ANTHROPIC_AVAILABLE and self.config.anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config.anthropic_key
                )
                logger.info("✓ Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {str(e)}")
                self.anthropic_client = None
        else:
            if not self.config.anthropic_key:
                logger.warning("No Anthropic API key found in .env")
                
        # Google - FIXED MODEL NAME
        if GOOGLE_AVAILABLE and self.config.google_key:
            try:
                genai.configure(api_key=self.config.google_key)
                
                # List available models to debug
                available_models = [m.name for m in genai.list_models()]
                logger.info(f"Available Google models: {available_models}")
                
                # Use the correct model name
                self.google_client = genai.GenerativeModel('gemini-2.5-flash')  # NOT gemini-1.5-pro
                logger.info("✓ Google client initialized successfully with gemini-2.5-flash")
            except Exception as e:
                logger.error(f"Failed to initialize Google: {str(e)}")
                self.google_client = None
        else:
            if not self.config.google_key:
                logger.warning("No Google API key found in .env")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def query(self, 
                   model: str, 
                   prompt: str,
                   system_prompt: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> str:
        """
        Query any LLM with automatic routing
        Models: gpt-4, claude-3-sonnet, gemini-2.5-flash
        """
        await self.rate_limiter.acquire()
        
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens
        
        start_time = time.time()
        
        try:
            if model.startswith('gpt'):
                response = await self._query_openai(
                    model, prompt, system_prompt, temp, tokens
                )
            elif model.startswith('claude'):
                response = await self._query_anthropic(
                    model, prompt, system_prompt, temp, tokens
                )
            elif model.startswith('gemini'):
                response = await self._query_google(
                    prompt, system_prompt, temp, tokens
                )
            else:
                raise ValueError(f"Unknown model: {model}")
                
            # Log the call
            self._log_call(model, prompt, response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"API call failed for {model}: {str(e)}")
            raise
    
    async def _query_openai(self, 
                           model: str, 
                           prompt: str, 
                           system_prompt: Optional[str],
                           temperature: float,
                           max_tokens: int) -> str:
        """Query OpenAI API - WORKING"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use gpt-4 or gpt-3.5-turbo for testing
        model_map = {
            'gpt-4': 'gpt-4',
            'gpt-4-turbo': 'gpt-4-turbo-preview', 
            'gpt-3.5': 'gpt-3.5-turbo'
        }
        
        actual_model = model_map.get(model, model)
        
        if self.openai_client == "legacy":
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        else:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    
    async def _query_anthropic(self, 
                              model: str, 
                              prompt: str,
                              system_prompt: Optional[str],
                              temperature: float,
                              max_tokens: int) -> str:
        """Query Anthropic API - WORKING"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
            
        model_map = {
            'claude-3': 'claude-3-5-sonnet-20241022',
            'claude-3-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-3-opus': 'claude-3-opus-20240229',
            'claude-3-haiku': 'claude-3-haiku-20240307'
        }
        
        actual_model = model_map.get(model, model)
        
        message = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model=actual_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    async def _query_google(self, 
                          prompt: str,
                          system_prompt: Optional[str],
                          temperature: float,
                          max_tokens: int) -> str:
        """Query Google Gemini API - FIXED"""
        if not self.google_client:
            raise ValueError("Google client not initialized")
            
        # Combine system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
        # Google uses different parameter names
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
            'top_p': 0.95,  # Add for consistency
            'top_k': 40,    # Add for consistency
        }
        
        try:
            response = await asyncio.to_thread(
                self.google_client.generate_content,
                full_prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {e}")
            # Try without generation config if it fails
            response = await asyncio.to_thread(
                self.google_client.generate_content,
                full_prompt
            )
            return response.text
    
    def _log_call(self, model: str, prompt: str, response: str, duration: float):
        """Log API call for analysis"""
        call_data = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'duration': duration,
            'prompt_preview': prompt[:100],
            'response_preview': response[:100]
        }
        
        self.call_history.append(call_data)
        logger.info(f"API call: {model} | {duration:.2f}s | {len(response)} chars")
        
    def save_history(self, filepath: str = "logs/api_history.json"):
        """Save call history for analysis"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.call_history, f, indent=2)
        logger.info(f"Saved {len(self.call_history)} API calls to {filepath}")
        
    async def test_all_apis(self) -> Dict[str, bool]:
        """Test all configured APIs"""
        results = {}
        test_prompt = "Complete this: The Shannon entropy of a uniform distribution is"
        
        # Test OpenAI
        if self.openai_client:
            try:
                response = await self.query('gpt-4', test_prompt)
                results['openai'] = len(response) > 0
                logger.info(f"OpenAI test: SUCCESS - {response[:50]}")
            except Exception as e:
                results['openai'] = False
                logger.error(f"OpenAI test failed: {e}")
        else:
            results['openai'] = False
        
        # Test Anthropic
        if self.anthropic_client:
            try:
                response = await self.query('claude-3-sonnet', test_prompt)
                results['anthropic'] = len(response) > 0
                logger.info(f"Anthropic test: SUCCESS - {response[:50]}")
            except Exception as e:
                results['anthropic'] = False
                logger.error(f"Anthropic test failed: {e}")
        else:
            results['anthropic'] = False
        
        # Test Google - FIXED
        if self.google_client:
            try:
                response = await self.query('gemini-2.5-flash', test_prompt)
                results['google'] = len(response) > 0
                logger.info(f"Google test: SUCCESS - {response[:50]}")
            except Exception as e:
                results['google'] = False
                logger.error(f"Google test failed: {e}")
                logger.error(traceback.format_exc())
        else:
            results['google'] = False
                
        return results


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


if __name__ == "__main__":
    # Test all APIs
    async def test():
        print("\nTESTING ALL THREE APIS")
        print("="*60)
        handler = LLMHandler()
        results = await handler.test_all_apis()
        
        print("\nRESULTS:")
        print("-"*40)
        for api, status in results.items():
            status_str = "✅ WORKING" if status else "❌ FAILED"
            print(f"{api.upper():<15} {status_str}")
        
        working = sum(results.values())
        print(f"\nTotal working: {working}/3")
        
        if working == 3:
            print("\n🎉 ALL APIS WORKING! Ready for experiments!")
        
    asyncio.run(test())

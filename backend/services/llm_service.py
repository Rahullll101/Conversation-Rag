import logging
import requests
from backend.config.settings import settings

logger = logging.getLogger(__name__)

def generate_answer(prompt: str) -> str:
    """
    Synchronous wrapper for internal services (like evaluation) that do not need streaming.
    """
    return "".join(list(generate_answer_stream(prompt)))

def generate_answer_stream(prompt: str):
    """
    Calls the OpenRouter/OpenAI API to generate an answer and yields tokens as they stream in.
    """
    logger.info(f"Initiating streaming LLM generation (model: {settings.llm_model_name})")
    
    if not settings.openrouter_api_key or settings.openrouter_api_key.strip() == "" or "YOUR_OPENROUTER_API_KEY_HERE" in settings.openrouter_api_key:
        logger.warning("No OpenRouter API key found. Using deterministic mock fallback.")
        yield _mock_generate(prompt)
        return

    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )

        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            stream=True,
            # If the model supports reasoning, this enables it
            extra_body={"reasoning": {"enabled": True}}
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        error_msg = f"[SYSTEM ERROR] OpenRouter/OpenAI API failed: {str(e)}"
        logger.error(error_msg)
        yield error_msg

def _mock_generate(prompt: str) -> str:
    """
    Deterministic mock fallback that enforces the strict grounding rules.
    If the prompt doesn't seem to contain much context, it refuses.
    Otherwise it fakes an answer.
    """
    # Extremely simple heuristic for tests to simulate hallucination guardrails
    if "Chunk 1:" not in prompt:
        return "I could not find enough information in the document."
        
    return "[MOCK ANSWER] Based on the context, this is a simulated grounded answer. Please provide a valid OpenRouter API key for a real response."

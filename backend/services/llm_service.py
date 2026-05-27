import logging
import os
from typing import Any

from backend.config.settings import settings

logger = logging.getLogger(__name__)

def generate_answer(prompt: str) -> str:
    """
    Synchronous wrapper for internal services (like evaluation) that do not need streaming.
    """
    return "".join(list(generate_answer_stream(prompt)))

def generate_answer_stream(prompt: str):
    """
    Calls the configured LangChain/OpenRouter LLM and yields tokens as they stream in.
    """
    logger.info(f"Initiating streaming LLM generation (model: {settings.llm_model_name})")
    
    if not settings.openrouter_api_key or settings.openrouter_api_key.strip() == "" or "YOUR_OPENROUTER_API_KEY_HERE" in settings.openrouter_api_key:
        logger.warning("No OpenRouter API key found. Using deterministic mock fallback.")
        yield _mock_generate(prompt)
        return

    if settings.llm_orchestration.lower() == "langchain":
        try:
            yield from _generate_with_langchain_openrouter(prompt)
            return
        except ImportError as exc:
            if not settings.openai_sdk_fallback_enabled:
                raise
            logger.warning("LangChain OpenRouter unavailable; using OpenAI SDK fallback: %s", exc)
        except Exception as exc:
            error_msg = f"[SYSTEM ERROR] LangChain/OpenRouter generation failed: {str(exc)}"
            logger.error(error_msg)
            yield error_msg
            return

    yield from _generate_with_openai_sdk(prompt)


def _generate_with_langchain_openrouter(prompt: str):
    from langchain_core.messages import HumanMessage
    from langchain_openrouter import ChatOpenRouter

    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
    if settings.openrouter_app_url:
        os.environ["OPENROUTER_APP_URL"] = settings.openrouter_app_url
    if settings.openrouter_app_title:
        os.environ["OPENROUTER_APP_TITLE"] = settings.openrouter_app_title

    model_kwargs: dict[str, Any] = {
        "model": settings.llm_model_name,
        "temperature": settings.temperature,
        "max_tokens": settings.max_generation_tokens,
    }
    if settings.openrouter_app_url:
        model_kwargs["app_url"] = settings.openrouter_app_url
    if settings.openrouter_app_title:
        model_kwargs["app_title"] = settings.openrouter_app_title

    model = ChatOpenRouter(**model_kwargs)
    for chunk in model.stream([HumanMessage(content=prompt)]):
        text = _chunk_content_to_text(chunk)
        if text:
            yield text


def _generate_with_openai_sdk(prompt: str):
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

        extra_headers = {}
        if settings.openrouter_app_url:
            extra_headers["HTTP-Referer"] = settings.openrouter_app_url
        if settings.openrouter_app_title:
            extra_headers["X-Title"] = settings.openrouter_app_title

        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_generation_tokens,
            stream=True,
            extra_headers=extra_headers or None,
            extra_body={"reasoning": {"enabled": True}},
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        error_msg = f"[SYSTEM ERROR] OpenRouter/OpenAI API failed: {str(e)}"
        logger.error(error_msg)
        yield error_msg


def _chunk_content_to_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content else ""

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

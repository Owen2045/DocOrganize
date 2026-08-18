from openai import OpenAI as _OAI
from dotenv import dotenv_values
from app.config import OPENAI_API_KEY
from pathlib import Path

_ENV_FILE = Path(".env")


def get_llm_clients() -> list[tuple[_OAI, str]]:
    """依 .env 的 LLM_PROVIDERS 順序回傳 [(client, model_name), ...]，供 fallback 迴圈與 judge 共用。"""
    env = dotenv_values(_ENV_FILE) if _ENV_FILE.exists() else {}
    providers = [p.strip() for p in (env.get("LLM_PROVIDERS") or "openai").split(",")]
    result = []
    for p in providers:
        if p == "openai":
            result.append(
                (
                    _OAI(api_key=env.get("OPENAI_API_KEY") or OPENAI_API_KEY),
                    env.get("OPENAI_MODEL") or "gpt-4o-mini",
                )
            )
        elif p == "gemini" and env.get("GEMINI_API_KEY"):
            result.append(
                (
                    _OAI(
                        api_key=env.get("GEMINI_API_KEY"),
                        # Gemini 提供相容 OpenAI 的端點，直接用 openai SDK 即可
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    ),
                    env.get("GEMINI_MODEL") or "gemini-2.5-flash",
                )
            )
    return result or [(_OAI(api_key=OPENAI_API_KEY), "gpt-4o-mini")]

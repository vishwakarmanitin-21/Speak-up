from __future__ import annotations

from openai import AsyncOpenAI

from src.config import Config
from src.rewrite.modes import RewriteMode
from src.rewrite.prompts import SYSTEM_PROMPT, build_user_prompt


class RewriteEngine:
    """Sends transcribed text to GPT for intelligent rewriting."""

    def __init__(self) -> None:
        config = Config()
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self._model = config.gpt_model
        self._temperature = config.temperature

    async def rewrite(
        self,
        raw_text: str,
        mode: RewriteMode,
        context: str | None = None,
    ) -> str:
        """Rewrite raw_text using the specified mode.

        Args:
            raw_text: The transcribed speech text.
            mode: The rewrite mode to apply.
            context: Optional context (clipboard, selection, etc.).

        Returns:
            Rewritten text.
        """
        from src.services.error_handler import RewriteError

        user_prompt = build_user_prompt(mode, raw_text, context)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RewriteError(
                f"GPT rewrite failed: {e}",
                "Rewrite failed. Check your API key and internet connection.",
            ) from e

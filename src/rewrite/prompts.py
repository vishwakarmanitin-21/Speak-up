from __future__ import annotations

from src.rewrite.modes import RewriteMode

SYSTEM_PROMPT = """\
You are an intelligent voice thinking assistant.

The user speaks in raw, fragmented, stream-of-consciousness thoughts.

Your job:
- Clean grammar
- Preserve meaning
- Improve clarity
- Structure when requested
- Never hallucinate new content
- Maintain professional tone unless otherwise specified
- Always return clean formatted text."""

MODE_PROMPTS: dict[RewriteMode, str] = {
    RewriteMode.CLEAN_GRAMMAR: (
        "Rewrite the following text clearly. Fix grammar, punctuation, and "
        "sentence structure. Do not add new content. Preserve the original "
        "meaning exactly."
    ),
    RewriteMode.STRUCTURED_NOTES: (
        "Convert the following raw thoughts into well-structured notes with "
        "clear headings, bullet points, and logical grouping."
    ),
    RewriteMode.CONVERT_TO_PRD: (
        "Convert the following into a structured Product Requirements Document "
        "with these sections:\n"
        "- Vision\n"
        "- Features\n"
        "- User Flow\n"
        "- Technical Considerations"
    ),
    RewriteMode.PROFESSIONAL_EMAIL: (
        "Rewrite the following as a professional email. Include an appropriate "
        "subject line suggestion, greeting, body, and sign-off."
    ),
    RewriteMode.LINKEDIN_POST: (
        "Convert the following into an engaging LinkedIn post with:\n"
        "- A strong hook (first line)\n"
        "- A story or context\n"
        "- An insight or lesson\n"
        "- A clear takeaway or call to action"
    ),
    RewriteMode.DEVELOPER_COMMENT: (
        "Rewrite the following as concise, clear developer documentation "
        "comments. Use appropriate formatting for code documentation."
    ),
    RewriteMode.BRAIN_DUMP: (
        "The following is a raw brain dump of ideas. Organize it into a "
        "clear, structured output with:\n"
        "- Key themes identified\n"
        "- Ideas grouped logically\n"
        "- Action items extracted (if any)\n"
        "- Summary at the top"
    ),
}


def build_user_prompt(
    mode: RewriteMode,
    raw_text: str,
    context: str | None = None,
) -> str:
    """Build the full user prompt with mode instruction and optional context."""
    parts = [MODE_PROMPTS[mode]]
    if context:
        parts.append(f"\n\n--- Context ---\n{context}\n--- End Context ---")
    parts.append(f"\n\n--- Input ---\n{raw_text}\n--- End Input ---")
    return "\n".join(parts)

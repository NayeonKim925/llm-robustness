"""Single source of truth for prompt construction and chat formatting.

The original notebooks re-declared ``SYSTEM_PROMPT`` / ``build_prompt`` /
``format_sample`` in three or four places, and -- critically -- the confidence
notebook fed the model a *raw* prompt string while the fine-tuning and
generation-evaluation notebooks used the chat template. Training on one format
and scoring on another is a silent train/inference mismatch.

Keeping every prompt in one module makes that class of bug structurally
impossible: training, generation and calibration all import the same functions.
"""

from __future__ import annotations

from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a stance classification expert for social media discussions about "
    "rumours.\n\n"
    "Classify the stance of the TARGET reply using exactly one of these four "
    "labels:\n\n"
    "- support: The reply explicitly states the rumour IS true or confirmed.\n"
    "- deny: The reply explicitly states the rumour IS false or fabricated.\n"
    "- query: The reply asks for sources, evidence, or verification.\n"
    "- comment: Everything else. The reply does not directly address whether "
    "the rumour is true or false.\n\n"
    "Respond with ONLY one word: support, deny, query, or comment. No explanation."
)

# Human-readable rewrites of the structural markers used when the conditions
# were built (see :mod:`data`).
_MARKER_REWRITES = {
    "[Source]": "Rumour post:",
    "[Context]": "Previous reply:",
    "[Misleading]": "Another reply:",
    "[Target]": "Reply to classify:",
}


def _clean(text: str) -> str:
    for marker, replacement in _MARKER_REWRITES.items():
        text = text.replace(marker, replacement)
    return text


def build_messages(context_text: str) -> List[Dict[str, str]]:
    """Build the chat-format message list for one example.

    ``context_text`` is one of the condition strings produced by
    :func:`data.build_conditions` (e.g. the ``useful`` or ``lexical`` field).
    """
    user_content = (
        "Read the following and classify the stance of the 'Reply to classify'.\n\n"
        f"{_clean(context_text)}\n\n"
        "Stance label (support / deny / query / comment):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def format_for_sft(context_text: str, label: str, tokenizer) -> str:
    """Render a full training string (prompt + gold label) via the chat template.

    Using ``tokenizer.apply_chat_template`` here -- rather than hand-writing
    ``<|im_start|>`` tokens as the original notebook did -- guarantees the
    training format matches exactly what the model sees at inference time.
    """
    messages = build_messages(context_text) + [{"role": "assistant", "content": label}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_inference_text(context_text: str, tokenizer) -> str:
    """Render the prompt up to the assistant turn, ready for generation/scoring."""
    messages = build_messages(context_text)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

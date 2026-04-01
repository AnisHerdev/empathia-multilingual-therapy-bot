"""
multilingual_pipeline.py
========================
Shared module imported by all project phases.

Provides:
  - MultilingualPipeline  : wraps XLM-RoBERTa tokenizer for NER / Phase 3
  - detect_language()     : lightweight language detection for any text
  - build_chat_prompt()   : formats a conversation turn for Qwen2.5 fine-tuning

Usage (any phase):
    from multilingual_pipeline import MultilingualPipeline, detect_language, build_chat_prompt
    pipe = MultilingualPipeline()
    lang = detect_language("Mujhe bahut anxiety ho rahi hai")
    tokens = pipe.encode("I feel bahut anxious", max_length=128)
"""

from __future__ import annotations

import re
from typing import Optional

import torch
from transformers import AutoTokenizer

# ── Language detection ─────────────────────────────────────────────────────────

# Unicode ranges for script detection
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN_ONLY = re.compile(r"^[A-Za-z\s\d.,!?'\"-]+$")

_SPANISH_MARKERS = {
    "siento", "estoy", "muy", "pero", "porque", "quiero",
    "tengo", "para", "como", "cuando", "que", "me", "mi",
    "es", "en", "con", "por", "una", "del",
}
_HINDI_LATIN_MARKERS = {
    "mujhe", "lagta", "hai", "nahi", "bahut", "kabhi",
    "hua", "toh", "aur", "kya", "hoon", "kar", "bhi",
    "agar", "lekin", "phir", "sab", "apna", "isko",
}


def detect_language(text: str) -> str:
    """
    Returns one of: 'hi' | 'es' | 'en' | 'hi-en' | 'es-en'

    Logic:
      1. Devanagari script → 'hi'
      2. Pure Latin → check word overlap with Hindi/Spanish marker sets
      3. Mixed → label as code-switched ('hi-en' or 'es-en')
      4. Default → 'en'
    """
    if not text or not text.strip():
        return "en"

    # Devanagari script present → Hindi (possibly mixed but call it hi)
    if _DEVANAGARI.search(text):
        return "hi"

    words = set(text.lower().split())

    hi_hits = words & _HINDI_LATIN_MARKERS
    es_hits = words & _SPANISH_MARKERS

    has_english = bool(words - _HINDI_LATIN_MARKERS - _SPANISH_MARKERS)

    if hi_hits and has_english:
        return "hi-en"
    if es_hits and has_english:
        return "es-en"
    if hi_hits:
        return "hi"
    if es_hits:
        return "es"
    return "en"


# ── Prompt builder for Qwen2.5 ─────────────────────────────────────────────────

SYSTEM_THERAPIST = (
    "You are a compassionate, licensed therapist. "
    "Listen carefully, reflect the user's feelings back to them, "
    "and ask one open-ended follow-up question. "
    "Never give direct advice unless explicitly asked. "
    "Respond in the same language or language mix the user used."
)

SYSTEM_FRIEND = (
    "You are a warm, supportive friend who genuinely cares. "
    "Respond naturally and conversationally. "
    "Acknowledge feelings first, then gently offer perspective if helpful. "
    "Match the user's language and tone exactly."
)

STYLES = {"therapist": SYSTEM_THERAPIST, "friend": SYSTEM_FRIEND}


def build_chat_prompt(
    user_message: str,
    conversation_history: Optional[list[dict]] = None,
    style: str = "therapist",
    situation: Optional[str] = None,
    emotion: Optional[str] = None,
) -> list[dict]:
    """
    Builds a Qwen2.5 chat-format message list.

    Args:
        user_message        : the current user turn
        conversation_history: list of {"role": ..., "content": ...} dicts (prior turns)
        style               : "therapist" or "friend"
        situation           : optional EmpatheticDialogues situation context
        emotion             : optional emotion label to prime the model

    Returns:
        messages list ready for tokenizer.apply_chat_template()

    Example:
        messages = build_chat_prompt("I feel bahut anxious", style="therapist")
        # pass to: tokenizer.apply_chat_template(messages, tokenize=False)
    """
    system_text = STYLES.get(style, SYSTEM_THERAPIST)

    # Optionally inject situation/emotion context into system prompt
    if situation or emotion:
        context_parts = []
        if emotion:
            context_parts.append(f"The user is feeling: {emotion}")
        if situation:
            context_parts.append(f"Context: {situation}")
        system_text = system_text + "\n\n" + " | ".join(context_parts)

    messages = [{"role": "system", "content": system_text}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message})
    return messages


# ── MultilingualPipeline class ─────────────────────────────────────────────────

class MultilingualPipeline:
    """
    Wraps the XLM-RoBERTa tokenizer for use in Phase 3 NER / embedding tasks.

    This is intentionally separate from Qwen2.5's own tokenizer.
    - MultilingualPipeline  → used by Phase 3 (NER, sentiment, embeddings)
    - Qwen2.5 tokenizer     → used by Phase 2 (fine-tuning) and Phase 4 (inference)

    Args:
        model_name : HuggingFace model id. Defaults to xlm-roberta-base.
        device     : "cuda", "cpu", or None (auto-detects).
    """

    MODEL_NAME = "xlm-roberta-base"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MultilingualPipeline] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[MultilingualPipeline] Ready on device: {self.device}")

    def encode(
        self,
        text: str | list[str],
        max_length: int = 128,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        """
        Tokenizes text and returns input_ids + attention_mask.

        Args:
            text           : single string or list of strings
            max_length     : pad / truncate to this length
            return_tensors : "pt" (PyTorch) or "np" (NumPy)
            padding        : pad to max_length
            truncation     : truncate if longer than max_length

        Returns:
            dict with keys: input_ids, attention_mask (tensors on self.device)

        Example:
            pipe = MultilingualPipeline()
            encoded = pipe.encode("Mujhe bahut anxiety hai", max_length=64)
            # encoded["input_ids"].shape → torch.Size([1, 64])
        """
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        if return_tensors == "pt":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded

    def decode(self, token_ids) -> str:
        """Converts token IDs back to a string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def tokenize_raw(self, text: str) -> list[str]:
        """Returns raw subword tokens (useful for NER alignment)."""
        return self.tokenizer.tokenize(text)

    def detect_language(self, text: str) -> str:
        """Convenience wrapper — same as module-level detect_language()."""
        return detect_language(text)

    def encode_batch_for_ner(
        self,
        texts: list[str],
        max_length: int = 128,
    ) -> dict:
        """
        Encodes a batch of texts for NER inference.
        Returns input_ids, attention_mask, and offset_mapping
        so NER predictions can be aligned back to character positions.

        Example:
            batch = pipe.encode_batch_for_ner(["I feel anxious", "Mujhe dukh hai"])
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return {"encoded": encoded, "offset_mapping": offset_mapping}

    def __repr__(self) -> str:
        return (
            f"MultilingualPipeline(model='{self.model_name}', device='{self.device}')"
        )

import os
import json
from typing import List

from openai import OpenAI


class GroqDenoiser:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        base_url: str = "https://api.groq.com/openai/v1",
        api_key: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Please set GROQ_API_KEY in your environment.")
        self.model = model or os.environ.get("GROQ_MODEL", "whisper-large-v3")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def denoise_tokens(self, tokens: List[int], chunk_size: int = 512, overlap: int = 32) -> List[int]:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        if len(tokens) <= chunk_size:
            return self._denoise_chunk(tokens)

        cleaned: List[int] = [],3
        step = chunk_size - overlap
        for start in range(0, len(tokens), step):
            chunk = tokens[start : start + chunk_size]
            if not chunk:
                break
            out = self._denoise_chunk(chunk)
            if len(out) != len(chunk):
                # Fallback: keep original chunk if model output shape is wrong
                out = chunk
            if start == 0:
                cleaned.extend(out)
            else:
                cleaned.extend(out[overlap:])
        return cleaned[: len(tokens)]

    def _denoise_chunk(self, tokens: List[int]) -> List[int]:
        system = (
            "You are a denoiser for semantic speech unit sequences. "
            "Input is a list of integers (0-499) representing HuBERT kmeans units. "
            "Output must be JSON with key 'tokens' containing a list of integers. "
            "The output list MUST be the same length as input. "
            "Preserve timing/rhythm while removing spurious or inconsistent units."
        )
        user = json.dumps({"tokens": tokens})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content or ""
            data = json.loads(content)
            out = data.get("tokens", [])
            if not isinstance(out, list):
                return tokens
            return [int(x) for x in out]
        except Exception:
            return tokens

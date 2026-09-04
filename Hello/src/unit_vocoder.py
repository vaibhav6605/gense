import os
import urllib.request
from pathlib import Path
import numpy as np
import torch


class UnitHiFiGAN:
    def __init__(self, model_ref: str | None = None, device: str | None = None, n_units: int | None = None):
        try:
            from unit_hifigan import UnitVocoder
        except Exception as e:
            raise ImportError(
                "unit-hifigan is required for audio reconstruction. "
                "Install it with: pip install unit-hifigan"
            ) from e

        self.model_ref = model_ref or os.environ.get("UNIT_HIFIGAN_MODEL")
        if not self.model_ref:
            raise ValueError(
                "UNIT_HIFIGAN_MODEL is not set. Provide a HF repo id or local path "
                "to a Unit-HiFiGAN checkpoint."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if n_units is None:
            n_units_env = os.environ.get("UNIT_HIFIGAN_N_UNITS")
            n_units = int(n_units_env) if n_units_env else 500

        if str(self.model_ref).startswith("http"):
            # Legacy textlesslib-style checkpoints (expects generator.pt at URL)
            generator_url = str(self.model_ref).rstrip("/") + "/generator.pt"
            cache_dir = Path(os.environ.get("UNIT_HIFIGAN_CACHE_DIR", "outputs/.cache/unit_hifigan"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_path = cache_dir / "generator.pt"
            if not local_path.exists():
                urllib.request.urlretrieve(generator_url, local_path)
            self.vocoder = UnitVocoder.from_legacy_pretrained(str(local_path)).to(self.device)
        else:
            self.vocoder = UnitVocoder.from_pretrained(self.model_ref, n_units=n_units).to(self.device)
        self.vocoder.eval()

    def decode(self, tokens, speaker_id: int = 0, style_id: int = 0):
        units = torch.LongTensor(tokens).unsqueeze(0).to(self.device)

        # Unit-HiFiGAN expects speaker/style tensors for some checkpoints
        with torch.no_grad():
            speaker = torch.tensor([[speaker_id]], dtype=torch.long).to(self.device)
            style = torch.tensor([[style_id]], dtype=torch.long).to(self.device)
            try:
                audio = self.vocoder(units, speaker=speaker, style=style)
            except TypeError:
                # Older checkpoints might ignore style or require only speaker
                try:
                    audio = self.vocoder(units, speaker=speaker)
                except TypeError:
                    audio = self.vocoder(units)

        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        audio = audio.squeeze().detach().cpu().numpy()
        return audio.astype(np.float32)

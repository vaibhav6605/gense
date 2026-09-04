import onnxruntime as ort
import numpy as np
import soundfile as sf

class DNSMOS:
    def __init__(self, model_path):
        # Set providers to ['CPUExecutionProvider'] if you haven't fixed the GPU issue yet
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        # The model expects exactly 144160 samples
        self.target_len = 144160 

    def prepare_audio(self, audio):
        # 1. Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # 2. Fix length to exactly 144,160 samples
        length = len(audio)
        if length < self.target_len:
            # Pad with zeros at the end if too short
            audio = np.pad(audio, (0, self.target_len - length), mode='constant')
        elif length > self.target_len:
            # Center crop if too long
            start = (length - self.target_len) // 2
            audio = audio[start : start + self.target_len]

        return audio.astype(np.float32)

    def __call__(self, wav):
        # Prepare the audio to meet ONNX dimensions
        audio = self.prepare_audio(wav)

        # Run inference
        # Input shape needs to be [batch, samples], e.g., [1, 144160]
        scores = self.session.run(None, {self.input_name: audio[None, :]})[0][0]

        return {
            "SIG": round(float(scores[0]), 3),
            "BAK": round(float(scores[1]), 3),
            "OVRL": round(float(scores[2]), 3)
        }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import os
    
    model_file = "sig_bak_ovr.onnx"
    input_file = "enhanced2.wav"

    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found in current directory.")
    elif not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Generate it using infer.py first!")
    else:
        evaluator = DNSMOS(model_file)

        # Load audio
        wav, sr = sf.read(input_file)

        # Pre-processing
        if wav.ndim > 1:
            wav = wav.mean(axis=1) # Convert stereo to mono

        if sr != 16000:
            # You could use librosa.resample here, but for UGP it's better to 
            # ensure your inference script outputs 16k directly.
            print(f"Warning: SR is {sr}. DNSMOS requires 16000Hz for accuracy.")

        # Get scores
        try:
            scores = evaluator(wav)
            print("\n--- DNSMOS Evaluation Results ---")
            print(f"Signal Quality (SIG):     {scores['SIG']}")
            print(f"Background Noise (BAK):   {scores['BAK']}")
            print(f"Overall Quality (OVRL):   {scores['OVRL']}")
            print("---------------------------------")
        except Exception as e:
            print(f"Inference failed: {e}")
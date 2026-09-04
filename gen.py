import numpy as np
import librosa
import soundfile as sf

def add_noise_at_snr(clean_path, output_path, snr_db=20):
    # 1. Load the clean speech
    clean, sr = librosa.load(clean_path, sr=None)
    
    # 2. Generate White Noise (same length as clean speech)
    noise = np.random.normal(0, 1, len(clean))
    
    # 3. Calculate Power (RMS)
    p_clean = np.mean(clean**2)
    p_noise = np.mean(noise**2)
    
    # 4. Calculate scalar for noise to achieve desired SNR
    # SNR_db = 10 * log10(P_clean / (scalar^2 * P_noise))
    scalar = np.sqrt(p_clean / (10**(snr_db / 10) * p_noise))
    
    # 5. Mix
    noisy = clean + scalar * noise
    
    # 6. Normalize to prevent clipping
    if np.max(np.abs(noisy)) > 1.0:
        noisy = noisy / np.max(np.abs(noisy))
        
    # 7. Save
    sf.write(output_path, noisy, sr)
    print(f"Saved {snr_db}dB noisy file to: {output_path}")

# Usage
add_noise_at_snr("enhanced.wav", "rvq.wav", snr_db=20)
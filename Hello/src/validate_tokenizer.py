'''import numpy as np
import soundfile as sf
from local_tokenizer import LocalTokenizer

def validate_basic_sanity():
    """Check basic properties of the tokenizer"""
    print("=== BASIC SANITY CHECKS ===\n")
    
    tokenizer = LocalTokenizer(kmeans_path="models/kmeans_hubert_base_l9_c500.pt")
    
    # Test 1: Token range check
    print("Test 1: Token Range")
    tokens = tokenizer.extract_tokens("input/p234_001.wav")
    
    min_token = tokens.min()
    max_token = tokens.max()
    unique_tokens = len(np.unique(tokens))
    
    print(f"  Min token: {min_token}")
    print(f"  Max token: {max_token}")
    print(f"  Unique tokens: {unique_tokens}")
    print(f"  Expected range: [0, 1023]")
    
    assert min_token >= 0, "❌ Tokens below 0!"
    assert max_token < 1024, "❌ Tokens above 1023!"
    print("  ✅ All tokens in valid range [0, 1023]\n")
    
    # Test 2: Token sequence length
    print("Test 2: Sequence Length")
    wav, sr = sf.read("input/p234_001.wav")
    duration = len(wav) / sr
    
    expected_frames = int(duration * 50)  # HuBERT ~50 frames/sec
    actual_frames = len(tokens)
    ratio = actual_frames / expected_frames
    
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  Expected frames: ~{expected_frames}")
    print(f"  Actual frames: {actual_frames}")
    print(f"  Ratio: {ratio:.2f}")
    
    assert 0.8 < ratio < 1.2, "❌ Frame rate too different!"
    print("  ✅ Frame rate is reasonable (~50 Hz)\n")
    
    # Test 3: Token distribution
    print("Test 3: Token Distribution")
    token_counts = np.bincount(tokens, minlength=1024)
    used_clusters = np.sum(token_counts > 0)
    
    print(f"  Clusters used: {used_clusters}/1024")
    print(f"  Usage ratio: {used_clusters/1024:.2%}")
    
    # Test 4: No constant sequences (sanity)
    print("Test 4: Variability")
    max_consecutive = 0
    current_consecutive = 1
    
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_consecutive += 1
        else:
            max_consecutive = max(max_consecutive, current_consecutive)
            current_consecutive = 1
    
    print(f"  Max consecutive same token: {max_consecutive}")
    
    # Shouldn't have extremely long constant sequences (indicates issue)
    assert max_consecutive < len(tokens) * 0.5, "❌ Too many repeated tokens!"
    print("  ✅ Token sequence has variability\n")
    
    print("=" * 50)
    print("✅ ALL BASIC SANITY CHECKS PASSED!")
    print("=" * 50)

if __name__ == "__main__":
    validate_basic_sanity()

import numpy as np
from local_tokenizer import LocalTokenizer

def validate_consistency():
    """Check that tokenizer is deterministic"""
    print("=== CONSISTENCY CHECKS ===\n")
    
    tokenizer = LocalTokenizer(kmeans_path="models/kmeans_hubert_base_l9_c500.pt")
    
    # Test 1: Same audio, same tokens
    print("Test 1: Determinism")
    tokens1 = tokenizer.extract_tokens("input/p234_001.wav")
    tokens2 = tokenizer.extract_tokens("input/p234_001.wav")
    
    if np.array_equal(tokens1, tokens2):
        print("  ✅ Same audio produces identical tokens")
    else:
        diff = np.sum(tokens1 != tokens2)
        print(f"  ❌ Tokens differ in {diff}/{len(tokens1)} positions!")
        
    # Test 2: Different initializations
    print("\nTest 2: Different Tokenizer Instances")
    tokenizer_a = LocalTokenizer(kmeans_path="models/kmeans_hubert_base_l9_c500.pt")
    tokenizer_b = LocalTokenizer(kmeans_path="models/kmeans_hubert_base_l9_c500.pt")
    
    tokens_a = tokenizer_a.extract_tokens("input/p234_001.wav")
    tokens_b = tokenizer_b.extract_tokens("input/p234_001.wav")
    
    if np.array_equal(tokens_a, tokens_b):
        print("  ✅ Different instances produce identical tokens")
    else:
        diff = np.sum(tokens_a != tokens_b)
        print(f"  ❌ Tokens differ in {diff}/{len(tokens_a)} positions!")
    
    # Test 3: Batch processing consistency
    print("\nTest 3: Multiple Files")
    test_files = [f"input/p234_00{i}.wav" for i in range(1, 5)]
    
    for file in test_files:
        try:
            t1 = tokenizer.extract_tokens(file)
            t2 = tokenizer.extract_tokens(file)
            
            if np.array_equal(t1, t2):
                print(f"  ✅ {file}: Consistent")
            else:
                print(f"  ❌ {file}: Inconsistent!")
        except Exception as e:
            print(f"  ⚠️ {file}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("✅ CONSISTENCY CHECKS COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    validate_consistency()


import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from local_tokenizer import LocalTokenizer

def visualize_tokens():
    """Visualize token sequences"""
    print("=== VISUALIZING TOKENS ===\n")
    
    tokenizer = LocalTokenizer(kmeans_path="models/kmeans_hubert_base_l9_c500.pt")
    
    # Load audio and extract tokens
    audio_path = "input/p234_001.wav"
    wav, sr = sf.read(audio_path)
    tokens = tokenizer.extract_tokens(audio_path)
    
    print(f"Audio: {len(wav)/sr:.2f}s, {len(tokens)} tokens")
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    # Plot 1: Waveform
    time_audio = np.arange(len(wav)) / sr
    axes[0].plot(time_audio, wav, linewidth=0.5)
    axes[0].set_title("Audio Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Token sequence
    time_tokens = np.arange(len(tokens)) / 50  # ~50 tokens/sec
    axes[1].plot(time_tokens, tokens, marker='o', markersize=2, linewidth=0.5)
    axes[1].set_title("Token Sequence Over Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Token ID")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Token histogram
    axes[2].hist(tokens, bins=100, edgecolor='black')
    axes[2].set_title("Token Distribution")
    axes[2].set_xlabel("Token ID")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Token transition matrix (first 100 unique tokens)
    unique_tokens = np.unique(tokens)[:100]
    token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
    
    transition_matrix = np.zeros((len(unique_tokens), len(unique_tokens)))
    for i in range(len(tokens) - 1):
        if tokens[i] in token_to_idx and tokens[i+1] in token_to_idx:
            transition_matrix[token_to_idx[tokens[i]], token_to_idx[tokens[i+1]]] += 1
    
    im = axes[3].imshow(np.log1p(transition_matrix), aspect='auto', cmap='hot')
    axes[3].set_title("Token Transitions (log scale, first 100 unique tokens)")
    axes[3].set_xlabel("Next Token")
    axes[3].set_ylabel("Current Token")
    plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig("outputs/token_visualization.png", dpi=150)
    print("✅ Saved visualization to outputs/token_visualization.png")
    plt.show()
    
    # Print statistics
    print("\n=== TOKEN STATISTICS ===")
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique tokens: {len(unique_tokens)}")
    print(f"Most common token: {np.bincount(tokens).argmax()} (appears {np.bincount(tokens).max()} times)")
    print(f"Token entropy: {-np.sum((np.bincount(tokens)/len(tokens)) * np.log2(np.bincount(tokens)/len(tokens) + 1e-10)):.2f} bits")

if __name__ == "__main__":
    visualize_tokens()'''

import numpy as np
import soundfile as sf
from scipy.spatial.distance import jensenshannon
from local_tokenizer import LocalTokenizer

# -----------------------------
# Utility functions
# -----------------------------

def normalize_tokens(tokens):
    """Convert token sequence to probability distribution"""
    counts = np.bincount(tokens, minlength=1024)
    return counts / counts.sum()

def lcs_length(a, b):
    """Longest Common Subsequence length"""
    dp = np.zeros((len(a)+1, len(b)+1), dtype=np.int32)
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1, j+1] = dp[i, j] + 1
            else:
                dp[i+1, j+1] = max(dp[i, j+1], dp[i+1, j])
    return dp[-1, -1]

# -----------------------------
# Main validation
# -----------------------------

import numpy as np
from local_tokenizer import LocalTokenizer

def compare_token_sequences(
    file_a,
    file_b,
    kmeans_path="models/kmeans_hubert_base_l9_c500.pt",
    max_print=200
):
    print("\n=== TOKEN SEQUENCE COMPARISON ===\n")

    tokenizer = LocalTokenizer(kmeans_path=kmeans_path)

    # Extract tokens
    tokens_a = tokenizer.extract_tokens(file_a)
    tokens_b = tokenizer.extract_tokens(file_b)

    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    print(f"Tokens A length: {len(tokens_a)}")
    print(f"Tokens B length: {len(tokens_b)}\n")

    # Align to same length
    L = min(len(tokens_a), len(tokens_b))
    t1 = tokens_a[:L]
    t2 = tokens_b[:L]

    # Count mismatches
    mismatches = np.sum(t1 != t2)
    match_ratio = 1.0 - mismatches / L

    print("=== SUMMARY ===")
    print(f"Aligned length     : {L}")
    print(f"Matched tokens     : {L - mismatches}")
    print(f"Mismatched tokens  : {mismatches}")
    print(f"Match ratio        : {match_ratio:.2%}\n")

    # Print token sequences (truncated for readability)
    print("=== TOKEN SEQUENCES (TRUNCATED) ===")
    print(f"Showing first {min(max_print, L)} tokens\n")

    print("File A tokens:")
    print(t1[:max_print])

    print("\nFile B tokens:")
    print(t2[:max_print])

    # Show mismatch positions
    print("\n=== FIRST MISMATCH POSITIONS ===")
    mismatch_indices = np.where(t1 != t2)[0]

    if len(mismatch_indices) == 0:
        print("✅ No mismatches — sequences identical")
    else:
        for idx in mismatch_indices[:20]:
            print(
                f"Index {idx:4d}: "
                f"A={t1[idx]:4d} | B={t2[idx]:4d}"
            )

    print("\n" + "=" * 60)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    compare_token_sequences(
        "test.wav",
        "noisy.wav"
    )

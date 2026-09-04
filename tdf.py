import matplotlib.pyplot as plt

# Training steps (in thousands)
steps = [5, 10, 15, 20, 25, 30]

# Entropy values
entropy_128 = [8, 30, 42, 54, 55, 56]
entropy_256 = [10, 35, 51, 69, 74, 75]
entropy_512 = [13, 44, 74, 89, 100, 100]
entropy_1024 = [17, 52, 80, 100, 100, 100]
entropy_2048 = [24, 68, 92, 100, 100, 100]

plt.figure()

plt.plot(steps, entropy_128, marker='o', label='128 codes')
plt.plot(steps, entropy_256, marker='o', label='256 codes')
plt.plot(steps, entropy_512, marker='o', label='512 codes')
plt.plot(steps, entropy_1024, marker='o', label='1024 codes')
plt.plot(steps, entropy_2048, marker='o', label='2048 codes')

plt.xlabel("Training Steps (in thousands)")
plt.ylabel("Token Entropy")
plt.title("Token Entropy vs Training Steps for Different Codebook Sizes")
plt.legend()
plt.grid()

plt.savefig("token_entropy.png", dpi=300)
plt.show()
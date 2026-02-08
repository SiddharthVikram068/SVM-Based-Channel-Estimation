import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# SYSTEM PARAMETERS (Fig. 3)

K = 4          # number of users
N = 32         # number of BS antennas
Tt = 20        # pilot length
SNR_dB = np.arange(-15, 31, 5)
SNR = 10**(SNR_dB/10)
num_trials = 200  # Monte Carlo runs


# 1-bit quantizer
def one_bit_quantize(X):
    return np.sign(X)


# Generate DFT-based pilot matrix (Eq. 11)
def generate_pilots(Tt, K):
    F = np.fft.fft(np.eye(Tt))   # DFT matrix
    X_complex = F[:, :K]         # first K columns used as pilots

    # Convert to real representation
    Xt_real = np.block([
        [np.real(X_complex).T,  np.imag(X_complex).T],
        [-np.imag(X_complex).T, np.real(X_complex).T]
    ])

    return Xt_real


# Generate i.i.d. uncorrelated channel, complex Gaussian entries are used
def generate_channel(N, K):
    H_complex = (np.random.randn(N, K) + 1j*np.random.randn(N, K)) / np.sqrt(2)

    # Convert to real representation
    H_real = np.block([
        [np.real(H_complex), -np.imag(H_complex)],
        [np.imag(H_complex),  np.real(H_complex)]
    ])

    return H_complex, H_real


# svm channel estimation (robust version with check for all +1 or all -1 labels)
def svm_channel_estimation(Yt, Xt, K, N):
    H_est_real = np.zeros((2*N, 2*K))

    for i in range(2*N):
        labels = Yt[i, :]
        # If all labels are +1 or all -1, skip SVM(SVM solver needs atleast two classes)
        if np.all(labels == 1) or np.all(labels == -1):

            # assign small random vector and normalize
            h_tilde = np.random.randn(2*K)
            h_hat = np.sqrt(K) * h_tilde / np.linalg.norm(h_tilde)
            H_est_real[i, :] = h_hat
            continue

        clf = LinearSVC(C=1.0, loss='squared_hinge', max_iter=10000)
        clf.fit(Xt.T, labels)

        h_tilde = clf.coef_.flatten()

        # Normalize as in Eq. (14)
        h_hat = np.sqrt(K) * h_tilde / np.linalg.norm(h_tilde)
        H_est_real[i, :] = h_hat

    # Convert back to complex channel estimate
    H_complex_est = H_est_real[:N, :K] + 1j * H_est_real[N:, :K]
    return H_complex_est


# NMSE metric
def nmse(H_true, H_est):
    return np.linalg.norm(H_true - H_est, 'fro')**2 / (K*N)


# Monte Carlo simulation
Xt = generate_pilots(Tt, K)
nmse_results = []

for snr in SNR:
    errors = []

    for _ in range(num_trials):
        H_complex, H_real = generate_channel(N, K)

        # Generate AWGN noise in real domain
        noise_std = np.sqrt(1/(2*snr))
        Z = noise_std * np.random.randn(2*N, 2*Tt)

        # Received 1-bit quantized pilots
        Yt = one_bit_quantize(H_real @ Xt + Z) 

        # Estimate channel using SVM
        H_est = svm_channel_estimation(Yt, Xt, K, N)

        errors.append(nmse(H_complex, H_est))

    nmse_results.append(np.mean(errors))


# Plot results (like Fig. 3)

plt.figure()
plt.plot(SNR_dB, 10*np.log10(nmse_results), marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.title("SVM-based Channel Estimation (Uncorrelated)")
plt.grid(True)
plt.show()
plt.savefig("nmse_uncorrelated.png")

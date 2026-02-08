import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import circulant


# 1. SYSTEM PARAMETERS

Nc = 256            # Number of subcarriers
K = 2               # Number of users
N = 16              # Number of BS antennas
L_tap = 8           # Number of channel taps
C_svm_param = 1.0   # SVM regularization

snr_db_range = np.array([-15, -10, -5, 0, 5, 10, 15, 20, 25, 30])
num_trials = 150

np.random.seed(0) # seed for reproducibility



def one_bit_quantize_complex(X):
    """1-bit quantization: sign of real and imaginary parts."""
    return np.sign(np.real(X)) + 1j * np.sign(np.imag(X))

def complex_to_real_mat(A):
    # Convert complex matrix A (m x p) to real representation:
    Re = np.real(A)
    Im = np.imag(A)
    top = np.hstack([Re, -Im])
    bot = np.hstack([Im,  Re])
    return np.vstack([top, bot])

def complex_vec_to_real_vec(v):
    # Convert complex vector v (length p) to real vector
    return np.concatenate([np.real(v), np.imag(v)])

def real_vec_to_complex(v):
    #Inverse of complex_vec_to_real_vec.
    p = len(v) // 2
    return v[:p] + 1j * v[p:]


# Pilot generation
def generate_ofdm_pilots(Nc, K):
    """
    Your original pilot design:
    - User 1: even subcarriers
    - User 2: odd subcarriers
    - QPSK symbols on assigned tones
    """
    X_pilots_FD = np.zeros((Nc, K), dtype=complex)

    # User 1 (even indices)
    idx_u1 = np.arange(0, Nc, 2)
    syms_u1 = (2 * np.random.randint(0, 2, len(idx_u1)) - 1) + \
              1j * (2 * np.random.randint(0, 2, len(idx_u1)) - 1)
    X_pilots_FD[idx_u1, 0] = syms_u1

    # User 2 (odd indices)
    idx_u2 = np.arange(1, Nc, 2)
    syms_u2 = (2 * np.random.randint(0, 2, len(idx_u2)) - 1) + \
              1j * (2 * np.random.randint(0, 2, len(idx_u2)) - 1)
    X_pilots_FD[idx_u2, 1] = syms_u2

    return X_pilots_FD


# build training matrix (Eq. 42)
def build_Phi_TD_Ltap(X_pilots_FD, Nc, L_tap):

    # Time-domain pilots (normalization preserved)
    Phi_pilots_TD = np.fft.ifft(X_pilots_FD, axis=0) * np.sqrt(Nc)

    Phi_list = []
    for k in range(K):
        Phi_k = circulant(Phi_pilots_TD[:, k])[:, :L_tap]
        Phi_list.append(Phi_k)

    Phi_train_complex = np.hstack(Phi_list)  
    Phi_train_real = complex_to_real_mat(Phi_train_complex)

    return Phi_train_complex, Phi_train_real


# Channel generation
def generate_ofdm_channel(K, L_tap, N):
    return np.sqrt(1.0 / (2 * L_tap)) * \
           (np.random.randn(K * L_tap, N) + 1j * np.random.randn(K * L_tap, N))


# per-antenna svm estimation
def svm_estimate_h_i(y_i_real, Phi_train_real, K, L_tap, C_svm_param):
    """
    Solve:
    min 0.5||h||^2 + C * sum(xi^2)
    s.t. y_n * (phi_n^T h) >= 1 - xi_n
    """

    dim_h = 2 * K * L_tap
    h_var = cp.Variable(dim_h)
    slacks = cp.Variable(2 * Nc, nonneg=True)

    constraint_expr = cp.multiply(y_i_real, Phi_train_real @ h_var)
    constraints = [constraint_expr >= 1 - slacks]

    obj = 0.5 * cp.sum_squares(h_var) + C_svm_param * cp.sum_squares(slacks)
    problem = cp.Problem(cp.Minimize(obj), constraints)

    try:
        problem.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, verbose=False)
    except:
        problem.solve(solver=cp.SCS, verbose=False)

    # Robust fallback if solver fails
    if h_var.value is None:
        h_tilde = np.random.randn(dim_h)
    else:
        h_tilde = h_var.value

    # Normalize as in paper: sqrt(K) * h / ||h||
    norm_h = np.linalg.norm(h_tilde)
    if norm_h > 0:
        h_tilde = np.sqrt(K) * h_tilde / norm_h

    # Convert back to complex
    return real_vec_to_complex(h_tilde)


# Main simulation
def run_ofdm():
    print(f"Simulating SVM-based OFDM Channel Estimation (Fig 12)")
    print(f"Nc={Nc}, K={K}, L_tap={L_tap}, Antennas={N}")

    nmse_results = []

    for snr_db in snr_db_range:
        snr_linear = 10**(snr_db / 10.0)
        n0 = 1.0 / snr_linear
        nmse_accum = 0.0

        for mc in range(num_trials):

            # Pilot and trainng matrix generation
            X_pilots_FD = generate_ofdm_pilots(Nc, K)
            Phi_train_complex, Phi_train_real = build_Phi_TD_Ltap(
                X_pilots_FD, Nc, L_tap
            )

            # Channel generation
            H_taps_complex = generate_ofdm_channel(K, L_tap, N)

            # Received signal with AWGN noise
            Z_noise = np.sqrt(n0 / 2) * \
                      (np.random.randn(Nc, N) + 1j * np.random.randn(Nc, N))

            Y_unquantized = Phi_train_complex @ H_taps_complex + Z_noise
            Y_quantized = one_bit_quantize_complex(Y_unquantized)

            # SVM estimation for each antenna
            H_hat_all = np.zeros_like(H_taps_complex)

            for i in range(N):
                y_i_complex = Y_quantized[:, i]
                y_i_real = np.concatenate([np.real(y_i_complex),
                                           np.imag(y_i_complex)])

                h_hat_i = svm_estimate_h_i(
                    y_i_real, Phi_train_real, K, L_tap, C_svm_param
                )

                H_hat_all[:, i] = h_hat_i

            # NMSE calculation
            error = np.linalg.norm(H_taps_complex - H_hat_all, 'fro')**2
            nmse_accum += error / (K * N)

        avg_nmse = nmse_accum / num_trials
        nmse_db = 10 * np.log10(avg_nmse)
        nmse_results.append(nmse_db)

        print(f"SNR: {snr_db} dB, NMSE: {nmse_db:.4f} dB")

    # Plotting results
    plt.figure(figsize=(8, 6))
    plt.plot(snr_db_range, nmse_results, 'o-', color='darkred',
             mfc='none', label='Proposed SVM-based (OFDM)')
    plt.xlabel('SNR in dB')
    plt.ylabel('NMSE in dB')
    plt.title('Fig. 12: NMSE vs SNR for OFDM System (Recreated)')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    plt.xticks(np.arange(-15, 31, 5))
    plt.savefig('ofdm_svm.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_ofdm()

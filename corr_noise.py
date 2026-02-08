import numpy as np
import cvxpy as cp
from sklearn.svm import LinearSVC
from scipy.linalg import toeplitz, sqrtm
import matplotlib.pyplot as plt

# --- System Parameters ---
N = 32          # Base Station antennas
K = 4           # Users
Tt = 20         # Pilot length
SNRs_dB = range(-15, 31, 5) # SNR range from -15 to 30 dB

def complex_to_real_channel(H_complex):
    # Converts complex channel to real as per Eq (10).
    return np.hstack([np.real(H_complex), np.imag(H_complex)])

def complex_to_real_pilot(X_complex):
    # Converts complex pilot to real as per Eq (11).
    X_real = np.real(X_complex)
    X_imag = np.imag(X_complex)

    # Stack columns
    top = np.hstack([X_real, X_imag])
    bottom = np.hstack([-X_imag, X_real])
    return np.vstack([top, bottom])

def generate_spatial_correlation_matrix(N, rho=0.5):
    # Generates a Toeplitz correlation matrix for spatial correlation
    col = [rho**i for i in range(N)]
    return toeplitz(col)

def generate_noise(shape, N0, correlation_matrix=None):
    """
    Generates noise. If correlation_matrix is provided, noise is spatially correlated.
    shape: (N, T)
    """
    noise_iid = (np.random.randn(*shape) + 1j * np.random.randn(*shape)) * np.sqrt(N0/2)
    
    if correlation_matrix is not None:
        # Apply spatial correlation
        R_sqrt = sqrtm(correlation_matrix)
        # We assume correlation is across antennas (rows)
        noise_corr = R_sqrt @ noise_iid
        return noise_corr
    else:
        return noise_iid

# --- SVM Channel Estimation Methods ---

def svm_estimator_uncorrelated(Y_real, X_pilot_real, C_param=1.0):
    N_antennas = Y_real.shape[0]
    K2 = X_pilot_real.shape[0] # 2K
    H_hat = np.zeros((N_antennas, K2))
    
    for i in range(N_antennas):
        y_i = Y_real[i, :]
        
        # Fit SVM
        clf = LinearSVC(C=C_param, loss='squared_hinge', penalty='l2', 
                        dual=False, fit_intercept=False, tol=1e-4, max_iter=2000)
        clf.fit(X_pilot_real.T, y_i)
        
        # The weights w correspond to h_{t,i}
        w = clf.coef_.flatten()
        
        # Normalization
        norm_w = np.linalg.norm(w)
        if norm_w > 0:
            h_hat_i = (np.sqrt(K) * w) / norm_w
        else:
            h_hat_i = w
            
        H_hat[i, :] = h_hat_i
        
    return H_hat

def svm_estimator_correlated(Y_real, X_pilot_real, Ck_matrices, C_param=1.0):
    N_antennas = Y_real.shape[0]
    K_users = len(Ck_matrices) # K
    Tt2 = Y_real.shape[1]      # 2Tt
    
    # Optimization Variable
    H_var = cp.Variable((N_antennas, 2 * K_users))
    slack = cp.Variable((N_antennas, Tt2), nonneg=True)
    
    # Objective
    obj_term_1 = 0
    for k in range(K_users):
        # Extract Real and Imaginary parts of the k-th user's channel vector
        h_k_re = H_var[:, k]
        h_k_im = H_var[:, k + K_users]
        h_k_vec = cp.hstack([h_k_re, h_k_im]) # Size 2N
        
        # Real-valued Covariance Matrix
        Ck = Ck_matrices[k]
        Ck_real = np.block([[np.real(Ck), -np.imag(Ck)], 
                            [np.imag(Ck),  np.real(Ck)]])
        
        # C_k is known and fixed. We pre-calculate inverse.
        Ck_inv = np.linalg.inv(Ck_real)
        
        obj_term_1 += cp.quad_form(h_k_vec, Ck_inv)
        
    obj = 0.5 * obj_term_1 + C_param * cp.sum(slack)
    
    # Constraints
    constraints = [
        cp.multiply(Y_real, (H_var @ X_pilot_real)) >= 1 - slack
    ]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS, eps=1e-3) # SCS is a splitting cone solver, good for this.
    
    H_optimal = H_var.value
    
    # Normalization
    if H_optimal is None: return np.zeros((N_antennas, 2*K_users))

    norm_F = np.linalg.norm(H_optimal, 'fro')
    scaling_factor = np.sqrt(K_users * N_antennas) / norm_F
    return H_optimal * scaling_factor

# Main Simulation Loop

def run_simulation():
    # Noise Correlation Matrix
    noise_corr_matrix = generate_spatial_correlation_matrix(N, rho=0.6)
    
    # Channel Correlation Matrices
    channel_corr_matrix = generate_spatial_correlation_matrix(N, rho=0.8)
    Ck_matrices = [channel_corr_matrix for _ in range(K)]

    nmse_uncorr_iid = []
    nmse_uncorr_corr = []
    nmse_corr_iid = []
    nmse_corr_corr = []

    print(f"Simulating for SNRs: {list(SNRs_dB)} dB")

    for snr_db in SNRs_dB:
        snr_linear = 10**(snr_db / 10.0)
        N0 = 1 / snr_linear # Signal power is normalized to 1
        
        mse_u_i, mse_u_c, mse_c_i, mse_c_c = 0, 0, 0, 0
        num_trials = 20 # Reduced trials for demonstration speed
        
        for _ in range(num_trials):
            # Uncorrelated Channel Data Generation
            H_uncorr = (np.random.randn(N, K) + 1j * np.random.randn(N, K)) / np.sqrt(2)
            # Pilot
            X_pilot_complex = (np.random.randn(K, Tt) + 1j * np.random.randn(K, Tt)) / np.sqrt(2) # Random pilot
            
            # Uncorrelated Channel + IID Noise
            Z_iid = generate_noise((N, Tt), N0, correlation_matrix=None)
            Y_u_i = np.sign(np.real(H_uncorr @ X_pilot_complex + Z_iid)) + \
                    1j * np.sign(np.imag(H_uncorr @ X_pilot_complex + Z_iid))
            
            # Uncorrelated Channel + Correlated Noise
            Z_corr = generate_noise((N, Tt), N0, correlation_matrix=noise_corr_matrix)
            Y_u_c = np.sign(np.real(H_uncorr @ X_pilot_complex + Z_corr)) + \
                    1j * np.sign(np.imag(H_uncorr @ X_pilot_complex + Z_corr))
            
            # Pre-process for SVM
            X_real = complex_to_real_pilot(X_pilot_complex)
            Y_u_i_real = complex_to_real_channel(Y_u_i)
            Y_u_c_real = complex_to_real_channel(Y_u_c)
            
            # Solve Uncorrelated SVM
            H_hat_u_i = svm_estimator_uncorrelated(Y_u_i_real, X_real)
            H_hat_u_c = svm_estimator_uncorrelated(Y_u_c_real, X_real)
            
            # Calculate NMSE (Uncorrelated)
            H_real_true = complex_to_real_channel(H_uncorr)
            mse_u_i += np.linalg.norm(H_hat_u_i - H_real_true)**2 / np.linalg.norm(H_real_true)**2
            mse_u_c += np.linalg.norm(H_hat_u_c - H_real_true)**2 / np.linalg.norm(H_real_true)**2
            
            # Correlated Channel Data Generation
            R_sqrt = sqrtm(channel_corr_matrix)
            H_w = (np.random.randn(N, K) + 1j * np.random.randn(N, K)) / np.sqrt(2)
            H_corr = R_sqrt @ H_w
            
            # Correlated Channel + IID Noise
            Z_iid = generate_noise((N, Tt), N0, correlation_matrix=None)
            Y_c_i = np.sign(np.real(H_corr @ X_pilot_complex + Z_iid)) + \
                    1j * np.sign(np.imag(H_corr @ X_pilot_complex + Z_iid))

            # Correlated Channel + Correlated Noise
            Z_corr = generate_noise((N, Tt), N0, correlation_matrix=noise_corr_matrix)
            Y_c_c = np.sign(np.real(H_corr @ X_pilot_complex + Z_corr)) + \
                    1j * np.sign(np.imag(H_corr @ X_pilot_complex + Z_corr))
            
            # Pre-process for SVM
            Y_c_i_real = complex_to_real_channel(Y_c_i)
            Y_c_c_real = complex_to_real_channel(Y_c_c)
            
            # Solve Correlated SVM (Joint)
            H_hat_c_i = svm_estimator_correlated(Y_c_i_real, X_real, Ck_matrices)
            H_hat_c_c = svm_estimator_correlated(Y_c_c_real, X_real, Ck_matrices)
            
            # Calculate NMSE (Correlated)
            H_corr_real_true = complex_to_real_channel(H_corr)
            mse_c_i += np.linalg.norm(H_hat_c_i - H_corr_real_true)**2 / np.linalg.norm(H_corr_real_true)**2
            mse_c_c += np.linalg.norm(H_hat_c_c - H_corr_real_true)**2 / np.linalg.norm(H_corr_real_true)**2

        # Average and store
        nmse_uncorr_iid.append(10 * np.log10(mse_u_i / num_trials))
        nmse_uncorr_corr.append(10 * np.log10(mse_u_c / num_trials))
        nmse_corr_iid.append(10 * np.log10(mse_c_i / num_trials))
        nmse_corr_corr.append(10 * np.log10(mse_c_c / num_trials))
        
        print(f"SNR {snr_db}dB Done.")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(SNRs_dB, nmse_uncorr_iid, 'b-o', label='Uncorrelated Channel (IID Noise)')
    plt.plot(SNRs_dB, nmse_uncorr_corr, 'b--x', label='Uncorrelated Channel (Corr Noise)')
    plt.plot(SNRs_dB, nmse_corr_iid, 'r-o', label='Correlated Channel (IID Noise)')
    plt.plot(SNRs_dB, nmse_corr_corr, 'r--x', label='Correlated Channel (Corr Noise)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('SVM-based Channel Estimation: Effect of Correlated Noise')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("nmse_correlated_noise.png")

if __name__ == "__main__":
    run_simulation()
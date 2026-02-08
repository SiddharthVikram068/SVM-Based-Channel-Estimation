# correlated_vs_uncorrelated_svm.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import cvxpy as cp
import time

# ---------------------
# Parameters (paper-like)
# ---------------------
K = 4          # users
N = 32         # BS antennas
Tt = 20        # pilot length
SNR_dB = np.arange(-5, 31, 5)   # smaller negative range for speed; change as needed
SNR = 10 ** (SNR_dB / 10)
num_trials = 60   # Monte Carlo trials (increase for smoother curves; paper used more)
C_svm = 1.0       # SVM slack parameter
C_qp = 1.0        # weight of slack in QP formulation (paper uses C)

# ---------------------
# Utilities
# ---------------------
def one_bit_quantize(X):
    return np.sign(X)

def generate_pilots(Tt, K):
    # DFT-based pilots: produce Xt (2K x 2Tt) as in paper
    F = np.fft.fft(np.eye(Tt))
    Xc = F[:, :K]              # (Tt x K)
    # Real-valued stacking (Eq. 11 style)
    Xt = np.block([
        [np.real(Xc).T,  np.imag(Xc).T],
        [-np.imag(Xc).T, np.real(Xc).T]
    ])   # (2K) x (2Tt)
    return Xt

def generate_iid_channel(N, K):
    Hc = (np.random.randn(N, K) + 1j*np.random.randn(N, K)) / np.sqrt(2)
    Hreal = np.block([
        [np.real(Hc), -np.imag(Hc)],
        [np.imag(Hc),  np.real(Hc)]
    ])  # (2N x 2K)
    return Hc, Hreal

def generate_spatial_covariance(N, angle_spread_deg=10):
    # Simple Laplacian-like angular correlation approximation
    # Using uniform sample of angles for a ULA-like model
    angles = np.linspace(-np.pi/2, np.pi/2, N)
    spread = np.deg2rad(angle_spread_deg)
    # pairwise Laplacian kernel
    C = np.exp(-np.abs(angles[:, None] - angles[None, :]) / spread)
    # normalize (so average power is unity-ish)
    C = C / np.max(np.real(np.linalg.eigvals(C)))
    return C

def complex_cov_to_real(Cbar):
    # Convert complex covariance Cbar (N x N) to real 2N x 2N form:
    # [[Re(Cbar), -Im(Cbar)]; [Im(Cbar), Re(Cbar)]]
    return np.block([[np.real(Cbar), -np.imag(Cbar)], [np.imag(Cbar), np.real(Cbar)]])

def generate_correlated_channel(N, K, Cbar):
    # generate Hc ~ CN(0, Cbar) per column
    # We'll draw real and imag jointly using multivariate normal with cov Cbar
    Hc = np.zeros((N, K), dtype=complex)
    # Use eigen-decomposition to sample correlated complex Gaussian
    # Cbar is Hermitian PSD
    vals, vecs = np.linalg.eigh(Cbar)
    vals = np.clip(vals, 0, None)
    sqrtC = vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T
    for k in range(K):
        z = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        Hc[:, k] = sqrtC @ z
    Hreal = np.block([[np.real(Hc), -np.imag(Hc)], [np.imag(Hc), np.real(Hc)]])
    return Hc, Hreal

def nmse(H_true, H_est):
    return np.linalg.norm(H_true - H_est, 'fro')**2 / (K * N)

# ---------------------
# Uncorrelated estimator (fast): per-row SVM
# ---------------------
def estimate_uncorrelated_svm(Yt, Xt, K, N, C=C_svm):
    # Yt: (2N x 2Tt), Xt: (2K x 2Tt)
    H_est_real = np.zeros((2*N, 2*K))
    for i in range(2*N):
        labels = Yt[i, :]
        # handle single-class label case
        if np.all(labels == 1) or np.all(labels == -1):
            # randomized normalized vector
            h_tilde = np.random.randn(2*K)
            H_est_real[i, :] = (np.sqrt(K) * h_tilde / np.linalg.norm(h_tilde))
            continue
        clf = LinearSVC(C=C, loss='squared_hinge', max_iter=20000)
        clf.fit(Xt.T, labels)
        h_tilde = clf.coef_.flatten()
        H_est_real[i, :] = np.sqrt(K) * h_tilde / np.linalg.norm(h_tilde)
    Hc_est = H_est_real[:N, :K] + 1j * H_est_real[N:, :K]
    return Hc_est

# ---------------------
# Correct correlated estimator: joint convex QP via cvxpy
# Solve Eq. (20) in real domain:
# minimize 0.5 * sum_k hk^T Ck_inv hk + C * sum xi
# s.t. y_{i,n} * (h_row_i^T x_n) >= 1 - xi_{i,n}, xi >= 0
# where H is (2N x 2K) real variable; hk = H[:, k] for k=0..K-1
# ---------------------
def estimate_correlated_qp(Yt, Xt, K, N, Cbar, C_param=1.0, solver_prefer='OSQP'):
    # Yt: (2N x 2Tt), Xt: (2K x 2Tt)
    twoN, twoK = 2*N, 2*K
    # Build real covariance for each user (assume same Cbar for each user)
    Ck_real = complex_cov_to_real(Cbar)   # 2N x 2N
    # regularize and invert
    reg = 1e-8
    Ck_inv = np.linalg.inv(Ck_real + reg * np.eye(2*N))

    # cvxpy variables
    H = cp.Variable((twoN, twoK))   # real representation
    xi = cp.Variable((twoN, 2*Tt), nonneg=True)

    # objective: 0.5 * sum_k h_k^T Ck_inv h_k + C * sum(xi)
    quad_terms = []
    # NOTE: hk corresponds to H[:, k] for k in 0..K-1 (first K columns)
    for k in range(K):
        hk = H[:, k]
        # use quadratic form with symmetric matrix Ck_inv
        quad_terms.append(cp.quad_form(hk, Ck_inv))
    obj = 0.5 * cp.sum(quad_terms) + C_param * cp.sum(xi)

    # constraints: for each i (0..2N-1) and n (0..2Tt-1)
    constraints = []
    Xt_cols = [Xt[:, n] for n in range(2*Tt)]
    for i in range(twoN):
        row_i = H[i, :]   # length 2K
        for n in range(2*Tt):
            yi = float(Yt[i, n])
            xcol = Xt_cols[n]
            constraints.append(yi * (row_i @ xcol) >= 1 - xi[i, n])

    # Solve QP
    prob = cp.Problem(cp.Minimize(obj), constraints)

    # Choose solver
    solvers_to_try = []
    if solver_prefer == 'OSQP':
        solvers_to_try = [cp.OSQP, cp.ECOS, cp.SCS]
    elif solver_prefer == 'ECOS':
        solvers_to_try = [cp.ECOS, cp.OSQP, cp.SCS]
    else:
        solvers_to_try = [cp.SCS, cp.ECOS, cp.OSQP]

    solved = False
    last_solution = None
    for solver in solvers_to_try:
        try:
            prob.solve(solver=solver, verbose=False, warm_start=True, eps_abs=1e-4)
            if H.value is not None:
                solved = True
                last_solution = solver.__name__
                break
        except Exception as e:
            # try next solver
            # print("Solver", solver, "failed:", e)
            continue

    if not solved:
        raise RuntimeError("QP solver failed (tried OSQP/ECOS/SCS). Install OSQP/ECOS and retry.")

    H_opt = H.value  # 2N x 2K
    # normalize scaling step from paper: scale each column vector (hk) so that ||H||_F matches trace{Cbar}
    # paper used H_hat = (trace{Cbar} / ||H_tilde||_F) * H_tilde
    H_tilde = H_opt
    scale_factor = np.trace(complex_cov_to_real(Cbar)) / (np.linalg.norm(H_tilde, 'fro') + 1e-12)
    H_hat = scale_factor * H_tilde

    # convert to complex
    Hc_hat = H_hat[:N, :K] + 1j * H_hat[N:, :K]
    return Hc_hat

# ---------------------
# Main Monte Carlo (compare methods)
# ---------------------
def main():
    Xt = generate_pilots(Tt, K)
    # prepare spatial covariance (complex)
    Cbar = generate_spatial_covariance(N, angle_spread_deg=10) + 0j  # make complex-compatible
    # ensure hermitian
    Cbar = (Cbar + Cbar.conj().T) / 2

    nmse_uncorr = np.zeros(len(SNR))
    nmse_corr = np.zeros(len(SNR))
    t0 = time.time()
    for idx, snr in enumerate(SNR):
        errs_u = []
        errs_c = []

        for trial in range(num_trials):
            # generate correlated channel (true) but also uncorrelated case uses i.i.d. 
            Hc_true, Hreal_corr = generate_correlated_channel(N, K, Cbar)

            # noise in real domain
            noise_std = np.sqrt(1/(2*snr))
            Z = noise_std * np.random.randn(2*N, 2*Tt)

            Yt = one_bit_quantize(Hreal_corr @ Xt + Z)  # quantized pilot measurements (2N x 2Tt)

            # Uncorrelated SVM estimator (per-row SVM) -- note: this estimator ignores correlation
            H_est_u = estimate_uncorrelated_svm(Yt, Xt, K, N, C=C_svm)
            errs_u.append(nmse(Hc_true, H_est_u))

            # Correlated estimator (QP) that jointly estimates H using known Cbar
            try:
                H_est_c = estimate_correlated_qp(Yt, Xt, K, N, Cbar, C_param=C_qp, solver_prefer='OSQP')
            except RuntimeError as re:
                print("QP solver failed on trial", trial, ":", re)
                # fallback to per-row SVM for this trial (should rarely happen)
                H_est_c = H_est_u.copy()
            errs_c.append(nmse(Hc_true, H_est_c))

        nmse_uncorr[idx] = np.mean(errs_u)
        nmse_corr[idx] = np.mean(errs_c)
        print(f"SNR {SNR_dB[idx]} dB: Uncorr NMSE {10*np.log10(nmse_uncorr[idx]):.2f} dB, Corr NMSE {10*np.log10(nmse_corr[idx]):.2f} dB")

    print("Total time (s):", time.time()-t0)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(SNR_dB, 10*np.log10(nmse_uncorr), marker='o', label='Uncorrelated SVM (per-row)')
    plt.plot(SNR_dB, 10*np.log10(nmse_corr), marker='s', label='Correlated SVM (joint QP)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('SVM-based Channel Estimation: Uncorr vs Correlated (QP)')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("nmse_comparison.png")

if __name__ == "__main__":
    main()

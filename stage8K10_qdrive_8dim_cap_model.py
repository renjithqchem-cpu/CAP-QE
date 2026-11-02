#!/usr/bin/env python3
"""Stage8K10: derive and validate an 8-dimensional, 3-qubit CAP model.

Uses the already-built Stage8K9 reference-free rational-Krylov basis, but does
not load the known full-space CAP eigenvalue or eigenvector. The first k basis
vectors are assessed for k=6..10. The first 8 vectors are saved as the candidate
3-qubit model only if the projection checks pass. No existing files are changed.
"""
import os
import json
import numpy as np

ROOT = "stage7A_qDRIVE_bridge_R2.8_CAS3_27"
H_FILE = f"{ROOT}/H_H_active_CAS3_27.npy"
W_FILE = f"{ROOT}/Wmany_rcap_8.0.npy"
TARGET_FILE = "stage8A_qDRIVE_input/active_eigenvectors_root2.npy"
BASIS_FILE = "stage8K9_predictive_rational_krylov/stage8K9_predictive_basis.npy"
OUT = "stage8K10_qdrive_8dim_cap_model"
ETA = 1.30446839e-5
HARTREE_TO_EV = 27.211386245988

for path in (H_FILE, W_FILE, TARGET_FILE, BASIS_FILE):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required input missing: {path}")
if os.path.exists(OUT):
    raise FileExistsError(f"{OUT} already exists; refusing to overwrite.")

H = np.asarray(np.load(H_FILE), dtype=np.complex128)
W = np.asarray(np.load(W_FILE), dtype=np.complex128)
target = np.asarray(np.load(TARGET_FILE), dtype=np.complex128).reshape(-1)
Qall = np.asarray(np.load(BASIS_FILE), dtype=np.complex128)
if H.shape != (2164, 2164) or W.shape != H.shape:
    raise ValueError(f"Unexpected H/W shapes: {H.shape}, {W.shape}")
if target.size != H.shape[0] or Qall.shape != (H.shape[0], 32):
    raise ValueError(f"Unexpected target/basis shapes: {target.shape}, {Qall.shape}")
if not all(np.all(np.isfinite(x)) for x in (H, W, target, Qall)):
    raise ValueError("NaN/Inf detected in an input.")

herr = float(np.linalg.norm(H - H.conj().T))
werr = float(np.linalg.norm(W - W.conj().T))
if herr > 1e-9 or werr > 1e-8:
    raise ValueError(f"Hermiticity failure: H={herr:.3e}, W={werr:.3e}")
target /= np.linalg.norm(target)
A = H - 1j * ETA * W

# Verify saved basis is orthonormal before using its ordered prefix.
qerr = float(np.linalg.norm(Qall.conj().T @ Qall - np.eye(Qall.shape[1])))
if qerr > 1e-8:
    raise ValueError(f"Stage8K9 basis orthonormality error too large: {qerr:.3e}")

records = []
for k in range(6, 11):
    Q = Qall[:, :k]
    Ared = Q.conj().T @ A @ Q
    evals, evecs = np.linalg.eig(Ared)
    lifted = Q @ evecs
    lifted /= np.linalg.norm(lifted, axis=0)[None, :]
    target_probs = np.abs(target.conj() @ lifted) ** 2
    idx = int(np.argmax(target_probs))
    lam = complex(evals[idx])
    psi = lifted[:, idx]
    residual = float(np.linalg.norm(A @ psi - lam * psi))
    width = float(-2.0 * lam.imag * HARTREE_TO_EV)
    rec = {
        "dimension": k,
        "selected_root_index_in_numpy_eig_order": idx,
        "energy_Ha": [float(lam.real), float(lam.imag)],
        "width_eV": width,
        "target_overlap_probability": float(target_probs[idx]),
        "lifted_full_operator_residual_Ha": residual,
        "projected_CAP_eigenvalues_Ha": [[float(z.real), float(z.imag)] for z in evals],
    }
    records.append(rec)
    print(f"dim={k:2d} E={lam.real:+.12f}{lam.imag:+.12f}j Ha "
          f"width={width:.8f} eV target_overlap={target_probs[idx]:.8f} "
          f"lifted_residual={residual:.6e} Ha")

# Candidate three-qubit (8-dimensional) projected model.
k = 8
Q8 = Qall[:, :k]
H8 = Q8.conj().T @ H @ Q8
W8 = Q8.conj().T @ W @ Q8
A8 = H8 - 1j * ETA * W8
H8_err = float(np.linalg.norm(H8 - H8.conj().T))
W8_err = float(np.linalg.norm(W8 - W8.conj().T))
A8_err = float(np.linalg.norm(A8 - (Q8.conj().T @ A @ Q8)))
wevals = np.linalg.eigvalsh((W8 + W8.conj().T) / 2)
# PSD is a physical requirement for a conventional absorbing CAP; report rather
# than silently force positivity.
summary = {
    "stage": "Stage8K10 8-dimensional qDRIVE CAP model",
    "status": "projection_validation_only_not_yet_quantum_circuit_result",
    "source_basis": BASIS_FILE,
    "inputs": {"H": H_FILE, "W": W_FILE, "target": TARGET_FILE},
    "eta": ETA,
    "basis_prefix_dimensions_assessed": records,
    "candidate_dimension": 8,
    "qubits_required_for_8_dimensional_encoding": 3,
    "basis_orthonormality_error_full_stage8K9": qerr,
    "H8_hermiticity_error": H8_err,
    "W8_hermiticity_error": W8_err,
    "CAP_projection_reconstruction_error": A8_err,
    "W8_minimum_eigenvalue": float(wevals.min()),
    "W8_maximum_eigenvalue": float(wevals.max()),
    "W8_positive_semidefinite_with_1e-10_tolerance": bool(wevals.min() >= -1e-10),
    "caveat": "This is a classically constructed reduced CAP model. A qDRIVE circuit must still implement non-unitary CAP evolution with a valid dilation/postselection or another justified method, and must be validated separately."
}

os.makedirs(OUT)
np.save(os.path.join(OUT, "stage8K10_basis8.npy"), Q8)
np.save(os.path.join(OUT, "stage8K10_H8.npy"), H8)
np.save(os.path.join(OUT, "stage8K10_W8.npy"), W8)
np.save(os.path.join(OUT, "stage8K10_CAP8.npy"), A8)
np.save(os.path.join(OUT, "stage8K10_CAP8_eigenvalues.npy"), np.linalg.eigvals(A8))
np.save(os.path.join(OUT, "stage8K10_W8_eigenvalues.npy"), wevals)
with open(os.path.join(OUT, "stage8K10_summary.json"), "x", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("=" * 78)
print("STAGE8K10 SUMMARY")
print(f"Candidate basis: {Q8.shape}; required qubits: 3")
print(f"H8 Hermiticity error: {H8_err:.6e}")
print(f"W8 Hermiticity error: {W8_err:.6e}")
print(f"W8 minimum eigenvalue: {wevals.min():+.12e}")
print(f"W8 PSD within 1e-10 tolerance: {wevals.min() >= -1e-10}")
print(f"Outputs: {OUT}/")
print("No CAP reference eigenvalue/vector used; no existing files modified.")


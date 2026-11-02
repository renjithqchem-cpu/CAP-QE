# CAP-QE
A reduced space quantum eigensolver workflow for computing complex absorbing potential resonance eigenvalues, with classical validation and quantum hardware implementation.

**What this is.** CAP-QE obtains the position (Er) and width (Gamma) of the 2Sigma_u+ shape resonance of H2- at R = 2.8 a0 from the expectation values of two Hermitian operators, H and W, of a complex absorbing potential (CAP) model, A(eta) = H - i*eta*W. A 2164-determinant CAS(3,27) full CI model is projected onto an eight-dimensional subspace, encoded on three qubits, and measured on IBM `ibm_kingston` (108 circuits, 4096 shots per circuit, no error mitigation).

**Environment.** Python 3.10 with the packages in `requirements.txt` (Qiskit 1.3.3, Aer 0.16.4, IBM Runtime 0.40.1, PySCF 2.13.1, NumPy 2.2.6, SciPy 1.15.2). Install with `pip install -r requirements.txt`. Running the hardware scripts requires your own IBM Quantum account; no credentials are included.

## Reduced model (SI S9)

| File | Contents / purpose |
|---|---|
| `stage8K9_predictive_basis.npy` | Orthonormal rational Krylov basis; its first eight vectors define Q8. |
| `stage8K10_H8.npy` | Reduced Hamiltonian H8 = Q8^T H Q8 (8 x 8). |
| `stage8K10_W8.npy` | Reduced absorbing operator W8 = Q8^T W Q8 (8 x 8). |
| `stage8K10_qdrive_8dim_cap_model.py` | Builds H8 and W8 from the Krylov basis and checks the reduced eigenvalue. |

## Hardware execution and analysis scripts

| File | Contents / purpose |
|---|---|
| `h2cap_ibm_hardware.py` | Loads H8 and W8, prepares the fitted three-qubit EfficientSU2 state, builds the 27 measurement settings per observable, and runs them with SamplerV2. |
| `h2cap_ibm_hardware_progressive_4096plus.py` | Hardware run script for the progressive shot runs up to 4096 shots per circuit. |
| `stage8L1_readout_triple_comparison.py` | Measures the 8 x 8 readout confusion matrices of candidate qubit triples and compares them for qubit selection. |
| `h2cap_offline_readout_mitigation_analysis.py` | Applies the measured confusion matrices to the archived counts (offline mitigation) and recomputes Er and Gamma. |
| `independent_audit_h2cap_ibm_results.py` | Raw count audit: recounts all shots and rebuilds the saved estimates from the archived counts. |
| `h2cap_noise_final.py` | Aer noise simulation: simplified three-qubit noise model, 54 circuits at 2048 shots, six noise conditions. |

## Production hardware archive (SI S1, S14)

| File | Contents / purpose |
|---|---|
| `h2cap_ibm_hardware_20261013_070039_circuits.qpy` | The 108 transpiled production circuits (Qiskit QPY format, Qiskit 1.3.3). |
| `h2cap_ibm_hardware_20261013_070039_groups.json` | Mapping of each circuit to its Pauli group (27 for H8, 27 for W8, per qubit triple). |
| `h2cap_ibm_hardware_20261013_070039_results.json` | Saved production results: raw counts and aggregate estimates for (43,44,45) and (20,21,36). |
| `h2cap_ibm_independent_audit_20261013.json` | Output of the raw count audit (108 circuits, 442 368 shots, zero difference from saved values). |
| `h2cap_ideal_hardware_reconciliation_20261013.json` / `.csv` | Comparison of the ideal statevector values with the hardware estimates. |
| `h2cap_qpy_basis_audit.json` | Relative unitary audit of the measurement basis rotations (106 comparisons, no errors). |
| `h2cap_qpy_model_audit.json` | Pauli/source matrix audit: H8 and W8 rebuilt from the mapped Pauli labels (108 records, tolerance 1e-9). |
| `circuit_archive_summary.json` | Summary of the archived circuit set. |
| `all_108_actual_transpiled_circuits.pdf` | Diagrams of all 108 transpiled circuits. |
| `actual_H_triple_43-44-45_group00.png` | Example transpiled circuit for (43,44,45), group 0. |
| `actual_H_triple_20-21-36_group00.png` | Example transpiled circuit for (20,21,36), group 0. |

## Key reference values

Reduced-model target: Er = 0.21239113 eV, Gamma = 0.07516884 eV (eta = 1.30446839e-5 Eh). Full-space benchmark (eta = 1.303e-5 Eh): Er = 0.21238637 eV, Gamma = 0.07515184 eV. Estimators: Er = (<H> - E_thr) x 27.211386245988 eV/Eh, Gamma = 2 x eta x <W> x 27.211386245988 eV/Eh.

## Citation

If you use these files, please cite the manuscript (reference to be added on publication) and this repository (DOI to be added).

## License

Released under the MIT License (see `LICENSE`).

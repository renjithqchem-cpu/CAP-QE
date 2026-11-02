#!/usr/bin/env python3
"""Submit and analyse an H2- CAP 8D Pauli-measurement pilot on IBM hardware.
Uses the validated 8x8 H/W matrices and the same statevector optimization,
Pauli grouping, bit ordering, and estimators as h2cap_noise_decomposition.py.
"""
from pathlib import Path
import json, time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

CAP_DIR = Path("/home/renjith/Documents/quantum_computing/H2_minus_qDRIVE/01_classical_CAP/stage1_R2.8/stage8K10_qdrive_8dim_cap_model")
ACCOUNT_FILE = "/home/renjith/.qdrive_ibm_platform_account.json"
ACCOUNT_NAME = "qdrive_ibm_platform"
BACKEND_NAME = "ibm_kingston"
SHOTS = 512  # 54 circuits x 512 = 27,648 hardware shots for a time-limited pilot
ETA = 1.30446839e-5
THRESHOLD_HA = -1.423580144198899
HARTREE_TO_EV = 27.211386245988
REF_E_EV = 0.21239112537024876
REF_GAMMA_EV = 0.07516884295821497
SEED = 121012


def load_inputs():
    hp, wp = CAP_DIR / "stage8K10_H8.npy", CAP_DIR / "stage8K10_W8.npy"
    if not hp.is_file() or not wp.is_file():
        raise FileNotFoundError(f"Missing H/W matrices: {hp} or {wp}")
    h, w = np.load(hp), np.load(wp)
    if h.shape != (8, 8) or w.shape != (8, 8):
        raise ValueError(f"Expected (8,8) matrices; got H={h.shape}, W={w.shape}")
    if not np.allclose(h, h.conj().T, atol=1e-10) or not np.allclose(w, w.conj().T, atol=1e-10):
        raise ValueError("H/W Hermiticity validation failed")
    return h, w


def optimize_state(h, w):
    target = complex(-1.4157749143054177, -0.0013812020137213283)
    cap = h.astype(complex) - 1j * ETA * w
    residual = cap - target * np.eye(8)
    bmat = residual.conj().T @ residual
    bmat = (bmat + bmat.conj().T) / 2
    ansatz = EfficientSU2(3, su2_gates=["ry", "rz"], reps=2, entanglement="linear")
    params = list(ansatz.parameters)
    def vec(x):
        return Statevector.from_instruction(ansatz.assign_parameters(dict(zip(params, x)))).data
    def obj(x):
        v = vec(x)
        return float(np.real(np.vdot(v, bmat @ v)))
    def grad(x):
        g = np.zeros_like(x)
        for j in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[j] += np.pi/2; xm[j] -= np.pi/2
            g[j] = 0.5 * (obj(xp) - obj(xm))
        return g
    opt = minimize(obj, np.zeros(len(params)), jac=grad, method="BFGS", options={"maxiter": 2000, "gtol": 1e-10})
    if not opt.success or opt.fun > 1e-10:
        raise RuntimeError(f"State optimization validation failed: success={opt.success}, objective={opt.fun:.6e}, {opt.message}")
    return ansatz.assign_parameters(dict(zip(params, opt.x))), float(opt.fun)


def build_circuits(prepared, h, w, backend):
    circuits, records = [], []
    for name, matrix in (("H", h), ("W", w)):
        op = SparsePauliOp.from_operator(matrix, atol=1e-12, rtol=0)
        for group in op.group_commuting(qubit_wise=True):
            labels = [p.to_label() for p in group.paulis]
            coeffs = np.asarray(group.coeffs, dtype=complex)
            if np.max(np.abs(coeffs.imag)) > 1e-10:
                raise ValueError(f"Complex Pauli coefficients found for {name}")
            coeffs = coeffs.real
            qc = QuantumCircuit(3)
            qc.compose(prepared, inplace=True)
            for q in range(3):
                bases = {label[2-q] for label in labels} - {"I"}
                if len(bases) > 1:
                    raise ValueError("Group is not qubit-wise commuting")
                if bases == {"X"}:
                    qc.h(q)
                elif bases == {"Y"}:
                    qc.sdg(q); qc.h(q)
            qc.measure_all()
            circuits.append(qc)
            records.append((name, labels, coeffs))
    if len(circuits) != 54:
        raise ValueError(f"Expected 54 grouped circuits, got {len(circuits)}")
    compiled = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=SEED)
    if len(compiled) != 54:
        raise RuntimeError("Transpilation changed circuit count")
    if any(c.num_clbits != 3 for c in compiled):
        raise RuntimeError("Unexpected classical-bit count after transpilation")
    return compiled, records


def estimate_counts(counts_by_circuit, records):
    means, varmeans = {"H": 0.0, "W": 0.0}, {"H": 0.0, "W": 0.0}
    for counts, (name, labels, coeffs) in zip(counts_by_circuit, records):
        n = sum(counts.values())
        if n != SHOTS:
            raise RuntimeError(f"Expected {SHOTS} shots, received {n} for {name}")
        s1 = s2 = 0.0
        for bitstring, freq in counts.items():
            bits = bitstring.replace(" ", "")
            value = 0.0
            for label, coeff in zip(labels, coeffs):
                parity = 1
                for q in range(3):
                    if label[2-q] != "I" and bits[-1-q] == "1":
                        parity *= -1
                value += coeff * parity
            s1 += freq * value; s2 += freq * value * value
        mean = s1/n
        svar = max(0.0, (s2/n - mean*mean) * n/(n-1)) if n > 1 else 0.0
        means[name] += mean
        varmeans[name] += svar/n
    return means, varmeans


def main():
    h, w = load_inputs()
    print("CAP matrix checks: PASS", flush=True)
    prepared, optfun = optimize_state(h, w)
    print(f"Statevector optimization residual: {optfun:.6e}", flush=True)
    service = QiskitRuntimeService(filename=ACCOUNT_FILE, name=ACCOUNT_NAME)
    backend = service.backend(BACKEND_NAME)
    st = backend.status()
    if not st.operational:
        raise RuntimeError(f"Backend {BACKEND_NAME} is not operational")
    print(f"Backend: {BACKEND_NAME}; pending jobs reported: {st.pending_jobs}", flush=True)
    circuits, records = build_circuits(prepared, h, w, backend)
    depths = [c.depth() for c in circuits]
    print(f"Circuit validation: PASS; count={len(circuits)}; max_depth={max(depths)}; shots/circuit={SHOTS}; total_shots={len(circuits)*SHOTS}", flush=True)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=SHOTS)
    job_id = job.job_id()
    record = {"job_id": job_id, "backend": BACKEND_NAME, "shots_per_circuit": SHOTS, "circuits": len(circuits), "total_shots": len(circuits)*SHOTS, "optimizer_residual": optfun, "max_transpiled_depth": max(depths), "status": str(job.status())}
    out = Path("h2cap_ibm_hardware_pilot.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"SUBMITTED_JOB_ID: {job_id}", flush=True)
    print(f"Job status: {job.status()}", flush=True)
    print(f"Submission record: {out.resolve()}", flush=True)
    result = job.result()
    counts_by_circuit = [pub.data.meas.get_counts() for pub in result]
    means, varmeans = estimate_counts(counts_by_circuit, records)
    energy = (means["H"] - THRESHOLD_HA) * HARTREE_TO_EV
    energy_se = np.sqrt(varmeans["H"]) * HARTREE_TO_EV
    width = 2*ETA*means["W"]*HARTREE_TO_EV
    width_se = 2*ETA*np.sqrt(varmeans["W"]) * HARTREE_TO_EV
    summary = {**record, "status": str(job.status()), "H_Ha": means["H"], "H_SE_Ha": float(np.sqrt(varmeans["H"])), "W": means["W"], "W_SE": float(np.sqrt(varmeans["W"])), "energy_eV": energy, "energy_SE_eV": energy_se, "width_eV": width, "width_SE_eV": width_se, "reference_energy_eV": REF_E_EV, "reference_width_eV": REF_GAMMA_EV}
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULTS: hardware counts retrieved for all 54 circuits", flush=True)
    print(f"<H> = {means['H']:.10f} +/- {np.sqrt(varmeans['H']):.6g} Ha", flush=True)
    print(f"<W> = {means['W']:.6f} +/- {np.sqrt(varmeans['W']):.6g}", flush=True)
    print(f"Energy = {energy:.8f} +/- {energy_se:.8f} eV (reference {REF_E_EV:.8f} eV)", flush=True)
    print(f"Width = {width:.8f} +/- {width_se:.8f} eV (reference {REF_GAMMA_EV:.8f} eV)", flush=True)
    print(f"Saved: {out.resolve()}", flush=True)
    print("NOTE: hardware estimates are unmitigated; results include device noise and finite-shot uncertainty.", flush=True)

if __name__ == "__main__":
    main()


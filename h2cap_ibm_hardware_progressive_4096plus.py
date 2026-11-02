#!/usr/bin/env python3
"""Submit and analyse an H2- CAP 8D Pauli-measurement pilot on IBM hardware.
Uses the validated 8x8 H/W matrices and the same statevector optimization,
Pauli grouping, bit ordering, and estimators as h2cap_noise_decomposition.py.
"""
from pathlib import Path
import argparse
import json, time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

CAP_DIR = Path("/home/renjith/Documents/quantum_computing/H2_minus_qDRIVE/01_classical_CAP/stage1_R2.8/stage8K10_qdrive_8dim_cap_model")
ACCOUNT_FILE = "/home/renjith/.qdrive_ibm_platform_account.json"
ACCOUNT_NAME = "qdrive_ibm_platform"
BACKEND_NAME = "ibm_kingston"
DEFAULT_SHOTS = 4096  # minimum production setting; CLI permits larger values
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


def parse_triples(text):
    triples = []
    for item in text.split(';'):
        if not item.strip():
            continue
        vals = tuple(int(x.strip()) for x in item.split(','))
        if len(vals) != 3 or len(set(vals)) != 3 or min(vals) < 0:
            raise ValueError(f"Invalid triple {item!r}; need three distinct non-negative physical indices")
        triples.append(vals)
    if not triples:
        raise ValueError("At least one --triples entry is required")
    if len({q for tr in triples for q in tr}) != sum(len(tr) for tr in triples):
        raise ValueError("Physical triples overlap; use disjoint triples for a clean comparison")
    return triples


def target_two_qubit_pairs(backend):
    pairs = set()
    target = backend.target
    for opname in target.operation_names:
        try:
            # In Qiskit Target, target[opname] is normally a mapping
            # from qarg tuples to InstructionProperties, not an Instruction.
            qarg_properties = target[opname]
            for pair in qarg_properties.keys():
                if pair is not None and len(pair) == 2:
                    pairs.add(frozenset((int(pair[0]), int(pair[1]))))
        except Exception:
            continue
    return pairs


def validate_triples(triples, backend):
    if any(max(tr) >= backend.num_qubits for tr in triples):
        raise ValueError(f"Triple contains a qubit index outside backend width {backend.num_qubits}")
    supported_pairs = target_two_qubit_pairs(backend)
    for tr in triples:
        required = (frozenset((tr[0], tr[1])), frozenset((tr[1], tr[2])))
        missing = [tuple(sorted(p)) for p in required if p not in supported_pairs]
        if missing:
            raise ValueError(
                f"Triple {tr} does not have direct supported two-qubit links {missing}; "
                "choose a physical path with the centre at logical q1. No job submitted."
            )


def verify_measurement_layout(circuit, triple):
    layout = getattr(circuit, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        raise RuntimeError(f"Cannot verify final physical layout for {circuit.name}; no job submitted")
    logical_to_physical = layout.final_index_layout(filter_ancillas=True)
    if len(logical_to_physical) != 3:
        raise RuntimeError(
            f"Unexpected logical-to-physical map for {circuit.name}: {logical_to_physical}; no job submitted"
        )
    logical_to_physical = tuple(int(x) for x in logical_to_physical)
    # Routing may permute logical states within the chosen physical triple.
    # Reject any routing that escapes the selected triple, but track the final map.
    if set(logical_to_physical) != set(triple):
        raise RuntimeError(
            f"Circuit escaped selected physical triple {triple}: final map {logical_to_physical}; no job submitted"
        )
    actual_by_cbit = {}
    for inst in circuit.data:
        if getattr(inst.operation, "name", "") == "measure":
            phys = circuit.find_bit(inst.qubits[0]).index
            cbit = circuit.find_bit(inst.clbits[0]).index
            if cbit in actual_by_cbit:
                raise RuntimeError(f"Duplicate measurement of classical bit {cbit}; no job submitted")
            actual_by_cbit[cbit] = phys
    expected_by_cbit = {i: logical_to_physical[i] for i in range(3)}
    if actual_by_cbit != expected_by_cbit:
        raise RuntimeError(
            f"Measurement mapping mismatch for {circuit.name}: expected {expected_by_cbit}, "
            f"got {actual_by_cbit}; no job submitted"
        )


def build_circuits(prepared, h, w, backend, triples, seed, optimization_level=1):
    from qiskit import transpile
    circuits, records = [], []
    grouped_ops = []
    for name, matrix in (("H", h), ("W", w)):
        op = SparsePauliOp.from_operator(matrix, atol=1e-12, rtol=0)
        groups = op.group_commuting(qubit_wise=True)
        for group in groups:
            labels = [p.to_label() for p in group.paulis]
            coeffs = np.asarray(group.coeffs, dtype=complex)
            if np.max(np.abs(coeffs.imag)) > 1e-10:
                raise ValueError(f"Complex Pauli coefficients found for {name}")
            coeffs = coeffs.real
            grouped_ops.append((name, labels, coeffs))
    if len(grouped_ops) != 54:
        raise ValueError(f"Expected 54 grouped measurement circuits per triple, got {len(grouped_ops)}")

    # Compile each logical measurement set independently with an explicit physical layout.
    for triple_index, triple in enumerate(triples):
        logical_batch = []
        for group_index, (name, labels, coeffs) in enumerate(grouped_ops):
            qc = QuantumCircuit(3, name=f"{name}_t{triple_index}_g{group_index:02d}")
            qc.compose(prepared, inplace=True)
            for q in range(3):
                bases = {label[2-q] for label in labels} - {"I"}
                if len(bases) > 1:
                    raise ValueError("Group is not qubit-wise commuting")
                if bases == {"X"}:
                    qc.h(q)
                elif bases == {"Y"}:
                    qc.sdg(q)
                    qc.h(q)
            qc.measure_all()
            logical_batch.append(qc)
            records.append({
                "triple_index": triple_index,
                "triple": list(triple),
                "observable": name,
                "pauli_labels": labels,
                "coefficients": [float(c) for c in coeffs],
                "group_index_within_triple": group_index,
            })
        compiled = transpile(
            logical_batch,
            backend=backend,
            initial_layout=list(triple),
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        if len(compiled) != 54:
            raise RuntimeError(f"Transpilation changed circuit count for triple {triple}")
        for c in compiled:
            if c.num_clbits != 3:
                raise RuntimeError(f"Unexpected classical-bit count in {c.name}")
            verify_measurement_layout(c, triple)
        circuits.extend(compiled)
    if len(circuits) != 54 * len(triples) or len(records) != len(circuits):
        raise RuntimeError("Circuit/record count mismatch; no job submitted")
    return circuits, records


def estimate_counts(counts_by_circuit, records, shots):
    if len(counts_by_circuit) != len(records):
        raise ValueError(f"Circuit/count-record mismatch: {len(counts_by_circuit)} counts vs {len(records)} records")
    means = {"H": 0.0, "W": 0.0}
    varmeans = {"H": 0.0, "W": 0.0}
    for i, (counts, record) in enumerate(zip(counts_by_circuit, records)):
        name = record["observable"]
        labels = record["pauli_labels"]
        coeffs = record["coefficients"]
        if name not in means:
            raise ValueError(f"Unexpected observable {name!r} at group {i}")
        if len(labels) != len(coeffs) or not counts:
            raise ValueError(f"Invalid labels/coefficients/counts for group {i}")
        n = sum(counts.values())
        if n != shots:
            raise ValueError(f"Group {i} ({name}): expected {shots} shots, got {n}")
        values, frequencies = [], []
        for bitstring, freq in counts.items():
            bits = bitstring.replace(" ", "")
            if len(bits) != 3 or any(b not in "01" for b in bits):
                raise ValueError(f"Unexpected count key {bitstring!r} in group {i}")
            value = 0.0
            for label, coeff in zip(labels, coeffs):
                parity = 1
                for q in range(3):
                    if label[2-q] != "I" and bits[2-q] == "1":
                        parity *= -1
                value += float(coeff) * parity
            values.append(value)
            frequencies.append(freq)
        values = np.asarray(values, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)
        mean = np.dot(values, frequencies) / n
        sample_var = np.dot((values - mean) ** 2, frequencies) / (n - 1) if n > 1 else 0.0
        means[name] += mean
        varmeans[name] += sample_var / n
    return means, varmeans


def main():
    import argparse
    parser = argparse.ArgumentParser(description="H2- CAP hardware measurement with explicit physical layouts")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="shots per circuit (minimum 4096)")
    parser.add_argument("--triples", default="43,44,45;20,21,36",
                        help='disjoint physical triples; e.g. "43,44,45;20,21,36"')
    parser.add_argument("--backend", default=BACKEND_NAME)
    parser.add_argument("--seed-transpiler", type=int, default=SEED)
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate circuits and save QPY/mapping, but do not submit a Runtime job")
    args = parser.parse_args()
    if args.shots < 4096:
        parser.error("--shots must be >= 4096 for this production workflow")
    triples = parse_triples(args.triples)

    h, w = load_inputs()
    print("CAP matrix checks: PASS", flush=True)
    prepared, optfun = optimize_state(h, w)
    print(f"Statevector optimization residual: {optfun:.6e}", flush=True)
    service = QiskitRuntimeService(filename=ACCOUNT_FILE, name=ACCOUNT_NAME)
    backend = service.backend(args.backend)
    st = backend.status()
    if not st.operational:
        raise RuntimeError(f"Backend {args.backend} is not operational; no job submitted")
    validate_triples(triples, backend)
    print(f"Backend: {args.backend}; pending jobs reported: {st.pending_jobs}", flush=True)
    print("Physical triples (logical q0,q1,q2):", triples, flush=True)
    circuits, records = build_circuits(prepared, h, w, backend, triples, args.seed_transpiler, args.optimization_level)
    depths = [c.depth() for c in circuits]
    total_shots = len(circuits) * args.shots
    print(
        f"Circuit/layout validation: PASS; circuits={len(circuits)}; "
        f"circuits_per_triple=54; max_depth={max(depths)}; shots/circuit={args.shots}; "
        f"total_shots={total_shots}", flush=True
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    circuits_file = Path(f"h2cap_ibm_hardware_{stamp}_circuits.qpy")
    with circuits_file.open("wb") as f:
        qpy.dump(circuits, f)
    groups_file = Path(f"h2cap_ibm_hardware_{stamp}_groups.json")
    groups_file.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved exact transpiled circuits: {circuits_file.resolve()}", flush=True)
    print(f"Saved observable/layout mapping: {groups_file.resolve()}", flush=True)
    if args.dry_run:
        print("DRY RUN COMPLETE: no IBM Runtime job submitted and no QPU shots used.", flush=True)
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=args.shots)
    job_id = job.job_id()
    submission = {
        "job_id": job_id, "circuits_archive": str(circuits_file), "group_mapping": str(groups_file),
        "backend": args.backend, "shots_per_circuit": args.shots, "circuits": len(circuits),
        "circuits_per_triple": 54, "total_shots": total_shots, "triples": [list(t) for t in triples],
        "optimizer_residual": optfun, "max_transpiled_depth": max(depths),
        "seed_transpiler": args.seed_transpiler, "optimization_level": args.optimization_level,
        "status": str(job.status()),
    }
    out = Path(f"h2cap_ibm_hardware_{stamp}_results.json")
    out.write_text(json.dumps(submission, indent=2) + "\n")
    print(f"SUBMITTED_JOB_ID: {job_id}", flush=True)
    print(f"Job status: {job.status()}", flush=True)
    print(f"Submission record: {out.resolve()}", flush=True)

    result = job.result()
    counts_by_circuit = [pub.data.meas.get_counts() for pub in result]
    all_summaries = []
    for triple_index, triple in enumerate(triples):
        indices = [i for i, rec in enumerate(records) if rec["triple_index"] == triple_index]
        triple_counts = [counts_by_circuit[i] for i in indices]
        triple_records = [records[i] for i in indices]
        means, varmeans = estimate_counts(triple_counts, triple_records, args.shots)
        energy = (means["H"] - THRESHOLD_HA) * HARTREE_TO_EV
        energy_se = np.sqrt(varmeans["H"]) * HARTREE_TO_EV
        width = 2 * ETA * means["W"] * HARTREE_TO_EV
        width_se = 2 * ETA * np.sqrt(varmeans["W"]) * HARTREE_TO_EV
        summary = {
            "triple": list(triple), "H_Ha": means["H"], "H_SE_Ha": float(np.sqrt(varmeans["H"])),
            "W": means["W"], "W_SE": float(np.sqrt(varmeans["W"])),
            "energy_eV": float(energy), "energy_SE_eV": float(energy_se),
            "width_eV": float(width), "width_SE_eV": float(width_se),
            "reference_energy_eV": REF_E_EV, "reference_width_eV": REF_GAMMA_EV,
            "energy_delta_from_reference_eV": float(energy - REF_E_EV),
            "width_delta_from_reference_eV": float(width - REF_GAMMA_EV),
        }
        all_summaries.append(summary)
        print(f"\n=== RESULTS for triple {triple} ===", flush=True)
        print(f"<H> = {means['H']:.10f} +/- {np.sqrt(varmeans['H']):.6g} Ha", flush=True)
        print(f"<W> = {means['W']:.6f} +/- {np.sqrt(varmeans['W']):.6g}", flush=True)
        print(f"E_r = {energy:.8f} +/- {energy_se:.8f} eV (reference {REF_E_EV:.8f} eV)", flush=True)
        print(f"Gamma = {width:.8f} +/- {width_se:.8f} eV (reference {REF_GAMMA_EV:.8f} eV)", flush=True)
    final = {**submission, "status": str(job.status()), "results_by_triple": all_summaries,
             "mitigation": "none; raw hardware estimates", "uncertainty_note": "Within-job shot-noise estimates; do not include between-job or device systematic uncertainty."}
    out.write_text(json.dumps(final, indent=2) + "\n")
    print(f"\nRESULTS: retrieved {len(counts_by_circuit)} circuit count sets; saved {out.resolve()}", flush=True)
    print("NOTE: results are unmitigated. Per-group shot-noise uncertainty does not include hardware/systematic error.", flush=True)

if __name__ == "__main__":
    main()


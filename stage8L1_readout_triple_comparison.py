"""
Stage 8L1: readout confusion-matrix comparison on two qubit triples (one small job).

Purpose
  Decide whether the large errors seen on qubits 0,1,2 come from the DEVICE (those qubits at that time)
  or from the SCRIPT/ANALYSIS (label order, count association, layout).
  Same code, two triples, one job:
    * compare both triples as evidence; similar errors do NOT by themselves prove a script bug
    * a clean alternative triple supports a physical-qubit-specific explanation, subject to layout checks

What it does
  * builds the 8 computational-basis circuits (X gates only + measure) for each triple, explicit initial_layout
  * classical bit i is measured from the physical qubit triple[i]; label string is q2q1q0 (Qiskit order)
  * submits ONE SamplerV2 job (or runs a local noisy simulation with --local-test)
  * saves everything needed to audit it: job id, circuit order, layout, raw counts, calibration snapshot
  * analysis: full 8x8 matrix, per-qubit P(1|0) and P(0|1), tensor-product check, mean assignment error
  * does NOT apply any mitigation to the H2- data. (A helper `mitigate_counts` is provided for later use.)

Usage
  python stage8L1_readout_triple_comparison.py --backend ibm_kingston --shots 4096
  python stage8L1_readout_triple_comparison.py --backend ibm_kingston --triples "0,1,2;40,41,42"
  python stage8L1_readout_triple_comparison.py --analyse-only stage8L1_readout_<job>.json
  python stage8L1_readout_triple_comparison.py --local-test      # offline mechanics test, no IBM access
Notes
  Auto-selected triple = best connected 3-qubit path (readout + two-qubit-gate error from the backend's
  current calibration), not overlapping the original (0,1,2), so it is also usable for the later H2- circuits.
"""
import argparse, json, itertools, sys, time
import numpy as np

LABELS = [format(k, "03b") for k in range(8)]        # 'q2q1q0'


def build_circuits(triple_list):
    from qiskit import QuantumCircuit
    circs, meta = [], []
    for t, triple in enumerate(triple_list):
        for lab in LABELS:
            qc = QuantumCircuit(3, 3, name=f"prep_{lab}_t{t}")
            for i in range(3):                           # label char index 2-i corresponds to qubit i
                if lab[2 - i] == "1":
                    qc.x(i)
            qc.measure([0, 1, 2], [0, 1, 2])             # logical qubit i -> classical bit i
            circs.append(qc)
            meta.append(dict(triple_index=t, triple=list(triple), label=lab))
    return circs, meta


def transpile_all(circs, meta, backend):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    out = []
    for qc, m in zip(circs, meta):
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1,
                                          initial_layout=m["triple"])
        isa = pm.run(qc)
        out.append(isa)
    return out


def best_triples(backend, exclude=((0, 1, 2),), n=1):
    """Choose connected 3-qubit paths using current readout + 2Q-gate errors.

    Returned triples are disjoint from every excluded triple, so the alternative
    does not reuse physical qubits from the original (0,1,2) triple.
    """
    tg = backend.target
    twoq = next((g for g in ("cz", "ecr", "cx") if g in tg.operation_names), None)
    if twoq is None:
        raise RuntimeError("Backend target exposes no supported cz/ecr/cx two-qubit operation.")

    ro = {}
    for q in range(tg.num_qubits):
        try:
            prop = tg["measure"][(q,)]
            ro[q] = getattr(prop, "error", None) if prop is not None else None
        except Exception:
            ro[q] = None

    ge = {}
    for pair, props in tg[twoq].items():
        if props is None or getattr(props, "error", None) is None:
            continue
        pair = tuple(pair)
        if len(pair) != 2:
            continue
        key = frozenset(pair)
        ge[key] = min(float(props.error), ge.get(key, 1.0))

    nbr = {}
    for pair in ge:
        a, b = tuple(pair)
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)

    excluded_sets = [set(int(q) for q in tr) for tr in exclude]
    cands = []
    for center, neighbors in nbr.items():
        for left, right in itertools.combinations(sorted(neighbors), 2):
            triple = (left, center, right)
            if len(set(triple)) != 3:
                continue
            if any(set(triple) & ex for ex in excluded_sets):
                continue
            if any(ro.get(q) is None for q in triple):
                continue
            score = (sum(float(ro[q]) for q in triple)
                     + ge[frozenset((left, center))]
                     + ge[frozenset((center, right))])
            cands.append((score, triple))

    cands.sort(key=lambda x: x[0])
    out, used = [], set().union(*excluded_sets) if excluded_sets else set()
    for score, triple in cands:
        if set(triple) & used:
            continue
        out.append(triple)
        used.update(triple)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"Could find only {len(out)} disjoint calibrated triple(s); requested {n}. "
            "Supply a verified alternative using --triples if appropriate."
        )
    return out


def calibration_snapshot(backend, qubits):
    snap = {}
    try:
        tg = backend.target
        for q in qubits:
            qp = tg.qubit_properties[q]
            snap[str(q)] = dict(T1=getattr(qp, "t1", None), T2=getattr(qp, "t2", None),
                                readout_error=getattr(tg["measure"][(q,)], "error", None))
    except Exception as e:
        snap["error"] = repr(e)
    return snap


def counts_to_matrix(counts_by_label):
    """rows = prepared label, columns = observed label (both 'q2q1q0'); entries = probabilities"""
    M = np.zeros((8, 8))
    for i, lab in enumerate(LABELS):
        c = counts_by_label[lab]
        tot = sum(c.values())
        for key, v in c.items():
            M[i, int(key.replace(" ", ""), 2)] += v / tot
    return M


def analyse(M, name=""):
    print(f"\n=== readout matrix {name} (rows prepared, cols observed, label q2q1q0) ===")
    print("        " + "  ".join(LABELS))
    for i, lab in enumerate(LABELS):
        print(f"prep {lab}  " + "  ".join(f"{M[i, j]:.3f}" for j in range(8)))
    mean_err = 1 - np.mean(np.diag(M))
    print(f"mean assignment error = {mean_err:.4f}")
    # per-qubit marginals
    single = []
    for q in range(3):
        p10 = []   # P(read 1 | prepared 0)
        p01 = []   # P(read 0 | prepared 1)
        for i, lab in enumerate(LABELS):
            prep_bit = int(lab[2 - q])
            row_marg1 = sum(M[i, j] for j, l2 in enumerate(LABELS) if l2[2 - q] == "1")
            (p10 if prep_bit == 0 else p01).append(row_marg1 if prep_bit == 0 else 1 - row_marg1)
        single.append((float(np.mean(p10)), float(np.mean(p01))))
        print(f"qubit {q} (c{q}): P(1|0) = {single[-1][0]:.4f}   P(0|1) = {single[-1][1]:.4f}"
              f"   [spread over other qubits' states: P(1|0) in {min(p10):.3f}-{max(p10):.3f}]")
    # tensor-product model
    S = []
    for q in range(3):
        a, b = single[q]
        S.append(np.array([[1 - a, a], [b, 1 - b]]))
    T = np.zeros((8, 8))
    for i, li in enumerate(LABELS):
        for j, lj in enumerate(LABELS):
            p = 1.0
            for q in range(3):
                p *= S[q][int(li[2 - q]), int(lj[2 - q])]
            T[i, j] = p
    tv = 0.5 * np.abs(M - T).sum(axis=1).mean()
    print(f"mean total-variation distance from independent per-qubit model = {tv:.4f}"
          "  (large => correlated readout error / crosstalk)")
    return dict(mean_err=mean_err, single=single, tv=tv)


def verdict(results):
    print("\n=== VERDICT RULES ===")
    for name, r in results.items():
        worst = max(max(a, b) for a, b in r["single"])
        tag = "CLEAN (<= 3% per-qubit)" if worst <= 0.03 else "NOT CLEAN"
        print(f"{name}: worst single-qubit flip probability = {worst:.3f}  -> {tag}")
    names = list(results)
    if len(names) >= 2:
        w = [max(max(a, b) for a, b in results[n]["single"]) for n in names]
        if w[0] > 0.03 and all(x > 0.03 for x in w[1:]):
            print("Both triples show large errors. This does NOT prove a script bug; check saved layouts, count association, and whether both triples are affected.")
        elif w[0] > 0.03 and any(x <= 0.03 for x in w[1:]):
            print("Original triple is worse than at least one alternative. This supports a qubit/triple-specific effect; validate layout before changing H2- circuits.")
        elif all(x <= 0.03 for x in w):
            print("Both triples are clean today. The earlier anomaly was not reproduced; compare timing and historical calibration.")


def mitigate_counts(counts, M):
    """Constrained least-squares inversion of confusion matrix M (rows prepared, cols observed).
    Helper for LATER use. Do not apply to H2- data until the confusion matrix is verified."""
    from scipy.optimize import nnls
    p_obs = np.zeros(8)
    tot = sum(counts.values())
    for k, v in counts.items():
        p_obs[int(k.replace(" ", ""), 2)] += v / tot
    x, _ = nnls(M.T, p_obs)
    return x / x.sum()


def run_local_test():
    """Offline mechanics test: asymmetric readout noise on qubits of a fake 3-qubit device."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from qiskit import transpile
    a = [0.46, 0.25, 0.13]; b = [0.03, 0.03, 0.03]
    nm = NoiseModel()
    for q in range(3):
        nm.add_readout_error(ReadoutError([[1 - a[q], a[q]], [b[q], 1 - b[q]]]), [q])
    sim = AerSimulator(noise_model=nm, seed_simulator=7)
    circs, meta = build_circuits([(0, 1, 2)])
    res = sim.run(transpile(circs, sim, optimization_level=0), shots=20000).result()
    cbl = {m["label"]: res.get_counts(i) for i, m in enumerate(meta)}
    M = counts_to_matrix(cbl)
    r = analyse(M, "local synthetic (injected P(1|0) = 0.46/0.25/0.13, P(0|1)=0.03)")
    for q in range(3):
        assert abs(r["single"][q][0] - a[q]) < 0.01 and abs(r["single"][q][1] - b[q]) < 0.01, "marginal mismatch"
    assert r["tv"] < 0.01
    # mitigation helper
    ideal = np.zeros(8); ideal[0b101] = 1.0
    obs = M[0b101]
    counts = {format(j, "03b"): int(round(obs[j] * 100000)) for j in range(8)}
    mit = mitigate_counts(counts, M)
    assert mit[0b101] > 0.999, mit
    # order-sensitivity check: the qubit with the injected 0.46 error must be c0 (rightmost bit)
    assert np.argmax([s[0] for s in r["single"]]) == 0
    print("\nLOCAL MECHANICS TEST PASSED (analysis recovers injected per-qubit errors; bit order verified).")


def make_service(a):
    """Connect using the saved profile already used successfully for the user's IBM jobs.

    Do not read or print the API token. Optional --instance narrows the account if needed.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    account_file = "/home/renjith/.qdrive_ibm_platform_account.json"
    kw = {"filename": account_file, "name": "qdrive_ibm_platform"}
    if a.instance:
        kw["instance"] = a.instance
    try:
        svc = QiskitRuntimeService(**kw)
        print("IBM Runtime account profile loaded: qdrive_ibm_platform")
        return svc
    except Exception as e:
        sys.exit(
            "Could not connect using the previously working IBM Runtime profile "
            f"({account_file}, name='qdrive_ibm_platform'): {e!r}\n"
            "No job was submitted. Do not paste or print your API token."
        )


def circuit_audit_record(qc, meta):
    """JSON-safe record of the submitted circuit's layout and measurement map."""
    measure_map = []
    for inst in qc.data:
        if getattr(inst.operation, "name", "") == "measure":
            measure_map.append({
                "physical_circuit_qubit_index": qc.find_bit(inst.qubits[0]).index,
                "classical_bit_index": qc.find_bit(inst.clbits[0]).index,
            })
    layout_text = None
    initial_map = None
    final_map = None
    try:
        layout = qc.layout
        layout_text = str(layout)
        if layout is not None and getattr(layout, "initial_layout", None) is not None:
            initial_map = {str(bit): int(index) for bit, index in layout.initial_layout.get_virtual_bits().items()}
        if layout is not None and getattr(layout, "final_layout", None) is not None:
            final_map = str(layout.final_layout)
    except Exception as exc:
        layout_text = f"layout metadata unavailable: {exc!r}"
    return {
        **meta,
        "transpiled_name": qc.name,
        "num_qubits": qc.num_qubits,
        "num_clbits": qc.num_clbits,
        "depth": qc.depth(),
        "operations": {str(k): int(v) for k, v in qc.count_ops().items()},
        "layout_repr": layout_text,
        "initial_virtual_to_physical": initial_map,
        "final_layout_repr": final_map,
        "measurement_map": measure_map,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--triples", default=None, help='e.g. "0,1,2;40,41,42"')
    ap.add_argument("--analyse-only", default=None)
    ap.add_argument("--local-test", action="store_true")
    ap.add_argument("--instance", default=None, help="optional IBM Runtime instance CRN/name; omit to use the saved profile's available instance")
    a = ap.parse_args()

    if a.local_test:
        run_local_test(); return

    if a.analyse_only:
        data = json.load(open(a.analyse_only))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        if a.shots < 1:
            ap.error("--shots must be a positive integer")
        svc = make_service(a)
        be = svc.backend(a.backend)
        if a.triples:
            triples = [tuple(int(x.strip()) for x in s.split(",")) for s in a.triples.split(";") if s.strip()]
            if len(triples) < 2:
                ap.error("--triples must specify both triples, e.g. --triples '0,1,2;40,41,42'")
            for tr in triples:
                if len(tr) != 3 or len(set(tr)) != 3 or any(q < 0 or q >= be.num_qubits for q in tr):
                    ap.error(f"Invalid physical-qubit triple {tr}; each must contain 3 distinct valid qubit indices")
            if set(triples[0]) & set(triples[1]):
                ap.error("For a clean comparison, the two supplied triples must not overlap")
        else:
            triples = [(0, 1, 2)] + best_triples(be, exclude=((0, 1, 2),), n=1)
        print("Backend:", be.name)
        print("Triples (logical q0,q1,q2 -> listed physical qubits):", triples)
        print("Planned workload:", 16, "circuits x", a.shots, "shots =", 16 * a.shots, "shots")
        circs, meta = build_circuits(triples)
        isa = transpile_all(circs, meta, be)
        # Confirm each circuit's three measurements target the selected physical triple.
        audit = []
        for qc, m in zip(isa, meta):
            mm = []
            for inst in qc.data:
                if getattr(inst.operation, "name", "") == "measure":
                    mm.append((qc.find_bit(inst.qubits[0]).index, qc.find_bit(inst.clbits[0]).index))
            expected_by_cbit = {i: int(q) for i, q in enumerate(m["triple"])}
            actual_by_cbit = {cbit: phys for phys, cbit in mm}
            if actual_by_cbit != expected_by_cbit:
                raise RuntimeError(f"Measurement mapping mismatch before submission for {qc.name}: expected cbit->physical {expected_by_cbit}, got {actual_by_cbit}. No job submitted.")
            audit.append(circuit_audit_record(qc, m))

        # Save the exact transpiled circuits before submission so a later failure still leaves an audit trail.
        from qiskit import qpy
        pretag = time.strftime("%Y%m%d_%H%M%S")
        qpy_path = f"stage8L1_readout_pre_submit_{pretag}.qpy"
        with open(qpy_path, "wb") as qf:
            qpy.dump(isa, qf)
        print("Saved transpiled QPY before submission:", qpy_path)
        print("Submitting one IBM Runtime SamplerV2 job now...")
        sampler = SamplerV2(mode=be)
        job = sampler.run(isa, shots=a.shots)
        job_id = job.job_id()
        print("JOB:", job_id, "submitted", time.strftime("%Y-%m-%d %H:%M:%S"))
        data = dict(
            experiment="Stage 8L1 readout triple comparison",
            job_id=job_id, backend=be.name, shots=a.shots,
            total_shots_requested=16 * a.shots,
            triples=[list(t) for t in triples], meta=meta,
            transpiled_circuits=audit, qpy_file=qpy_path,
            submitted_local=time.strftime("%Y-%m-%d %H:%M:%S %z"),
            counts=[], calibration={},
        )
        # Persist job ID and layout before waiting for completion.
        fn = f"stage8L1_readout_{job_id}.json"
        with open(fn, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print("Initial audit record saved:", fn)
        try:
            res = job.result()
        except Exception as exc:
            data["result_error"] = repr(exc)
            with open(fn, "w") as f:
                json.dump(data, f, indent=2, default=str)
            raise RuntimeError(f"IBM job {job_id} was submitted but result retrieval failed; job ID and audit are in {fn}: {exc!r}") from exc
        if len(res) != len(meta):
            raise RuntimeError(f"Expected {len(meta)} PUB results, received {len(res)} for job {job_id}; raw association not assumed")
        for i, r in enumerate(res):
            if not hasattr(r.data, "c"):
                raise RuntimeError(f"PUB {i} has no classical register 'c'; available data fields: {list(getattr(r.data, '__dict__', {}).keys())}")
            data["counts"].append(dict(r.data.c.get_counts()))
        for t in triples:
            data["calibration"][",".join(map(str, t))] = calibration_snapshot(be, t)
        data["completed_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        with open(fn, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print("Completed results and audit saved:", fn)

    results = {}
    for t, triple in enumerate(data["triples"]):
        cbl = {}
        for m, c in zip(data["meta"], data["counts"]):
            if m["triple_index"] == t:
                cbl[m["label"]] = c
        M = counts_to_matrix(cbl)
        results[str(triple)] = analyse(M, f"triple {triple}")
        print("calibration snapshot:", data.get("calibration", {}).get(",".join(map(str, triple))))
    verdict(results)


if __name__ == "__main__":
    main()

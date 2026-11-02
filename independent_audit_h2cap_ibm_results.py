#!/usr/bin/env python3
"""Independent post-run audit of the archived H2- qDRIVE IBM Runtime job.

This script ONLY retrieves an already-completed job and recomputes grouped H/W
expectations from its raw counts. It does not submit a job or consume new QPU shots.
Run in the existing qdrive_ibm310 environment from the qDRIVE project directory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService

DEFAULT_JOB = "db6ojgamb58s738c4t8g"
DEFAULT_STAMP = "h2cap_ibm_hardware_20261013_070039"
ETA = 1.30446839e-5
THRESHOLD_HA = -1.423580144198899
HARTREE_TO_EV = 27.211386245988
REF_E_EV = 0.21239112537024876
REF_GAMMA_EV = 0.07516884295821497
SHOTS_EXPECTED = 4096


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def group_value(bitstring: str, labels: list[str], coeffs: list[float]) -> float:
    bits = bitstring.replace(" ", "")
    if len(bits) != 3 or any(b not in "01" for b in bits):
        raise ValueError(f"Unexpected measured bitstring {bitstring!r}")
    value = 0.0
    # Qiskit count keys are c2 c1 c0; Pauli labels are q2 q1 q0.
    for label, coeff in zip(labels, coeffs):
        if len(label) != 3:
            raise ValueError(f"Unexpected Pauli label {label!r}")
        parity = 1
        for q in range(3):
            if label[2-q] != "I" and bits[2-q] == "1":
                parity *= -1
        value += float(coeff) * parity
    return value


def recompute_one(counts: dict, record: dict, expected_shots: int):
    labels, coeffs = record["pauli_labels"], record["coefficients"]
    if len(labels) != len(coeffs) or not counts:
        raise ValueError("Missing counts or mismatched Pauli labels/coefficients")
    n = sum(int(v) for v in counts.values())
    if n != expected_shots:
        raise ValueError(f"Expected {expected_shots} shots, got {n}")
    values, frequencies = [], []
    for bitstring, freq in counts.items():
        freq = int(freq)
        if freq < 0:
            raise ValueError("Negative count frequency")
        values.append(group_value(bitstring, labels, coeffs))
        frequencies.append(freq)
    values = np.asarray(values, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    mean = float(np.dot(values, frequencies) / n)
    sample_var = float(np.dot((values-mean)**2, frequencies)/(n-1)) if n > 1 else 0.0
    return mean, sample_var/n, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-id", default=DEFAULT_JOB)
    p.add_argument("--qpy", default=f"{DEFAULT_STAMP}_circuits.qpy")
    p.add_argument("--groups", default=f"{DEFAULT_STAMP}_groups.json")
    p.add_argument("--summary", default=f"{DEFAULT_STAMP}_results.json",
                   help="Existing aggregate JSON to compare against; optional if file absent")
    p.add_argument("--shots", type=int, default=SHOTS_EXPECTED)
    p.add_argument("--output", default="h2cap_ibm_independent_audit.json")
    args = p.parse_args()

    qpy_path, groups_path = Path(args.qpy), Path(args.groups)
    for f in (qpy_path, groups_path):
        if not f.is_file():
            raise FileNotFoundError(f"Required archive not found: {f.resolve()}")
    with qpy_path.open("rb") as f:
        circuits = qpy.load(f)
    records = load_json(groups_path)
    if not isinstance(records, list):
        raise TypeError("Mapping JSON must be a list of per-circuit records")
    if len(circuits) != 108 or len(records) != 108:
        raise ValueError(f"Expected 108 QPY circuits and mapping records; got {len(circuits)} and {len(records)}")

    # Audit archive order and mapping association before retrieving/reducing counts.
    mismatches = []
    for i, (circuit, rec) in enumerate(zip(circuits, records)):
        expected_name = f"{rec['observable']}_t{rec['triple_index']}_g{int(rec['group_index_within_triple']):02d}"
        if circuit.name != expected_name:
            mismatches.append({"index": i, "circuit_name": circuit.name, "expected_name": expected_name})
    if mismatches:
        raise ValueError("QPY/mapping order or identity mismatch: " + json.dumps(mismatches[:5]))

    accounts = QiskitRuntimeService.saved_accounts()
    account = accounts.get("default-ibm-quantum")
    if not account or not account.get("token"):
        raise RuntimeError("Saved IBM account/token is unavailable")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=account["token"],
    )
    job = service.job(args.job_id)
    status = str(job.status())
    if status != "JobStatus.DONE" and not status.endswith("DONE"):
        raise RuntimeError(f"Job {args.job_id} is not DONE (status={status}); refusing to audit partial results")
    result = job.result()
    if len(result) != 108:
        raise ValueError(f"Expected 108 result PUBs; got {len(result)}")

    counts_by_circuit = []
    per_circuit = []
    total_shots = 0
    for i, (pub, rec, circuit) in enumerate(zip(result, records, circuits)):
        counts = pub.data.meas.get_counts()
        mean, variance_of_mean, n = recompute_one(counts, rec, args.shots)
        total_shots += n
        counts_by_circuit.append(counts)
        per_circuit.append({
            "index": i, "circuit_name": circuit.name,
            "triple": rec["triple"], "triple_index": rec["triple_index"],
            "observable": rec["observable"], "group_index_within_triple": rec["group_index_within_triple"],
            "shots": n, "distinct_bitstrings": len(counts),
            "raw_counts": {str(k): int(v) for k, v in counts.items()},
            "group_estimate": mean, "variance_of_group_mean": variance_of_mean,
        })

    summaries = []
    triple_keys = []
    for rec in records:
        key = (int(rec["triple_index"]), tuple(rec["triple"]))
        if key not in triple_keys:
            triple_keys.append(key)
    for triple_index, triple in triple_keys:
        means, vars_mean = {"H": 0.0, "W": 0.0}, {"H": 0.0, "W": 0.0}
        group_counts = Counter()
        for item in per_circuit:
            if item["triple_index"] != triple_index:
                continue
            name = item["observable"]
            means[name] += item["group_estimate"]
            vars_mean[name] += item["variance_of_group_mean"]
            group_counts[name] += 1
        if group_counts != Counter({"H": 27, "W": 27}):
            raise ValueError(f"Unexpected group counts for triple {triple}: {group_counts}")
        hse, wse = float(np.sqrt(vars_mean["H"])), float(np.sqrt(vars_mean["W"]))
        energy = (means["H"] - THRESHOLD_HA) * HARTREE_TO_EV
        energy_se = hse * HARTREE_TO_EV
        width = 2 * ETA * means["W"] * HARTREE_TO_EV
        width_se = 2 * ETA * wse * HARTREE_TO_EV
        summaries.append({
            "triple_index": triple_index, "triple": list(triple),
            "groups_H": group_counts["H"], "groups_W": group_counts["W"],
            "H_Ha": means["H"], "H_SE_Ha": hse,
            "W": means["W"], "W_SE": wse,
            "energy_eV": float(energy), "energy_SE_eV": float(energy_se),
            "width_eV": float(width), "width_SE_eV": float(width_se),
            "reference_energy_eV": REF_E_EV, "reference_width_eV": REF_GAMMA_EV,
            "energy_delta_from_reference_eV": float(energy-REF_E_EV),
            "width_delta_from_reference_eV": float(width-REF_GAMMA_EV),
        })

    audit = {
        "audit_type": "independent raw-count recomputation",
        "job_id": args.job_id, "job_status": status,
        "backend": str(getattr(job, "backend", lambda: "unknown")()),
        "qpy_archive": str(qpy_path.resolve()), "mapping_file": str(groups_path.resolve()),
        "qpy_circuits": len(circuits), "mapping_records": len(records),
        "result_pubs": len(result), "shots_per_pub_expected": args.shots,
        "total_shots_recomputed": total_shots,
        "mitigation": "none; uses raw SamplerV2 counts",
        "reference_constants": {"ETA": ETA, "THRESHOLD_HA": THRESHOLD_HA,
                                "HARTREE_TO_EV": HARTREE_TO_EV,
                                "reference_energy_eV": REF_E_EV, "reference_width_eV": REF_GAMMA_EV},
        "results_by_triple": summaries,
        "per_circuit_audit": per_circuit,
        "note": "Uncertainties are within-job shot-noise estimates only; not full hardware/model systematic uncertainties.",
    }
    summary_path = Path(args.summary)
    if summary_path.is_file():
        old = load_json(summary_path)
        old_results = old.get("results_by_triple", [])
        comparisons = []
        for new in summaries:
            match = next((x for x in old_results if x.get("triple") == new["triple"]), None)
            if match is None:
                comparisons.append({"triple": new["triple"], "saved_summary_match": False})
            else:
                fields = ("H_Ha", "H_SE_Ha", "W", "W_SE", "energy_eV", "energy_SE_eV", "width_eV", "width_SE_eV")
                comparisons.append({"triple": new["triple"], "saved_summary_match": True,
                                    "differences_new_minus_saved": {k: float(new[k]-match[k]) for k in fields if k in match}})
        audit["comparison_to_saved_summary"] = comparisons
    Path(args.output).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"JOB: {args.job_id} | status={status}")
    print(f"Archive/mapping/result checks: {len(circuits)}/{len(records)}/{len(result)} PASS")
    print(f"Total shots independently recounted: {total_shots:,}")
    for s in summaries:
        print(f"Triple {tuple(s['triple'])}: E_r={s['energy_eV']:.10f} +/- {s['energy_SE_eV']:.10f} eV; "
              f"Gamma={s['width_eV']:.10f} +/- {s['width_SE_eV']:.10f} eV; "
              f"dE={s['energy_delta_from_reference_eV']:+.10f} eV; dGamma={s['width_delta_from_reference_eV']:+.10f} eV")
    if "comparison_to_saved_summary" in audit:
        print("Comparison to saved aggregate JSON:")
        print(json.dumps(audit["comparison_to_saved_summary"], indent=2))
    print(f"Full raw-count audit saved: {Path(args.output).resolve()}")
    print("No new IBM job was submitted; no additional QPU shots were consumed.")

if __name__ == "__main__":
    main()


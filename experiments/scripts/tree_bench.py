"""Test bench for extract_tree.py's selection model, in seconds instead of a full solve.

Two modes, both pass/fail against results/sub-01/network/bench_ref.json (the
sub-01 checks: the real ACA route and its lumen, the ACA link, both A1s fed
from their own ICA, the landmarks, the VB length, the radius against the T2w
flow void):

  region  The selection program of the given extract_tree.py options (any of
          them, after the bench's own) on a frozen candidate set (--cand),
          started from a reference tree on those candidates (--ref); only the
          arcs with an end in the --regions boxes are free (and every flow),
          the rest stays as in the reference. The boxes are solved exactly
          and merged by extract_tree.py's region search, to a tree no box
          improves (~10-60 s on one core per box). Reports the objective of
          the reference and of the result under the model, the segments
          changed per region, and the candidate-space checks. Answers "does
          this model still want the right local configuration", not what the
          whole tree becomes.
  check   A full run NAME (extract_tree.py -> fit_radius.py ->
          label_centerlines.py done): the candidate-space checks on its
          selection, the landmarks and the VB length of NAME_labeled.json,
          and the fitted diameter per label against the T2w void FWHM
          (median over cross-sections, as in the radius checks).

The candidate-space checks read the selection directly (arc polylines of the
used segments, in T1w world): coverage = share of the reference points within
--tol mm of a used segment; an A1 is fed from its ICA if the used arcs along
it run lateral -> medial.

usage:
  pixi run python tree_bench.py region --ref centerlines_tree_t2native_ax1
  pixi run python tree_bench.py region --ref NAME -- --w_turn 3 --cap_slack 3
  pixi run python tree_bench.py check centerlines_tree_t2native_ax1
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import extract_tree as et
import numpy as np
from common import RESULTS
from scipy.spatial import cKDTree

OUT = RESULTS / "network"
REF = OUT / "bench_ref.json"
# region boxes (T1w world centre, side mm)
REGIONS = {
    "cow": ([-1.0, 22.0, 2.0], 40.0),  # ICA termini, A1s, ACom, PComs, basilar tip
    "aca": ([2.5, 46.0, 41.0], 30.0),  # the faint ACA connector and its lumen
    "va": ([4.0, 4.4, -41.0], 30.0),  # VA confluence: the VB routing
    "siphon_L": ([-22.0, 20.0, -20.0], 40.0),  # left carotid siphon
}


def load_run(name):
    """Candidates, selection and voxel -> world map of a run."""
    sys.modules["__main__"].Segment = et.Segment  # pickled from extract_tree main
    cand = pickle.loads((OUT / f"{name}_candidates.pkl").read_bytes())
    if len(cand) == 5:
        cand = (*cand[:3], cand[2], *cand[3:])
    sel_path = OUT / f"{name}_selection.pkl"
    sel = pickle.loads(sel_path.read_bytes()) if sel_path.exists() else None
    M = np.array(cand[5]["voxel_to_t1"])
    return cand, sel, M


def used_polylines(segs, A, used, M):
    """World polylines of the used arcs, in their direction of traversal."""
    out = []
    for a in used:
        k, fwd = A[a][0], A[a][1]
        pts = segs[k].pts if fwd else segs[k].pts[::-1]
        out.append(pts @ M[:3, :3].T + M[:3, 3])
    return out


def candidate_checks(polys, ref, tol):
    """Pass/fail rows (name, value, ok) of a selection's used arc polylines."""
    allp = np.vstack(polys)
    kd = cKDTree(allp)
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(polys)])
    cover = lambda q: float((kd.query(np.asarray(q))[0] < tol).mean())
    rows = [
        ("ACA real route", cover(ref["aca_route"]), cover(ref["aca_route"]) >= 0.9),
        ("ACA lumen 537", cover(ref["aca_lumen"]), cover(ref["aca_lumen"]) >= 0.9),
        ("ACA link", cover([ref["aca_link"]]), cover([ref["aca_link"]]) == 1.0),
    ]
    for side, q in ref["a1"].items():
        q = np.asarray(q)
        d, i = kd.query(q)
        on = d < tol
        if on.mean() < 0.8:
            rows.append((f"A1-{side} fed from ICA-{side}", "not used", False))
            continue
        # direction of traversal along the A1: the polyline's own order at each
        # matched point, projected on the lateral -> medial direction
        votes = []
        for j in np.flatnonzero(on):
            poly = polys[owner[i[j]]]
            loc = i[j] - np.flatnonzero(owner == owner[i[j]])[0]
            lo, hi = max(loc - 2, 0), min(loc + 2, len(poly) - 1)
            t = poly[hi] - poly[lo]
            ql = q[min(j + 1, len(q) - 1)] - q[max(j - 1, 0)]
            votes.append(np.sign(t @ ql))
        fwd = float(np.mean(np.array(votes) > 0))
        rows.append((f"A1-{side} fed from ICA-{side}", f"{fwd:.0%}", fwd >= 0.8))
    return rows


def print_rows(title, rows):
    print(title)
    ok = True
    for nm, v, good in rows:
        v = f"{v:.0%}" if isinstance(v, float) else str(v)
        print(f"  {'PASS' if good else 'FAIL'}  {nm:32s} {v}")
        ok &= bool(good)
    return ok


def region(argv):
    p = argparse.ArgumentParser(description="region mode")
    p.add_argument("--ref", required=True, help="run NAME of the reference tree")
    p.add_argument(
        "--cand", default=None, help="run NAME of the candidates (default: --ref)"
    )
    p.add_argument("--regions", nargs="+", default=list(REGIONS), choices=REGIONS)
    p.add_argument("--tol", type=float, default=0.5, help="mm, coverage")
    p.add_argument(
        "--force_a1",
        action="store_true",
        help="forbid the arcs running medial -> lateral along an A1 (probe: what "
        "the model charges for both A1s fed from their ICA)",
    )
    p.add_argument("--box_time", type=float, default=300.0, help="s per box")
    own, model_argv = (
        (argv[: argv.index("--")], argv[argv.index("--") + 1 :])
        if "--" in argv
        else (argv, [])
    )
    b = p.parse_args(own)
    args = et.parser().parse_args(model_argv)
    args.workers = len(b.regions)
    args.lns_box_time = b.box_time
    args.lns_confirm = True
    ref = json.loads(REF.read_text())

    (segs, node_pos, _, node_size, _, cand_info), _, M = load_run(b.cand or b.ref)
    _, sel, _ = load_run(b.ref)
    A_r, parent_r, _, inlets = sel
    sp = np.array(cand_info["spacing_mm"])
    h = float(np.mean(sp))
    t0 = time.time()
    prog = et.selection_program(segs, len(node_pos), node_size, inlets, args, h)
    m = prog.build()
    A, trans, n_arc, n_tr = prog.A, prog.trans, prog.n_arc, prog.n_tr
    X = n_arc
    idx = {(k, fwd): a for a, (k, fwd, *_) in enumerate(A)}
    tr_idx = {tr: t for t, tr in enumerate(trans)}
    old = {a: idx[A_r[a][0], A_r[a][1]] for a in parent_r}
    old[-1] = -1
    x = np.zeros(m.nv)
    for b_, a_ in parent_r.items():
        x[old[b_]] = 1
        x[X + tr_idx[old[a_], old[b_]]] = 1
    if b.force_a1:
        n_off = 0
        for q in ref["a1"].values():
            q = np.asarray(q)
            kq = cKDTree(q)
            for a, (k, fwd, *_) in enumerate(A):
                pts = segs[k].pts if fwd else segs[k].pts[::-1]
                w = pts @ M[:3, :3].T + M[:3, 3]
                d, j = kq.query(w)
                on = d < b.tol
                if on.sum() >= 3 and np.mean(np.diff(j[on]) < 0) > 0.5:
                    m.ub_v[a] = 0  # runs against the lateral -> medial order
                    x[a] = 0
                    n_off += 1
        print(f"force_a1: {n_off} arcs forbidden")
        # the reference may use them: drop its subtrees there (the box solve
        # reconnects them), so the start stays a feasible tree
        used = {a for a in range(n_arc) if x[a] > 0.5}
        for t in np.flatnonzero(x[X : X + n_tr] > 0.5):
            a, c = trans[t]
            if a >= 0 and a not in used:
                x[X + t] = 0
                x[c] = 0
        changed = True
        while changed:
            changed = False
            alive = {a for a in range(n_arc) if x[a] > 0.5}
            for t in np.flatnonzero(x[X : X + n_tr] > 0.5):
                a, c = trans[t]
                if (a >= 0 and a not in alive) or c not in alive:
                    x[X + t] = 0
                    x[c] = 0
                    changed = True
    x_ref = m.complete(x)
    assert m.feasible(x_ref), "reference tree infeasible under this model"
    obj_ref = float(m.c @ x_ref)

    # boxes in the voxel-mm frame of node_xyz
    node_xyz = np.asarray(node_pos) * sp
    Minv = np.linalg.inv(M)
    u = np.array([a[2] for a in A])
    v = np.array([a[3] for a in A])
    par = np.array([a for a, _ in trans])
    chi = np.array([c for _, c in trans])
    boxes = []
    for r in b.regions:
        centre, side = REGIONS[r]
        cv = (Minv[:3, :3] @ centre + Minv[:3, 3]) * sp
        inb = np.all(np.abs(node_xyz - cv) <= side / 2, axis=1)
        fa = inb[u] | inb[v]
        ft = fa[chi] | ((par >= 0) & fa[np.maximum(par, 0)])
        free = np.r_[np.flatnonzero(fa), n_arc + np.flatnonzero(ft)]
        boxes.append((free, set(free.tolist())))
    t1 = time.time()
    model = (m.c, m.M, m.lb, m.ub, m.ub_v, m.integer, m.cs)
    x_new, obj, st = et.region_search(
        model, n_arc + n_tr, [boxes], x_ref, m.complete, m.feasible, args
    )
    x_new = m.complete(x_new)

    def arcset(xx):
        return {(A[a][0], A[a][1]) for a in np.flatnonzero(xx[:n_arc] > 0.5)}

    print(
        f"model built {t1 - t0:.0f} s, regions solved {st['s']:.0f} s "
        f"({st['box_solves']} box solves, max {st['box_s_max']:.0f} s, "
        f"status {st['box_status']})"
    )
    print(
        f"objective: reference {obj_ref:.2f}  result {obj:.2f}  ({obj - obj_ref:+.2f})"
    )
    ch = arcset(x_ref) ^ arcset(x_new)
    for r, (free, _) in zip(b.regions, boxes):
        fs = {A[a][0] for a in free if a < n_arc}
        print(
            f"  {r:9s} {sum(1 for a in free if a < n_arc):5d} free arcs, "
            f"{len({k for k, _ in ch} & fs)} segments changed"
        )
    terms = {
        "arcs": slice(0, n_arc),
        "transitions": slice(n_arc, n_arc + n_tr),
        "capacity": slice(m.E, m.Lf),
    }
    print(
        "  terms (ref -> result): "
        + ", ".join(
            f"{k} {m.c[s] @ x_ref[s]:.1f} -> {m.c[s] @ x_new[s]:.1f}"
            for k, s in terms.items()
        )
    )
    polys = used_polylines(segs, A, np.flatnonzero(x_new[:n_arc] > 0.5), M)
    return print_rows("checks (result):", candidate_checks(polys, ref, b.tol))


def profile_diameters(name, labels):
    """Median T2w void FWHM and fitted 2r per label (profile_t2w.py)."""
    res = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("profile_t2w.py")),
            name,
            *labels,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = {}
    for line in res.stdout.splitlines()[2:]:
        f = line.split()
        if len(f) >= 10 and f[0] in labels:
            out[f[0]] = (float(f[2]), float(f[8]))  # void FWHM, fitted 2r (medians)
    return out


def check(argv):
    p = argparse.ArgumentParser(description="check mode")
    p.add_argument("name", help="run NAME (labeled)")
    p.add_argument("--tol", type=float, default=0.5, help="mm, coverage")
    p.add_argument("--no_profile", action="store_true", help="skip the T2w radii")
    b = p.parse_args(argv)
    ref = json.loads(REF.read_text())
    (segs, *_), (A, parent, info, _), M = load_run(b.name)
    rows = candidate_checks(used_polylines(segs, A, sorted(parent), M), ref, b.tol)
    lab = json.loads((OUT / f"{b.name}_labeled.json").read_text())
    for k, x in ref["landmarks"].items():
        y = lab["landmarks"].get(k)
        d = np.inf if y is None else float(np.linalg.norm(np.subtract(x, y)))
        rows.append((f"landmark {k}", f"{d:.1f} mm", d <= 3.0))
    vb = lab["segments"].get("VB", {}).get("length_mm", 0.0)
    lo, hi = ref["vb_length_mm"]
    rows.append(("VB length", f"{vb:.1f} mm", lo <= vb <= hi))
    fit = json.loads((OUT / f"{b.name}_radius_fit.json").read_text())
    viol = fit.get("murray_violations_3x", {}).get("after")
    if viol is not None:
        rows.append(("Murray violations > 3x", viol, viol == 0))
    if not b.no_profile:
        tol = ref["diameter_tol"]
        for k, (void, d) in profile_diameters(b.name, ref["diameter_labels"]).items():
            rows.append(
                (f"diameter {k} vs T2w void", f"{d:.2f} / {void:.2f} mm",
                 abs(d / void - 1) <= tol)
            )  # fmt: skip
    m = info
    print(
        f"{b.name}: objective {m['objective']:.2f}, status {m['status']}, "
        f"solve {m['solve_s']} s"
    )
    return print_rows("checks:", rows)


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    ok = {"region": region, "check": check}[mode](rest)
    sys.exit(0 if ok else 1)

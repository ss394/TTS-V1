"""Worker processes (spawned): long-lived search islands and validators.

Each island keeps its population in memory between short time slices, so a slider move only changes the weights
sent with the next slice and a mass edit only refreshes the physics terms of cached candidates (no re-decode).
An island's plan says which search (BRKGA with memetic local search, late-acceptance hill climbing or NSGA-II) runs
on which decoder (certified compound boxes or the height-map). Messages are plain tuples over multiprocessing
queues; results always carry the context id they were computed for, so the coordinator drops stale ones.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np


def _phys_from_wire(phys: dict):
    from ..physics.assembly import PartPhys
    out = {}
    for pid, v in phys.items():
        out[pid] = None if v is None else PartPhys(v[0], np.asarray(v[1], float), np.asarray(v[2], float))
    return out


def _cand_wire(c):
    return {"keys": c.keys, "poses": [(pid, int(o), np.asarray(t, float)) for pid, o, t in c.poses],
            "raw": c.raw, "cv": float(c.cv), "unplaced": list(c.unplaced), "g": float(c.g), "meta": c.meta,
            "penalty": float(c.penalty), "extra_mass": [(float(m), np.asarray(p).tolist(), np.asarray(i).tolist())
                                                        for m, p, i in c.extra_mass]}


def _keys_from_poses(ctx, dec, poses, rng, dim):
    """Warm start: chromosome whose decode order follows the given layout from the bottom up."""
    n = len(ctx.ids)
    idx = {pid: i for i, pid in enumerate(ctx.ids)}
    k = rng.random(dim)
    placed = sorted([(float(t[2] + ctx.orient_bounds[pid][o, 2]), pid, o) for pid, o, t in poses if pid in idx])
    for rank, (_, pid, o) in enumerate(placed):
        i = idx[pid]
        k[i] = (rank + 0.5) / (n + 1)
        allowed = dec.parts[pid].allowed
        if o in allowed:
            k[n + i] = (allowed.index(o) + 0.5) / len(allowed)
        k[2 * n + i] = 0.0
    return k


def make_decoder(ctx, kind: str, h: float = 2.0):
    """Box decoder, or the height-map decoder (Stage 1 without locked parts only); Stage 2 has its own model."""
    if ctx.stage == 2:
        return ctx.stage2_model(), "box"
    if kind == "heightmap" and ctx.stage == 1 and not ctx.fixed:
        from ..decoders.heightmap import HeightmapDecoder
        recs = ctx.records
        meshes = {pid: ctx.hm_meshes.get(pid, (recs[pid].v, recs[pid].f)) for pid in ctx.ids}
        return HeightmapDecoder({pid: (*meshes[pid], list(p.allowed)) for pid, p in zip(ctx.ids, ctx.proxies)},
                                ctx.dec_lo, ctx.dec_hi, h=h, gap=ctx.gap_pp), "heightmap"
    return ctx.decoder(), "box"


def constructive_seeds(ctx, dec, which: int | None = None):
    recs = ctx.records
    orders = [np.argsort([-(recs[i].volume_mm3 or 0) for i in ctx.ids]),
              np.argsort([-float(np.prod(recs[i].boxes[:, 3:] - recs[i].boxes[:, :3], axis=1).sum()) for i in ctx.ids]),
              np.argsort([-recs[i].obb_extents_mm.max() for i in ctx.ids]),
              np.argsort([-recs[i].obb_extents_mm.min() for i in ctx.ids])]
    from ..optimise.brkga import seed_keys
    n = len(ctx.ids)
    dim = getattr(dec, "dim", 3 * n)
    allowed = [dec.parts[pid].allowed for pid in ctx.ids]
    out = []
    for k, order in enumerate(orders):
        if which is not None and k % 2 != which % 2:
            continue
        r = dec.best_fit_all_orientations(list(order))
        hint = getattr(r, "keys_hint", None)
        o_of = {pid: o for pid, o, _ in r.poses}
        s = seed_keys(list(order), [o_of.get(i, allowed[j][0]) if o_of.get(i) in allowed[j] else allowed[j][0]
                                    for j, i in enumerate(ctx.ids)], allowed)
        if hint is not None:                 # the decoder's own keys (orientation choices, extra variables)
            s[n:2 * n] = hint[0]
            s = np.concatenate([s[:2 * n], hint[1]])
        if len(s) < dim:
            s = np.concatenate([s, np.full(dim - len(s), 0.5)])
        out.append(s[:dim])
    return out


def fitting_keys(keys_list, dim: int) -> list:
    """Migrants an island can take: chromosomes of its own length. One of another length (a layout from another
    context, e.g. a Stage 2 chromosome — 3 extra keys — sent to a Stage 1 island) is skipped: placed in the
    population it broke every later crossover."""
    out = [np.asarray(k, float) for k in keys_list or []]
    return [k for k in out if k.shape == (dim,)]


class BrkgaIsland:
    VARIANTS = [{"elite_frac": 0.2, "mutant_frac": 0.15, "bias": 0.7}, {"elite_frac": 0.15, "mutant_frac": 0.2, "bias": 0.65},
                {"elite_frac": 0.25, "mutant_frac": 0.1, "bias": 0.75}, {"elite_frac": 0.2, "mutant_frac": 0.25, "bias": 0.6}]

    def __init__(self, ev, seeds, rng, pop_size, variant, ls=(2, 6)):
        from ..optimise.brkga import IslandState
        n3 = ev.dim
        seeds = seeds[:max(1, pop_size // 10)]
        pop = np.vstack([np.array(seeds).reshape(-1, n3), rng.random((pop_size - len(seeds), n3))])
        self.ev, self.state, self.params, self.ls = ev, IslandState(pop=pop, rng=rng), self.VARIANTS[variant % 4], ls

    def step(self, budget, weights, alpha, norm, k):
        from ..optimise.brkga import best_candidates, run_island
        e0, g0 = self.state.evaluations, self.state.generations
        run_island(self.state, self.ev, norm, weights, alpha, time_budget_s=budget, ls_elites=self.ls[0],
                   ls_moves=self.ls[1], **self.params)
        return best_candidates(self.state, k), self.state.evaluations - e0, self.state.generations - g0

    def refresh(self):
        for c in self.state.cands:
            self.ev.refresh(c)

    def inject(self, keys_list):
        from ..optimise.brkga import sort_key
        keys_list = fitting_keys(keys_list, self.ev.dim)
        if not self.state.cands:
            return 0
        order = sorted(range(len(self.state.cands)), key=lambda i: sort_key(self.state.cands[i]))
        for slot, keys in zip(reversed(order), keys_list):
            c = self.ev.evaluate(keys)
            self.state.cands[slot] = c
            self.state.pop[slot] = c.keys
            self.state.evaluations += 1
        return len(keys_list)


class LahcIsland:
    def __init__(self, ev, seeds, rng):
        from ..optimise.brkga import LahcState
        c0 = ev.evaluate(seeds[0]) if seeds else ev.evaluate(rng.random(ev.dim))
        for s in seeds[1:]:
            c = ev.evaluate(s)
            if (c.cv, c.raw.get("L", np.inf)) < (c0.cv, c0.raw.get("L", np.inf)):
                c0 = c
        self.ev, self.rng, self.state = ev, rng, LahcState(current=c0, best=c0)

    def step(self, budget, weights, alpha, norm, k):
        from ..optimise.brkga import run_lahc
        e0 = self.state.evaluations
        run_lahc(self.state, self.ev, norm, weights, alpha, self.rng, time_budget_s=budget)
        out = [self.state.best] + ([self.state.current] if self.state.current is not self.state.best else [])
        return out[:k], self.state.evaluations - e0, 1

    def refresh(self):
        self.ev.refresh(self.state.current)
        self.ev.refresh(self.state.best)
        self.state.hist = None

    def inject(self, keys_list):
        from ..optimise.brkga import sort_key
        keys_list = fitting_keys(keys_list, self.ev.dim)
        if keys_list:
            c = self.ev.evaluate(keys_list[0])
            self.state.evaluations += 1
            if sort_key(c) < sort_key(self.state.current):
                self.state.current = c
                self.state.hist = None
        return len(keys_list)


class Nsga2Island:
    """pymoo NSGA-II (decision U1) on the same chromosome; objectives L, D, T (raw), constraint CV ≤ 0."""

    def __init__(self, ev, seeds, rng, pop_size):
        self.ev, self.rng, self.pop_size = ev, rng, pop_size
        self.cache: dict[bytes, object] = {}
        self._setup(np.vstack([np.array(seeds).reshape(-1, ev.dim),
                               rng.random((pop_size - len(seeds), ev.dim))]) if seeds else None)
        self.evaluations = 0

    def _setup(self, X0):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        n3 = self.ev.dim
        self.problem = Problem(n_var=n3, n_obj=3, n_ieq_constr=1, xl=np.zeros(n3), xu=np.full(n3, 1.0 - 1e-9))
        kw = {"sampling": X0} if X0 is not None else {}
        self.alg = NSGA2(pop_size=self.pop_size, eliminate_duplicates=True, **kw)
        self.alg.setup(self.problem, termination=("n_gen", 10 ** 9), seed=int(self.rng.integers(2 ** 31)), verbose=False)

    def _eval(self, X):
        F = np.empty((len(X), 3))
        G = np.empty((len(X), 1))
        for i, x in enumerate(X):
            x = np.clip(x, 0.0, 1.0 - 1e-9)
            c = self.ev.evaluate(x)
            self.evaluations += 1
            self.cache[x.tobytes()] = c
            F[i] = [c.raw.get("L", 1e9), c.raw.get("D", 1e9), c.raw.get("T", 1e9)]
            G[i, 0] = c.cv
        return F, G

    def step(self, budget, weights, alpha, norm, k):
        from pymoo.core.evaluator import Evaluator
        from pymoo.problems.static import StaticProblem
        from ..optimise.brkga import score
        e0 = self.evaluations
        t_end = time.perf_counter() + budget
        gens = 0
        while time.perf_counter() < t_end:
            pop = self.alg.ask()
            F, G = self._eval(pop.get("X"))
            Evaluator().eval(StaticProblem(self.problem, F=F, G=G), pop)
            self.alg.tell(infills=pop)
            gens += 1
        opt = self.alg.opt
        out = []
        if opt is not None:
            for x in opt.get("X")[: max(k, 8)]:
                c = self.cache.get(np.clip(x, 0.0, 1.0 - 1e-9).tobytes()) or self.ev.evaluate(np.clip(x, 0, 1 - 1e-9))
                score(c, norm, weights, alpha)
                out.append(c)
        if len(self.cache) > 20000:
            self.cache.clear()
        return out, self.evaluations - e0, gens

    def refresh(self):
        X = self.alg.pop.get("X") if self.alg.pop is not None else None
        self.cache.clear()
        self._setup(X)

    def inject(self, keys_list):
        return 0


def insertion_check(ctx, poses, meta) -> dict:
    """±Z insertion order (exact blocking matrix of swept hulls + greedy removal, complete for straight lines).
    Edge boards move with their standoffs."""
    from ..collision.geom import to_world
    from ..constraints.stage2 import assembly_order, blocking_matrix, inner_box
    hulls, pieces = {}, {}
    stand = {}
    for pid, b in (meta or {}).get("standoffs", []):
        b = np.asarray(b, float)
        corners = np.array([[x, y, z] for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])])
        stand.setdefault(pid, []).append(corners)
    for pid, o, t in poses:
        r = ctx.records[pid]
        h = to_world(r.v[r.hull.vertex_ids], o, t)
        ps = [to_world(p.vertices, o, t) for p in r.cover.pieces] if r.cover.certified and r.cover.pieces else [h]
        extra = stand.get(pid, [])
        hulls[pid] = np.vstack([h] + extra) if extra else h
        pieces[pid] = ps + extra
    lo, hi = inner_box()
    block = blocking_matrix(hulls, pieces, (float(lo[2]), float(hi[2])))
    order, stuck = assembly_order(block)
    return {"order": order, "stuck": stuck}


def _validate_stage2(ctx, validator, poses, meta):
    """Stage 2: validate under the layout's own rules; a layout that breaks an engineering rule (outside the
    enclosure, a part not mountable, COM outside the CDS box, not insertable) is certified under the relaxed
    rules instead and returned as INFEASIBLE with its violations listed."""
    model = ctx.stage2_model()
    meta = meta or {}
    violations = []
    if meta.get("overflow_mm", 0) > 0:
        violations.append(f"exceeds the usable length by {meta['overflow_mm']:.1f} mm")
    for pid in meta.get("unmounted", []):
        name = ctx.records[pid].name if pid in ctx.records else pid
        violations.append(f"{name}: {meta.get('mounts', {}).get(pid, {}).get('how', 'not mountable')}")
    relaxed = meta.get("overflow_mm", 0) > 0
    rep = validator.validate(poses, model.spec(meta, relaxed=relaxed), want_planes=False)
    ins = None
    if rep.valid and not violations:
        ins = insertion_check(ctx, poses, meta)
        if ins["order"] is None:
            names = [ctx.records[p].name if p in ctx.records else p for p in ins["stuck"]]
            violations.append("no straight ±Z insertion order: " + ", ".join(names) + " interlock")
    summary = {"valid": rep.valid, "min_clearance_mm": rep.min_clearance_mm, "levels": rep.levels,
               "violations": rep.violations[:20], "pairs": len(rep.pairs), "statics": len(rep.statics),
               "seconds": rep.seconds, "validator_version": rep.validator_version,
               "engineering_violations": violations, "relaxed": relaxed,
               "insertion_order": ins["order"] if ins else None}
    return summary, None


def island_main(worker_id: int, role: str, cmd_q, res_q, base_seed: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ..collision.geom import part_geom_from_record
    from ..collision.validate import Validator
    from ..optimise.brkga import Evaluator
    from ..optimise.compact import compactor_from_records
    from ..optimise.scalarize import Normaliser
    from .context import load_context

    ctx = ev = island = None
    ctx_id = None
    mass_rev = -1
    validator = compactor = None
    plan = {}
    res_q.put(("ready", worker_id, role, os.getpid()))
    while True:
        cmd = cmd_q.get()
        kind = cmd[0]
        if kind == "stop":
            break
        try:
            if kind == "context":
                _, new_id, path, mass_rev, phys, target, warm, pop_size, counter, plan = cmd
                ctx = load_context(path)
                ctx_id = new_id
                if role == "validator":
                    validator = Validator({pid: part_geom_from_record(r) for pid, r in ctx.records.items()})
                    compactor = compactor_from_records(ctx.records, ctx.spec) if ctx.stage == 1 else None
                    res_q.put(("context_ok", worker_id, ctx_id, {}))
                    continue
                t0 = time.perf_counter()
                dec, dec_kind = make_decoder(ctx, plan.get("decoder", "box"), plan.get("h", 2.0))
                ev = Evaluator(dec, ctx.ids, _phys_from_wire(phys), ctx.orient_bounds, ctx.z0, target=target,
                               com_box=ctx.com_box)
                rng = np.random.default_rng(base_seed + 7919 * worker_id + 104729 * counter)
                seeds = constructive_seeds(ctx, dec, None if plan.get("all_seeds") else worker_id)
                seeds += [_keys_from_poses(ctx, dec, p, rng, ev.dim) for p in warm or []]
                search = plan.get("search", "brkga")
                if search == "lahc":
                    island = LahcIsland(ev, seeds, rng)
                elif search == "nsga2":
                    island = Nsga2Island(ev, seeds, rng, pop_size)
                else:
                    island = BrkgaIsland(ev, seeds, rng, pop_size, worker_id)
                res_q.put(("context_ok", worker_id, ctx_id, {"decoder": dec_kind, "search": search,
                                                              "setup_s": time.perf_counter() - t0}))
            elif kind == "phys":
                _, for_ctx, mass_rev, phys, target = cmd
                if ev is not None and for_ctx == ctx_id:
                    ev.phys = _phys_from_wire(phys)
                    ev.target = np.asarray(target, float)
                    island.refresh()
            elif kind == "inject":
                _, for_ctx, keys_list = cmd
                if island is not None and for_ctx == ctx_id:
                    island.inject(keys_list)
            elif kind == "run":
                _, job, for_ctx, budget, weights, alpha, norm_wire, k = cmd
                if island is None or for_ctx != ctx_id:
                    res_q.put(("island", worker_id, job, for_ctx, mass_rev, 0, 0, [], "no context"))
                    continue
                norm = Normaliser(epoch=norm_wire["epoch"], utopia=dict(norm_wire["utopia"]), nadir=dict(norm_wire["nadir"]))
                t0 = time.perf_counter()
                best, evals, gens = island.step(budget, weights, alpha, norm, k)
                res_q.put(("island", worker_id, job, ctx_id, mass_rev, evals, gens, [_cand_wire(c) for c in best],
                           f"{time.perf_counter() - t0:.2f}s"))
            elif kind == "validate":
                _, job, for_ctx, poses, want_planes, compact, meta = cmd
                if validator is None or for_ctx != ctx_id:
                    res_q.put(("validated", worker_id, job, for_ctx, None, None))
                    continue
                poses = [(pid, int(o), np.asarray(t, float)) for pid, o, t in poses]
                if ctx.stage == 2:
                    res_q.put(("validated", worker_id, job, ctx_id, *_validate_stage2(ctx, validator, poses, meta)))
                    continue
                extra = None
                rep = None
                if compact:
                    # certified drop compaction; kept only when it shortens the used length (moving parts down
                    # also moves the COM, so it is not applied for its own sake)
                    def top(ps):
                        return max(t[2] + ctx.orient_bounds[pid][o, 5] for pid, o, t in ps)
                    fixed = {pid for pid, _, _ in ctx.fixed}
                    cres = compactor.compact(poses, passes=2, fixed=fixed, time_budget_s=10.0)
                    if top(cres.poses) < top(poses) - 1e-6:
                        rep_c = validator.validate(cres.poses, ctx.spec, want_planes=want_planes)
                        if rep_c.valid:
                            rep, poses = rep_c, cres.poses
                            extra = {"poses": cres.poses, "moved_mm": cres.moved_mm, "seconds": cres.seconds}
                        else:
                            res_q.put(("error", worker_id, "compaction produced an invalid layout (discarded)",
                                       "; ".join(rep_c.violations[:3])))
                if rep is None:
                    rep = validator.validate(poses, ctx.spec, want_planes=want_planes)
                res_q.put(("validated", worker_id, job, ctx_id, {
                    "valid": rep.valid, "min_clearance_mm": rep.min_clearance_mm, "levels": rep.levels,
                    "violations": rep.violations[:20], "pairs": len(rep.pairs), "statics": len(rep.statics),
                    "seconds": rep.seconds, "validator_version": rep.validator_version}, extra))
        except Exception as exc:  # noqa: BLE001
            res_q.put(("error", worker_id, f"{type(exc).__name__}: {exc}", traceback.format_exc()))

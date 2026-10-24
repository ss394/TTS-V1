"""Coordinator: the single writer of optimisation state (protocol §H.2, §M; api-contract §4).

* Parts, settings and revisions live here; workers never decide anything.
* Search islands run in short time slices with the current weights; their candidates are re-scored here with the
  current masses and weights, and only the validator may promote a layout to VALIDATED.
* Every validated layout goes to the archive. A slider move re-scores the archive at once (no new geometry), a
  mass edit recomputes the physics terms of archived layouts at once, so the best validated layout for the current
  inputs is always available without waiting for the search.
* A layout belongs to the inputs it was validated for: its parts signature (ids, file hashes, units) and its
  constraint signature (stage, locks, allowed orientations, tags). When the current inputs return to those
  signatures, the layout is current again and is re-stamped with the current revisions.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..collision.geom import part_geom_from_record, to_world
from ..collision.validate import Validator
from ..constraints import stage2 as s2
from ..core.fileio import atomic_write_text
from ..core.rotations import ROTATIONS_F, as_lists
from ..geometry.stl_io import UNIT_TO_MM, read_stl_bytes
from ..objectives.metrics import ENCLOSURE_HI, ENCLOSURE_LO
from ..optimise.brkga import Evaluator
from ..optimise.scalarize import Normaliser, alpha_from, dominates, scalarise, slider_weights
from ..optimise.selector import ALGORITHMS, AlgorithmChoice, grid_map, part_stats, select_algorithm, sweep
from ..physics.assembly import PartPhys, assembly, com_box_check, inertia_metrics
from .context import SearchContext, build_context, packable, stage2_config, write_context
from .project import PartEntry, ProjectStore, build_record_file, now_iso, part_json, quick_inspect

VERSION = "0.6.0"
ALGORITHM = {"id": "hybrid_box", "label": "Hybrid BRKGA + box decoder", "reason": "archived layout"}
SEARCH_LABELS = {"brkga": "BRKGA + memetic local search", "lahc": "late-acceptance hill climbing",
                 "nsga2": "NSGA-II (pymoo)"}
DECODER_LABELS = {"box": "box decoder", "heightmap": "height-map decoder"}
SELECTOR_SETTLE_S = 1.0


def _kill_tree(pid: int | None) -> None:
    """Terminate a study process and the worker processes it spawned (they are its children)."""
    if not pid:
        return
    try:
        import psutil
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except Exception:  # noqa: BLE001 - already gone
        pass


def _bench_main(out_dir: str, opts: dict) -> None:
    """Background benchmark process (protocol §J.3) on the accepted dataset."""
    import json as _json
    from ..bench.instances import dataset_instances
    from ..bench.runner import PRESET_IDS, run_benchmarks
    from ..parts.build import build_part
    repo = Path(__file__).resolve().parents[4]
    man = _json.loads((repo / "data" / "synthetic" / "manifest.json").read_text())
    recs = {p["name"]: build_part((repo / "data" / "synthetic" / p["file"]).read_bytes(), p["name"], p["name"],
                                  file=p["file"]) for p in man["parts"] if p["role"] == "packable"}
    want = set(opts.get("instances", ["I1", "I2", "I3", "I4", "I5"]))
    insts = [i for i in dataset_instances(recs) if i.id in want]
    prog = Path(out_dir) / "progress.json"

    def progress(frac, r):
        atomic_write_text(prog, _json.dumps({"status": "RUNNING", "progress": frac}))
    run_benchmarks(insts, Path(out_dir), budget=float(opts.get("budget", 5000)), seeds=int(opts.get("seeds", 10)),
                   workers=int(opts.get("workers", 4)), presets=tuple(opts.get("presets", PRESET_IDS)), progress=progress)
    atomic_write_text(prog, _json.dumps({"status": "DONE", "progress": 1.0}))


def _mode_name(meta: dict | None) -> str:
    return "continuous_cable" if (meta or {}).get("mode") == "continuous" else "default"


def perturbed_weights(weights: dict, k: int, sigma: float = 0.3) -> dict:
    """Island k searches slightly perturbed weights (protocol §F.7) so nearby slider positions are already covered
    in the archive; island 0 uses the exact weights. Deterministic per island; all candidates are re-scored with
    the exact weights in the coordinator."""
    if k == 0:
        return dict(weights)
    rng = np.random.default_rng(1000 + k)
    keys = sorted(weights)
    f = np.exp(sigma * rng.standard_normal(len(keys)))
    w = np.array([weights[x] for x in keys]) * f
    w /= w.sum()
    return {x: float(v) for x, v in zip(keys, w)}


def island_plans(choice: AlgorithmChoice, n: int) -> list[dict]:
    """Which search runs on which decoder in each island, for the selected algorithm."""
    plans = []
    if choice.id == "nsga2_box":                 # benchmark evidence favours NSGA-II: most islands run it
        plans = [{"search": "nsga2", "decoder": "box", "h": 2.0} for _ in range(n)]
        if n >= 3:
            plans[0] = {"search": "brkga", "decoder": "box", "h": 2.0}
        if n >= 2:
            plans[-1] = {"search": "lahc", "decoder": "box", "h": 2.0}
        plans[0]["all_seeds"] = True
        return plans
    for i in range(n):
        dec = choice.decoder if choice.decoder != "mixed" else ("box" if i % 2 == 0 else "heightmap")
        plans.append({"search": "brkga", "decoder": dec, "h": 2.0})
    if n >= 2:
        plans[-1]["search"] = "lahc"
    if n >= 4:
        plans[-2] = {"search": "nsga2", "decoder": "heightmap" if choice.decoder == "heightmap" else "box", "h": 2.0}
    if plans:
        plans[0]["all_seeds"] = True
    return plans
POP_SIZE = 60
SLICE_S = 0.5
PROGRESS_EVERY_S = 0.5
CANDIDATE_EVERY_S = 0.25
MAX_SEARCH_S = 6 * 3600.0


def _f(x):
    """JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _f(o)
    return o


def layout_hash(stage: int, poses) -> str:
    items = sorted((pid, int(o), [int(round(v * 1000)) for v in t]) for pid, o, t in poses)
    return hashlib.sha256(json.dumps([stage, items]).encode()).hexdigest()[:16]


@dataclass
class LayoutEntry:
    id: str
    hash: str
    stage: int
    mode: str
    status: str
    parts_sig: str
    constraint_sig: str
    poses: list                          # [(part_id, orient, t_mm ndarray)] — t is an exact multiple of 1 µm
    geo: dict                            # geometry-only metrics (length, span, clearance …)
    validation: dict
    origin: str = "search"
    created: str = field(default_factory=now_iso)
    revs: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    g: float = math.inf
    normalised: dict = field(default_factory=dict)
    norm_epoch: int = 0
    cv: float = 0.0
    keys: list | None = None
    algorithm: dict = field(default_factory=lambda: dict(ALGORITHM))
    spec_sig: str = ""
    meta: dict = field(default_factory=dict)     # Stage 2: stack, corridor, mounts, standoffs, overflow

    def to_store(self) -> dict:
        d = asdict(self)
        d["poses"] = [[pid, int(o), [int(round(v * 1000)) for v in t]] for pid, o, t in self.poses]
        return jsonable(d)

    @staticmethod
    def from_store(d: dict) -> "LayoutEntry":
        d = dict(d)
        d["poses"] = [(pid, int(o), np.asarray(t, float) / 1000.0) for pid, o, t in d["poses"]]
        d["g"] = d.get("g") if d.get("g") is not None else math.inf
        return LayoutEntry(**d)


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.seq = 0
        self.applied = 0
        self.want_candidates = False


class WorkerHandle:
    def __init__(self, wid: int, role: str, proc, cmd_q):
        self.id, self.role, self.proc, self.cmd_q = wid, role, proc, cmd_q
        self.ready = False
        self.ctx_id: str | None = None
        self.ctx_sent: tuple | None = None
        self.busy = False
        self.job: int | None = None
        self.evaluations = 0
        self.generations = 0
        self.best_g: float | None = None
        self.plan: dict = {}
        self.info: dict = {}


class Coordinator:
    def __init__(self, store: ProjectStore, n_islands: int | None = None, n_validators: int = 2,
                 auto_start: bool = True, seed: int = 20261021, workers: bool = True):
        self.store = store
        cpu = os.cpu_count() or 4
        self.n_islands = n_islands if n_islands is not None else max(1, min(6, cpu - 4))
        self.n_validators = n_validators
        self.auto_start = auto_start
        self.seed = seed
        self.use_workers = workers
        self.records: dict = {}
        self.geoms: dict = {}
        self.quick: dict[str, dict] = {}
        self.build_errors: dict[str, str] = {}
        self.builds: dict[tuple[str, str], asyncio.Future] = {}
        self.archive: dict[str, LayoutEntry] = {}
        self.by_hash: dict[str, str] = {}
        self.norms: dict[str, Normaliser] = {}
        self.raw_pool: dict[str, list] = {}
        self.ctx: SearchContext | None = None
        self.ctx_path: Path | None = None
        self.ctx_counter = 0
        self.ev: Evaluator | None = None
        self.run_status = "IDLE"
        self.status_detail = ""
        self.clients: set[Client] = set()
        self.workers: list[WorkerHandle] = []
        self.res_q = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.job_counter = 0
        self.val_queue: list[tuple] = []           # (g, hash, poses, raw, keys, ctx_id, sigs)
        self.val_pending: dict[int, tuple] = {}
        self.pending_hashes: set[str] = set()
        self.best_id: str | None = None
        self.best_g_hist: list[tuple[float, float]] = []
        self.search_t0: float | None = None
        self.last_improve = time.monotonic()
        self.evaluations = 0
        self.last_progress = 0.0
        self.last_candidate = 0.0
        self.last_inject = 0.0
        self.last_norm_check = 0.0
        self.last_inbox = 0.0
        self.candidate: dict | None = None
        self.proof_cache: dict[tuple, dict] = {}
        self.build_pool: ProcessPoolExecutor | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopping = False
        self.ready = False
        self.errors: list[str] = []
        self._qual: dict[tuple[str, str], bool] = {}
        self.last_pareto = 0.0
        self.search_requested = False
        self.choice: AlgorithmChoice | None = None
        self.choice_pending: tuple[AlgorithmChoice, float] | None = None
        self.least_id: str | None = None           # Stage 2: least-violating certified candidate
        self.selector_log: list[dict] = []
        self._stats = None
        self._stats_sig = None
        self.ladder_pool: ProcessPoolExecutor | None = None
        self.ladders: dict[str, dict] = {}         # part_id -> {future, path, mtime, status}
        self.reps: dict[str, dict] = {}            # part_id -> {algorithm: Representation} (chosen levels)
        self.last_ladder_poll = 0.0
        self.auto_ladder = True
        self.ladder_opts: dict = {}
        self.attribution_opts: dict = {}
        self.cpsat_pool: ProcessPoolExecutor | None = None
        self.cpsat_time_s = 60.0
        self.cpsat_results: dict[str, dict] = {}     # parts_sig -> {status, length_mm, lower_bound_mm, ...}
        self.use_cpsat = True
        self.evidence: list[dict] = []               # benchmark evidence rows for the selector (M8)
        self.evidence_source: str | None = None
        self.studies: dict[str, dict] = {}           # "size" | "benchmark" -> {proc, path, mtime}
        self.mfg_budget_s = 90.0

    # =========================================================================================== lifecycle
    @property
    def revs(self) -> dict:
        return {"epoch": self.store.counters["epoch"], **self.store.revs}

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.store.counters["epoch"] = self.store.counters.get("epoch", 0) + 1
        self.store.seed_from_dataset()
        self.store.save()
        self._load_archive()
        self._load_evidence()
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[var] = "1"
        spawn = mp.get_context("spawn")
        self.build_pool = ProcessPoolExecutor(max_workers=2, mp_context=spawn)
        self.ladder_pool = ProcessPoolExecutor(max_workers=1, mp_context=spawn)
        self.cpsat_pool = ProcessPoolExecutor(max_workers=1, mp_context=spawn)
        for e in self.store.entries.values():
            self._load_ladder(e)
        if self.use_workers:
            self.res_q = spawn.Queue()
            from .worker import island_main
            roles = ["island"] * self.n_islands + ["validator"] * self.n_validators
            for wid, role in enumerate(roles):
                q = spawn.Queue()
                p = spawn.Process(target=island_main, args=(wid, role, q, self.res_q, self.seed), daemon=True)
                p.start()
                self.workers.append(WorkerHandle(wid, role, p, q))
            threading.Thread(target=self._pump, daemon=True).start()
        # part records: cache hits load at once, the rest build in the background
        for e in list(self.store.entries.values()):
            rec = self.store.load_record(e)
            if rec is not None:
                self._set_record(rec)
            else:
                self._start_build(e, e.units)
        self._refresh_inputs("start", bump_parts=False)
        self.ready = True
        if self.auto_start and self.ctx is not None:
            self._set_status("SEARCHING", "searching")
        self._tasks.append(asyncio.create_task(self._ticker()))

    async def stop(self) -> None:
        self._stopping = True
        for t in self._tasks:
            t.cancel()
        for name in list(self.studies):
            self.stop_study(name)
            s = self.studies[name]
            _kill_tree(s["proc"].pid)
        for w in self.workers:
            try:
                w.cmd_q.put(("stop",))
            except Exception:  # noqa: BLE001
                pass
        for w in self.workers:
            w.proc.join(timeout=3)
            if w.proc.is_alive():
                w.proc.terminate()
        if self.res_q is not None:
            try:
                self.res_q.put(("shutdown",))
            except Exception:  # noqa: BLE001
                pass
        if self.build_pool:
            self.build_pool.shutdown(wait=False, cancel_futures=True)
        for pool in (self.ladder_pool, self.cpsat_pool):
            if not pool:
                continue
            procs = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for p in procs:                     # a ladder or CP-SAT run can take minutes: do not wait for it
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
        self.store.save()

    def _pump(self) -> None:
        while True:
            try:
                msg = self.res_q.get()
            except (EOFError, OSError):
                return
            if msg[0] == "shutdown" or self._stopping:
                return
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self._on_worker_msg, msg)

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                self._error(f"coordinator tick failed: {exc!r}")

    def _tick(self) -> None:
        now = time.monotonic()
        if now - self.last_inbox > 3.0:
            self.last_inbox = now
            new = self.store.scan_inbox()
            for e in new:
                self._on_new_entry(e)
            if new:
                self._bump("parts_rev")
                self.broadcast("project.changed", {"reason": f"{len(new)} file(s) picked up from the inbox"})
        if self.choice_pending is not None and now - self.choice_pending[1] >= SELECTOR_SETTLE_S:
            self._apply_choice(self.choice_pending[0], restart=True)
        if (self.ladders or self.studies) and now - self.last_ladder_poll > 1.0:
            self.last_ladder_poll = now
            self._poll_ladders()
            self._poll_studies()
        if self.run_status == "SEARCHING":
            self._dispatch()
            self._dispatch_validation()
            if now - self.last_inject > 10.0:
                self.last_inject = now
                self._migrate()
            if now - self.last_norm_check > 5.0:
                self.last_norm_check = now
                self._maybe_new_epoch()
            if now - self.last_progress > PROGRESS_EVERY_S:
                self.last_progress = now
                self.broadcast("progress", self._progress())
            stall = 15.0 + 6.0 * float(self.store.settings["sliders"].get("speed_quality", 50))
            if self.best_id is not None and now - self.last_improve > stall:
                self._set_status("CONVERGED", f"no improvement for {stall:.0f} s (speed/quality slider)")
            elif self.search_t0 is not None and now - self.search_t0 > MAX_SEARCH_S:
                self._set_status("CONVERGED", "6 h time limit reached")
        elif self.val_queue or self.val_pending:
            self._dispatch_validation()

    # =========================================================================================== records
    def _set_record(self, rec) -> None:
        self.records[rec.id] = rec
        self.geoms[rec.id] = part_geom_from_record(rec) if rec.status != "NEEDS_UNITS" else None
        self.build_errors.pop(rec.id, None)

    def _start_build(self, e: PartEntry, units: str) -> asyncio.Future:
        key = (e.id, units)
        if key in self.builds and not self.builds[key].done():
            return self.builds[key]
        cache = self.store.cache_path(e, units)
        if cache.exists():
            fut = self.loop.create_future()
            fut.set_result(str(cache))
        else:
            cf = self.build_pool.submit(build_record_file, str(self.store.stl_path(e)), str(cache), e.name, e.id, units,
                                        True, e.file)
            fut = asyncio.wrap_future(cf, loop=self.loop)
        self.builds[key] = fut
        fut.add_done_callback(lambda f, pid=e.id, u=units: self._on_build_done(pid, u, f))
        return fut

    def _on_build_done(self, pid: str, units: str, fut) -> None:
        e = self.store.entries.get(pid)
        if e is None or e.units != units:
            return
        if fut.cancelled():
            return
        if fut.exception() is not None:
            self.build_errors[pid] = f"analysis failed: {fut.exception()!r}"
            self._bump("parts_rev")
            self.broadcast("project.changed", {"reason": f"{e.name}: analysis failed"})
            return
        rec = self.store.load_record(e)
        if rec is None:
            self.build_errors[pid] = "analysis produced no record"
            return
        self._set_record(rec)
        self._refresh_inputs(f"part {e.name} analysed")
        self._maybe_auto_ladder(e)

    def _on_new_entry(self, e: PartEntry) -> None:
        try:
            self.quick[e.id] = quick_inspect(self.store.stl_path(e).read_bytes())
        except ValueError as exc:
            self.build_errors[e.id] = str(exc)
        self._start_build(e, e.units)          # speculative build with the suggested units

    def part_list(self) -> list[dict]:
        out = []
        for pid, e in self.store.entries.items():
            rec = self.records.get(pid)
            if rec is not None and rec.units != e.units:
                rec = None
            building = (pid, e.units) in self.builds and not self.builds[(pid, e.units)].done()
            d = part_json(e, rec, self.quick.get(pid) or self._quick(e), building=building)
            if pid in self.build_errors and rec is None:
                d["status"], d["status_detail"] = "WITHHELD", self.build_errors[pid]
            out.append(d)
        return jsonable(out)

    def _quick(self, e: PartEntry) -> dict:
        if e.id not in self.quick:
            try:
                self.quick[e.id] = quick_inspect(self.store.stl_path(e).read_bytes())
            except (ValueError, OSError):
                self.quick[e.id] = {}
        return self.quick[e.id]

    def _active_records(self) -> dict:
        out = {}
        for pid, e in self.store.entries.items():
            rec = self.records.get(pid)
            if rec is not None and rec.units == e.units and e.units_confirmed and packable(rec):
                out[pid] = rec
        return out

    # =========================================================================================== inputs
    def _bump(self, key: str) -> None:
        self.store.revs[key] = self.store.revs.get(key, 0) + 1
        self.store.save()

    def _phys(self) -> dict:
        out = {}
        for pid, rec in self._active_records().items():
            if rec.com_can_mm is None or rec.mass_kg is None:
                out[pid] = None
            else:
                out[pid] = PartPhys(float(rec.mass_kg), np.asarray(rec.com_can_mm, float), rec.inertia_com_kgm2())
        return out

    def _phys_wire(self) -> dict:
        return {pid: None if p is None else (p.mass_kg, p.com_can_mm.tolist(), p.inertia_com_kgm2.tolist())
                for pid, p in self._phys().items()}

    def _target(self) -> list[float]:
        return [float(x) for x in self.store.settings.get("target_com_mm", [0, 0, 0])]

    def _confirmed_builds_pending(self) -> int:
        n = 0
        for (pid, units), f in self.builds.items():
            e = self.store.entries.get(pid)
            if not f.done() and e is not None and e.units_confirmed and e.units == units:
                n += 1
        return n

    def _refresh_inputs(self, reason: str, bump_parts: bool = True) -> None:
        """Rebuild the search context if parts or constraints changed; re-stamp, re-score, re-select."""
        if bump_parts:
            self._bump("parts_rev")
        pending = self._confirmed_builds_pending()
        if pending:
            # wait for the last analysis so the workers restart once, not once per part
            if self.ctx is None:
                self._set_status("IDLE", f"analysing {pending} part(s)…")
            self.broadcast("project.changed", {"reason": reason})
            return
        recs = self._active_records()
        stage = int(self.store.settings.get("stage", 1))
        old = self.ctx
        if not recs:
            self.ctx = None
            self.ev = None
            self.best_id = None
            self._set_status("IDLE", "no packable parts")
            self.broadcast("project.changed", {"reason": reason})
            return
        ctx = build_context(recs, stage, {pid: self.reps[pid] for pid in recs if pid in self.reps},
                            stage2_mode=self._stage2_mode())
        if old is None or ctx.ctx_id != old.ctx_id:
            if old is not None and (ctx.constraint_sig != old.constraint_sig or ctx.stage != old.stage) \
                    and ctx.parts_sig == old.parts_sig:
                self._bump("constraint_rev")
            self.ctx = ctx
            self.ctx_counter += 1
            self.ctx_path = write_context(ctx, self.store.cache / "contexts")
            self.norms.setdefault(ctx.ctx_id, Normaliser())
            self.raw_pool.setdefault(ctx.ctx_id, [])
            self.val_queue = []
            warm = []
            best = self.archive.get(self.best_id) if self.best_id else None
            if best is not None:
                warm.append(best.poses)
            self._apply_choice(self._compute_choice(), restart=False)
            for w in self.workers:
                self._send_context(w, warm)
            self._start_cpsat()
            self.last_improve = time.monotonic()
            self.search_t0 = time.monotonic()
            if self.run_status in ("CONVERGED", "IDLE") and (self.auto_start or self.search_requested):
                self._set_status("SEARCHING", f"inputs changed: {reason}")
        self.ev = Evaluator(None, ctx.ids, self._phys(), ctx.orient_bounds, ctx.z0, target=self._target(),
                            com_box=ctx.com_box)
        self._restamp_and_rescore(physics=True)
        self.broadcast("project.changed", {"reason": reason})
        self._publish_best("archive", force=True)
        if self.best_id is None and self.least_id is not None:
            self.broadcast("layout.infeasible", {"layout": self.layout_json(self.archive[self.least_id]),
                                                 "verdict": "NO FEASIBLE LAYOUT FOUND (not proven infeasible)"})

    def _send_context(self, w: WorkerHandle, warm=()) -> None:
        if self.ctx is None or self.ctx_path is None:
            return
        tag = (self.ctx.ctx_id, self.ctx_counter)
        if w.ctx_sent == tag:
            return
        w.ctx_sent = tag
        w.ctx_id = None
        w.busy = False
        islands = [x for x in self.workers if x.role == "island"]
        plans = island_plans(self.choice, len(islands)) if self.choice is not None else []
        w.plan = plans[islands.index(w)] if w.role == "island" and plans else {}
        w.cmd_q.put(("context", self.ctx.ctx_id, str(self.ctx_path), self.store.revs["mass_rev"], self._phys_wire(),
                     self._target(), list(warm), POP_SIZE, self.ctx_counter, w.plan))

    # =========================================================================================== selector
    def _part_stats(self):
        recs = self._active_records()
        sig = self.ctx.parts_sig if self.ctx is not None else None
        if self._stats is None or self._stats_sig != sig:
            self._stats = part_stats(list(recs.values()))
            self._stats_sig = sig
        return self._stats

    def _mode(self) -> str:
        s = self.store.settings
        return "continuous_cable" if int(s["stage"]) == 2 and float(s["sliders"].get("cable", 0)) >= 70 else "default"

    def _stage2_mode(self) -> str:
        from ..constraints.stage2_model import mode_for
        return mode_for(float(self.store.settings["sliders"].get("cable", 0)))

    def _compute_choice(self) -> AlgorithmChoice:
        s = self.store.settings
        return select_algorithm(int(s["stage"]), s["sliders"], self._part_stats(), self._mode(), self.evidence or None)

    def _apply_choice(self, choice: AlgorithmChoice, restart: bool) -> None:
        changed = self.choice is None or choice.id != self.choice.id
        self.choice = choice
        self.choice_pending = None
        if changed:
            self.selector_log.append({"at": now_iso(), "algorithm": choice.id, "reasons": choice.reasons})
            del self.selector_log[:-200]
        self.broadcast("selector.changed", self.selector())
        if changed and restart and self.ctx is not None:
            self.ctx_counter += 1
            best = self.archive.get(self.best_id) if self.best_id else None
            warm = [best.poses] if best is not None else []
            for w in self.workers:
                if w.role == "island":
                    self._send_context(w, warm)

    def _algo_for(self, w: WorkerHandle) -> dict:
        c = self.choice
        p = w.plan or {}
        detail = f"{SEARCH_LABELS.get(p.get('search'), p.get('search'))} on the {DECODER_LABELS.get(p.get('decoder'), '?')}"
        return {"id": c.id if c else "hybrid_box", "label": c.label if c else ALGORITHMS["hybrid_box"],
                "reason": c.reasons[0] if c and c.reasons else "", "search": p.get("search"), "decoder": p.get("decoder"),
                "island": w.id, "detail": detail}

    def _current(self, e: LayoutEntry) -> bool:
        """A layout is current when it was validated under the same parts and validation rules and satisfies the
        current allowed orientations and locks (so locking a part where it already is keeps the layout)."""
        if self.ctx is None:
            return False
        key = (e.id, self.ctx.ctx_id)
        q = self._qual.get(key)
        if q is None:
            q = self._qual[key] = self._qualifies(e, self.ctx)
        return q

    @staticmethod
    def _qualifies(e: LayoutEntry, ctx: SearchContext) -> bool:
        if e.parts_sig != ctx.parts_sig or e.stage != ctx.stage or e.spec_sig != ctx.spec_sig:
            return False
        um = lambda t: tuple(int(round(v * 1000)) for v in t)  # noqa: E731
        pose = {pid: (int(o), um(t)) for pid, o, t in e.poses}
        if set(pose) != set(ctx.ids):
            return False
        allowed = {p.id: set(p.allowed) for p in ctx.proxies}
        if any(o not in allowed[pid] for pid, (o, _) in pose.items()):
            return False
        return all(pose[pid] == (int(o), um(t)) for pid, o, t in ctx.fixed)

    def _restamp_and_rescore(self, physics: bool) -> None:
        for e in self.archive.values():
            if not self._current(e) or e.status not in ("VALIDATED", "FINAL", "INFEASIBLE"):
                continue
            if physics:
                self._compute_physics(e)
            e.revs = {k: self.store.revs[k] for k in ("parts_rev", "constraint_rev", "mass_rev", "weights_rev")}
            self._score_entry(e)
        self._select_best()

    # =========================================================================================== scoring
    def _weights(self) -> dict:
        s = self.store.settings
        return slider_weights(s["sliders"], int(s["stage"]), "default")

    def _norm(self) -> Normaliser:
        if self.ctx is None:
            return Normaliser()
        return self.norms.setdefault(self.ctx.ctx_id, Normaliser())

    def _score_raw(self, raw: dict) -> tuple[float, dict]:
        norm = self._norm()
        alpha = alpha_from(self.store.settings["sliders"])
        if not norm.utopia:
            return float(raw.get("L", math.inf)), {}
        n = norm.norm(raw)
        g = scalarise(n, self._weights(), alpha)
        out = {"L": n.get("L"), "D": n.get("D"),
               "I": (alpha * n["T"] + (1 - alpha) * n["P"]) if "T" in n and "P" in n else None,
               "G": n.get("G"), "H": n.get("H"), "C": n.get("C")}
        return g, out

    def _score_entry(self, e: LayoutEntry) -> None:
        e.g, e.normalised = self._score_raw(e.raw)
        e.norm_epoch = self._norm().epoch

    def _compute_physics(self, e: LayoutEntry) -> None:
        phys = self._phys()
        items, excluded = [], []
        for pid, o, t in e.poses:
            p = phys.get(pid)
            if p is None:
                excluded.append(pid)
                continue
            r = ROTATIONS_F[o]
            items.append((p.mass_kg, r @ p.com_can_mm + t, r @ p.inertia_com_kgm2 @ r.T))
        hardware = []
        if e.meta.get("hardware"):                 # Manufacturable Design: every bracket, spacer and standoff
            hardware = [(m, np.asarray(c, float), np.asarray(i, float)) for m, c, i in e.meta["hardware"]]
            items += hardware
        elif e.meta.get("standoffs"):
            from ..constraints.stage2_model import Stage2Model
            hardware = Stage2Model.standoff_mass(e.meta["standoffs"])
            items += hardware
        m = dict(e.geo)
        if hardware:
            m["hardware_mass_kg"] = float(sum(h[0] for h in hardware))
        m["mass_excluded_parts"] = excluded
        raw = {"L": e.geo["length_used_mm"], "span": e.geo["span_mm"]}
        if "margin_objective" in e.geo and e.stage == 2:
            raw["G"] = e.geo["margin_objective"]
        if items:
            a = assembly([], [], np.zeros((0, 3)), items)
            im = inertia_metrics(a.inertia_kgm2)
            cb = com_box_check(a.com_mm, self._target())
            m.update({"mass_kg": a.mass_kg, "com_mm": a.com_mm.tolist(), "inertia_kgm2": a.inertia_kgm2.tolist(), **im, **cb})
            if e.stage == 2:
                fm, fc, fi = s2.frame_mass_props()
                m["frame_mass_kg"] = fm
                m["com_box_note"] = "CDS COM box checked on the parts only (frame mass excluded, decision U7)"
            raw.update({"D": cb["com_error_mm"], "T": im["trace_kgm2"], "P": im["products_norm_kgm2"],
                        "lambda_max": im["lambda_max_kgm2"], "com": a.com_mm.tolist()})
            if excluded:
                m["mass_note"] = f"incomplete — {len(excluded)} part(s) excluded (mass properties withheld)"
            e.cv = 0.0
            if e.stage == 2:
                if excluded:
                    m["com_box_ok"] = None
                    m["com_box_verdict"] = "NOT ASSESSABLE"
                excess = np.maximum(np.abs(a.com_mm) - np.array([20.0, 20.0, 70.0]), 0)
                e.cv = float(excess.sum() / 10.0)
        if e.stage == 2:
            ev_ = list(e.validation.get("engineering_violations", []))
            e.cv += float(len(e.meta.get("unmounted", []))) + float(e.meta.get("overflow_mm", 0.0)) / 10.0
            if m.get("com_box_ok") is False:
                ev_.append("COM outside the CDS box (|x| ≤ 20, |y| ≤ 20, |z| ≤ 70 mm): "
                           f"({m['com_mm'][0]:.1f}, {m['com_mm'][1]:.1f}, {m['com_mm'][2]:.1f}) mm")
            if m.get("com_box_ok") is None and excluded:
                ev_.append("COM box NOT ASSESSABLE: mass properties withheld for " + ", ".join(excluded))
            m["violations"] = ev_
            m["cable_blocked_fraction"] = e.meta.get("H")
            m["mounting"] = e.meta.get("mounts")
            if e.status in ("VALIDATED", "INFEASIBLE"):
                if not e.validation.get("certified"):
                    e.status = "REJECTED"
                elif ev_ or e.cv > 0:
                    e.status = "INFEASIBLE"
                else:
                    e.status = "VALIDATED"
        m.setdefault("insertion_order", None)
        m.setdefault("interlocked", [])
        m.setdefault("cable_blocked_fraction", None)
        m.setdefault("hardware_cost_rel", e.meta.get("hardware_cost_rel"))
        e.metrics = jsonable(m)
        e.raw = raw

    def _select_best(self) -> None:
        cur = [e for e in self.archive.values() if self._current(e) and e.status in ("VALIDATED", "FINAL")]
        if not cur:
            self.best_id = None
        else:
            best = min(cur, key=lambda e: (e.cv, e.g, e.raw.get("L", math.inf), e.id))
            if best.id != self.best_id:
                self.best_id = best.id
        self._select_least(publish=False)

    def _select_least(self, publish: bool = True) -> None:
        """Stage 2 with no feasible layout: the least-violating certified candidate (shown, never as VALIDATED)."""
        prev = self.least_id
        if self.best_id is not None:
            self.least_id = None
        else:
            inf = [e for e in self.archive.values() if self._current(e) and e.status == "INFEASIBLE"]
            self.least_id = min(inf, key=lambda e: (e.cv, e.g, e.id)).id if inf else None
        if publish and self.least_id is not None and self.least_id != prev:
            self.broadcast("layout.infeasible", {"layout": self.layout_json(self.archive[self.least_id]),
                                                 "verdict": "NO FEASIBLE LAYOUT FOUND (not proven infeasible)"})

    def _maybe_new_epoch(self, force: bool = False) -> None:
        if self.ctx is None:
            return
        norm = self._norm()
        raws = [e.raw for e in self.archive.values() if self._current(e) and e.status in ("VALIDATED", "FINAL")]
        raws += self.raw_pool.get(self.ctx.ctx_id, [])[-300:]
        raws = [r for r in raws if "D" in r]
        if len(raws) < 10:
            return
        if not force and norm.utopia:
            # only when some value fell outside the current range by > 5 % of it
            out = False
            for k in ("L", "D", "T", "P"):
                if k not in norm.utopia:
                    continue
                lo, hi = norm.utopia[k], norm.nadir[k]
                span = hi - lo
                vals = [r[k] for r in raws if k in r]
                if vals and (min(vals) < lo - 0.05 * span or max(vals) > hi + 0.05 * span):
                    out = True
            if not out:
                return
        norm.update_from(raws)
        prev = self.best_id
        for e in self.archive.values():
            if self._current(e):
                self._score_entry(e)
        self._select_best()
        if self.best_id != prev:
            self._publish_best("archive")
        self.broadcast("pareto.update", self.pareto())

    # =========================================================================================== search
    def _set_status(self, status: str, detail: str = "") -> None:
        if status == self.run_status and detail == self.status_detail:
            return
        if status == "SEARCHING" and self.run_status != "SEARCHING":
            self.last_improve = time.monotonic()
            if self.search_t0 is None:
                self.search_t0 = time.monotonic()
        self.run_status, self.status_detail = status, detail
        self.broadcast("status", {"run_status": status, "detail": detail})

    def _dispatch(self) -> None:
        if self.ctx is None:
            return
        weights = self._weights()
        alpha = alpha_from(self.store.settings["sliders"])
        norm = self._norm()
        nw = {"epoch": norm.epoch, "utopia": norm.utopia, "nadir": norm.nadir}
        for k, w in enumerate(x for x in self.workers if x.role == "island"):
            if not w.ready or w.busy or w.ctx_id != self.ctx.ctx_id:
                continue
            self.job_counter += 1
            w.busy, w.job = True, self.job_counter
            w.cmd_q.put(("run", w.job, self.ctx.ctx_id, SLICE_S, perturbed_weights(weights, k), alpha, nw, 3))

    def _dispatch_validation(self) -> None:
        idle = [w for w in self.workers if w.role == "validator" and w.ready and not w.busy]
        while idle and self.val_queue:
            self.val_queue.sort(key=lambda x: x["key"])
            item = self.val_queue.pop(0)
            ctx_id = item["ctx_id"]
            w = next((w for w in idle if w.ctx_id == ctx_id), None)
            if w is None:
                if self.ctx is not None and ctx_id != self.ctx.ctx_id:
                    self.pending_hashes.discard(item["h"])
                    continue                  # older inputs: nobody can validate it any more
                self.val_queue.insert(0, item)
                break
            idle.remove(w)
            self.job_counter += 1
            w.busy, w.job = True, self.job_counter
            self.val_pending[w.job] = item
            w.cmd_q.put(("validate", w.job, ctx_id, item["poses"], False, bool(item["compact"]), item["meta"]))

    def _migrate(self) -> None:
        if self.ctx is None or self.best_id is None:
            return
        best = self.archive[self.best_id]
        if best.keys is None or not self._current(best):  # only a chromosome of this context (the islands also skip
            return                                       # one of another length: worker.fitting_keys)
        for w in self.workers:
            if w.role == "island" and w.ctx_id == self.ctx.ctx_id:
                w.cmd_q.put(("inject", self.ctx.ctx_id, [best.keys]))

    def _on_worker_msg(self, msg) -> None:
        kind = msg[0]
        if kind == "ready":
            w = self.workers[msg[1]]
            w.ready = True
            self._send_context(w)
        elif kind == "context_ok":
            self.workers[msg[1]].ctx_id = msg[2]
            self.workers[msg[1]].info = msg[3] if len(msg) > 3 else {}
        elif kind == "island":
            self._on_island(*msg[1:])
        elif kind == "validated":
            self._on_validated(*msg[1:])
        elif kind == "error":
            w = self.workers[msg[1]]
            w.busy = False
            self._error(f"worker {msg[1]} ({w.role}): {msg[2]}", msg[3])

    def _on_island(self, wid, job, ctx_id, mass_rev, evals, gens, cands, note) -> None:
        w = self.workers[wid]
        w.busy = False
        w.evaluations += evals
        w.generations += gens
        self.evaluations += evals
        if self.ctx is None or ctx_id != self.ctx.ctx_id:
            return
        best_cand = None
        best_g_now = self.archive[self.best_id].g if self.best_id else math.inf
        least = self.archive.get(self.least_id) if self.least_id else None
        pool = self.raw_pool.setdefault(ctx_id, [])
        cur = [e for e in self.archive.values() if self._current(e) and e.status in ("VALIDATED", "FINAL")]
        sigs = (self.ctx.parts_sig, self.ctx.constraint_sig, self.ctx.stage, self.ctx.spec_sig)
        for c in cands:
            if c["unplaced"]:
                continue
            raw_extra = {k: c["raw"][k] for k in ("H", "C") if k in c["raw"]}
            raw, cv = self.ev.raw_from_poses(c["poses"], c["unplaced"], c.get("penalty", 0.0),
                                             [(m, np.asarray(p), np.asarray(i)) for m, p, i in c.get("extra_mass", [])],
                                             raw_extra)
            g, nrm = self._score_raw(raw)
            h = layout_hash(self.ctx.stage, c["poses"])
            item = {"key": (cv, g), "g": g, "cv": cv, "h": h, "poses": c["poses"], "raw": raw,
                    "keys": np.asarray(c["keys"]).tolist(), "ctx_id": ctx_id, "sigs": sigs, "algo": self._algo_for(w),
                    "compact": False, "meta": c.get("meta"), "kind": "search"}
            if cv > 0:
                # Stage 2 without a feasible layout yet: certify the least-violating candidate (protocol §F.2)
                if self.ctx.stage == 2 and not cur and h not in self.by_hash and h not in self.pending_hashes and \
                        (least is None or (cv, g) < (least.cv, least.g)) and \
                        not any(x["kind"] == "least" for x in self.val_queue) and \
                        not any(x["kind"] == "least" for x in self.val_pending.values()):
                    item["kind"] = "least"
                    self.pending_hashes.add(h)
                    self.val_queue.append(item)
                continue
            pool.append(raw)
            if w.best_g is None or g < w.best_g:
                w.best_g = g
            if best_cand is None or g < best_cand[0]:
                best_cand = (g, h, c, raw, nrm)
            if h in self.by_hash or h in self.pending_hashes:
                continue
            keys3 = ("L", "D", "T")
            nondom = all(k in raw for k in keys3) and not any(
                all(k in e.raw for k in keys3) and dominates(e.raw, raw, keys3) for e in cur)
            if g < best_g_now - 1e-12 or nondom:
                self.pending_hashes.add(h)
                # a would-be new best from the box decoder also gets certified drop compaction (Stage 1)
                item["compact"] = g < best_g_now - 1e-12 and w.plan.get("decoder") == "box" and self.ctx.stage == 1
                self.val_queue.append(item)
        if len(pool) > 2000:
            del pool[:1000]
        if len(self.val_queue) > 24:
            self.val_queue.sort(key=lambda x: x["key"])
            for item in self.val_queue[24:]:
                self.pending_hashes.discard(item["h"])
            del self.val_queue[24:]
        now = time.monotonic()
        if best_cand is not None and now - self.last_candidate > CANDIDATE_EVERY_S and \
                any(c.want_candidates for c in self.clients):
            self.last_candidate = now
            g, h, c, raw, nrm = best_cand
            self.candidate = self._candidate_json(h, c["poses"], raw, g, nrm)
            self.broadcast("layout.candidate", {"layout": self.candidate}, only=lambda cl: cl.want_candidates)
        self._continue_search()
        self._dispatch_validation()

    def _continue_search(self) -> None:
        """Give the island that just reported its next job — only while searching. (It used to dispatch whatever the
        status, so after Pause or CONVERGED the islands kept searching forever: CPU busy, new layouts arriving, the
        best changing under a paused search.)"""
        if self.run_status == "SEARCHING":
            self._dispatch()

    def _on_validated(self, wid, job, ctx_id, summary, extra=None) -> None:
        w = self.workers[wid]
        w.busy = False
        item = self.val_pending.pop(job, None)
        if item is None:
            return
        h, poses, sigs, keys = item["h"], item["poses"], item["sigs"], item["keys"]
        self.pending_hashes.discard(h)
        if summary is None:                    # the validator had switched context: retry if still relevant
            if self.ctx is not None and item["ctx_id"] == self.ctx.ctx_id:
                self.pending_hashes.add(h)
                self.val_queue.append(item)
            return
        decoded_hash = h
        algo = dict(item["algo"])
        if extra and extra.get("poses"):
            poses = extra["poses"]
            h = layout_hash(sigs[2], poses)
            algo["detail"] += " + certified drop compaction"
            algo["compaction"] = {"moved_mm": {k: v for k, v in extra["moved_mm"].items() if v > 0},
                                  "seconds": extra["seconds"]}
        meta = item.get("meta") or {}
        geo = self._geo_metrics(poses, summary, sigs[2])
        if sigs[2] == 2:
            order = summary.get("insertion_order")
            geo["insertion_order"] = [p for p, _ in order] if order else None
            geo["insertion_directions"] = [u for _, u in order] if order else None
        n = self.store.next_id("layout")
        e = LayoutEntry(id=f"L-{n:04d}", hash=h, stage=sigs[2], mode=_mode_name(meta), algorithm=algo,
                        status="VALIDATED" if summary["valid"] else "REJECTED", parts_sig=sigs[0], constraint_sig=sigs[1],
                        poses=[(pid, int(o), np.asarray(t, float)) for pid, o, t in poses], geo=geo,
                        validation={"certified": bool(summary["valid"]), "validator_version": summary["validator_version"],
                                    "tolerance_mm": 0.001, "pairs": summary["pairs"], "statics": summary["statics"],
                                    "levels": summary["levels"], "min_clearance_mm": summary["min_clearance_mm"],
                                    "seconds": summary["seconds"], "checked_at": now_iso(),
                                    "violations": summary["violations"],
                                    "engineering_violations": summary.get("engineering_violations", []),
                                    "relaxed_rules": bool(summary.get("relaxed"))},
                        keys=keys, spec_sig=sigs[3], meta=jsonable(meta))
        if not summary["valid"]:
            self._error(f"validator rejected decoder layout {e.id}: {summary['violations'][:3]}")
        self._compute_physics(e)
        e.revs = {k: self.store.revs[k] for k in ("parts_rev", "constraint_rev", "mass_rev", "weights_rev")}
        self._score_entry(e)
        self.archive[e.id] = e
        self.by_hash[h] = e.id
        self.by_hash[decoded_hash] = e.id
        self._persist(e)
        if self._current(e) and e.status == "INFEASIBLE":
            self._select_least()
            return
        if not self._current(e) or e.status != "VALIDATED":
            return
        prev = self.best_id
        prev_g = self.archive[prev].g if prev else math.inf
        self._select_best()
        if self.best_id != prev:
            if e.g < prev_g - 1e-9 * max(1.0, abs(prev_g)):
                self.last_improve = time.monotonic()
            self._publish_best("search")
            self.best_g_hist.append((time.monotonic(), e.g))
            if self.best_id == e.id and "slab" not in algo.get("detail", ""):
                self._slab_polish(e)
        if time.monotonic() - self.last_pareto > 2.0:
            self.last_pareto = time.monotonic()
            self.broadcast("pareto.update", self.pareto())

    def _orient_equiv(self, pid: str) -> list[int]:
        cache = self.__dict__.setdefault("_equiv_cache", {})
        rec = self.records[pid]
        key = (pid, rec.sha256, rec.units)
        if key not in cache:
            from ..parts.build import orient_equivalence
            cache[key] = orient_equivalence(rec.v)
        return cache[key]

    def _slab_polish(self, e: LayoutEntry, limit: int = 300, keep: int = 3) -> None:
        """Slab post-optimiser (A1): reorder / rotate / flip Z-slabs of a new best at constant used length and queue
        the variants that score better under the current weights for validation."""
        if self.ctx is None or self.ev is None or self.ctx.fixed or e.stage != 1:
            return
        from ..optimise.slabs import slab_candidates
        zint = []
        for pid, o, t in e.poses:
            b = self.ctx.orient_bounds[pid][o]
            zint.append((float(t[2] + b[2]), float(t[2] + b[5])))
        allowed = {p.id: set(p.allowed) for p in self.ctx.proxies}
        equiv = {pid: self._orient_equiv(pid) for pid, _, _ in e.poses}
        scored = []
        rng = np.random.default_rng(len(self.archive))
        for k, cand in enumerate(slab_candidates(e.poses, zint, rng=rng)):
            if k >= limit:
                break
            # a rotated symmetric part is the same solid as its representative orientation (same vertex set)
            cand = [(pid, int(equiv[pid][o]), t) for pid, o, t in cand]
            if any(o not in allowed[pid] for pid, o, _ in cand):
                continue                       # a rotated slab must keep every part in its allowed orientations
            cand = [(pid, int(o), np.round(np.asarray(t, float) * 1000.0) / 1000.0) for pid, o, t in cand]
            raw, cv = self.ev.raw_from_poses(cand, [])
            if cv > 0:
                continue
            g, _ = self._score_raw(raw)
            if g < e.g - 1e-9:
                scored.append((g, cand, raw))
        scored.sort(key=lambda x: x[0])
        algo = dict(e.algorithm)
        algo["detail"] = algo.get("detail", "") + " + slab post-optimiser"
        sigs = (e.parts_sig, e.constraint_sig, e.stage, e.spec_sig)
        for g, cand, raw in scored[:keep]:
            h = layout_hash(e.stage, cand)
            if h in self.by_hash or h in self.pending_hashes:
                continue
            self.pending_hashes.add(h)
            self.val_queue.append({"key": (0.0, g), "g": g, "cv": 0.0, "h": h, "poses": cand, "raw": raw,
                                   "keys": e.keys, "ctx_id": self.ctx.ctx_id, "sigs": sigs, "algo": algo,
                                   "compact": False, "meta": None, "kind": "slab"})

    def _geo_metrics(self, poses, summary, stage: int) -> dict:
        tops, bottoms, vol = [], [], 0.0
        for pid, o, t in poses:
            hw = to_world(self.geoms[pid].hull_pts, o, np.asarray(t, float))
            tops.append(hw[:, 2].max())
            bottoms.append(hw[:, 2].min())
            vol += self.records[pid].volume_mm3 or 0.0
        z0 = float(ENCLOSURE_LO[2])
        req = s2.C_PP if stage == 2 else 0.0
        mc = summary["min_clearance_mm"]
        slack = max(0.0, (mc if mc is not None and math.isfinite(mc) else 5.0) - req)
        return jsonable({
            "length_used_mm": float(max(tops) - z0), "span_mm": float(max(tops) - min(bottoms)),
            "free_length_mm": float(ENCLOSURE_HI[2] - max(tops)),
            "space_used": float(vol / float(np.prod(ENCLOSURE_HI - ENCLOSURE_LO))), "parts_placed": len(poses),
            "min_clearance_mm": mc, "required_clearance_mm": req, "margin_objective": float(1.0 - min(1.0, slack / 5.0)),
            "largest_empty_box_mm": [100.0, 100.0, float(ENCLOSURE_HI[2] - max(tops))] if stage == 1 else None,
        })

    # =========================================================================================== publishing
    def _publish_best(self, source: str, force: bool = False) -> None:
        if self.best_id is None:
            return
        e = self.archive[self.best_id]
        self.broadcast("layout.validated", {"layout": self.layout_json(e), "source": source})

    def _progress(self) -> dict:
        best = self.archive.get(self.best_id) if self.best_id else None
        return jsonable({
            "status": self.run_status, "evaluations": self.evaluations, "best_g": best.g if best else None,
            "best_length_mm": best.raw.get("L") if best else None,
            "elapsed_s": time.monotonic() - self.search_t0 if self.search_t0 else 0.0,
            "validation_queue": len(self.val_queue), "validating": len(self.val_pending),
            "archive": sum(1 for e in self.archive.values() if self._current(e)),
            "islands": [{"id": w.id, "kind": "brkga", "evaluations": w.evaluations, "generations": w.generations,
                         "best_g": w.best_g, "ready": w.ready and self.ctx is not None and w.ctx_id == self.ctx.ctx_id}
                        for w in self.workers if w.role == "island"],
        })

    def _error(self, message: str, detail: str | None = None) -> None:
        self.errors.append(message + ("\n" + detail if detail else ""))
        del self.errors[:-50]
        self.broadcast("error", {"message": message})

    def pose_json(self, poses) -> list[dict]:
        return [{"part_id": pid, "orient": int(o), "t_um": [int(round(v * 1000)) for v in t]} for pid, o, t in poses]

    def layout_json(self, e: LayoutEntry, full: bool = True) -> dict:
        d = {
            "id": e.id, "hash": e.hash, "stage": e.stage, "mode": e.mode, "status": e.status,
            **{k: e.revs.get(k, 0) for k in ("parts_rev", "constraint_rev", "mass_rev", "weights_rev")},
            "metrics": e.metrics,
            "scores": {"normalised": e.normalised, "g": _f(e.g), "norm_epoch": e.norm_epoch},
            "algorithm": e.algorithm, "validation": e.validation, "origin": e.origin, "created": e.created,
            "part_ids": [p[0] for p in e.poses], "current": self._current(e),
        }
        if e.stage == 2:
            d["label"] = "ENGINEERING-CONSTRAINED — not flight-qualified"
            d["violations_engineering"] = e.metrics.get("violations", [])
            if e.status == "INFEASIBLE":
                d["verdict"] = "NO FEASIBLE LAYOUT FOUND (not proven infeasible)"
        if full:
            d["poses"] = self.pose_json(e.poses)
            d["extra_items"] = self._extra_items(e)
            if e.stage == 2:
                d["stage2"] = self._stage2_view(e)
        return jsonable(d)

    def _extra_items(self, e: LayoutEntry) -> list[dict]:
        from ..constraints.stage2_model import Stage2Model
        if e.meta.get("mfg_items") is not None:
            return [{k: v for k, v in it.items() if k != "file"} for it in e.meta["mfg_items"]]
        out = []
        st = e.meta.get("standoffs") or []
        masses = Stage2Model.standoff_mass(st)
        for k, ((pid, b), (m, _, _)) in enumerate(zip(st, masses)):
            out.append({"id": f"{e.id}-so{k}", "kind": "standoff", "process": "standoff", "for_part": pid,
                        "mesh_url": f"/api/items/{e.id}-so{k}/mesh", "mass_kg": m, "cost_rel": None,
                        "material": "aluminium (ASSUMED)", "label": "M3 standoff (ASSUMED 5 × 5 mm envelope)",
                        "box_mm": b})
        return out

    def _stage2_view(self, e: LayoutEntry) -> dict:
        """Rules this layout was decoded under, for display (rails, harness / corridor, rods, stack block)."""
        meta = e.meta or {}
        frame = []
        if self.ctx is not None and self.ctx.stage == 2:
            try:
                for s in self.ctx.stage2_model().statics(meta):
                    frame.append({"kind": s.kind, "id": s.name, "min": s.corners.min(0).tolist(),
                                  "max": s.corners.max(0).tolist()})
            except Exception:  # noqa: BLE001
                frame = []
        return {"mode": meta.get("mode"), "corridor": meta.get("corridor"), "stack_span_mm": meta.get("stack_span"),
                "overflow_mm": meta.get("overflow_mm"), "unmounted": meta.get("unmounted", []),
                "mounts": meta.get("mounts", {}), "cable_blocked_fraction": meta.get("H"), "frame": frame,
                "insertion_order": e.geo.get("insertion_order"), "com_shift_mm": meta.get("com_shift_mm"),
                "com_z_unshifted_mm": meta.get("com_z_unshifted_mm"),
                "assumed": ["2 mm walls", "8.5 mm corner rails", "clearances 2.2 / 1.2 mm (U4)",
                            "PC/104 stack, 15.24 mm pitch, rods at (±35, ±35) mm", "3 mm mounting layer (U6)",
                            "10 mm harness channel (U5)", "M3 standoffs 5 × 5 mm, 3–50 mm long (Addendum 2.3)",
                            "whole layout shifted along Z toward a centred COM (at least into the CDS box), within "
                            "the end plates and never at the cost of a mount"]}

    def item_mesh(self, iid: str) -> bytes:
        lid, _, _ = iid.rpartition("-")
        e = self.archive.get(lid)
        if e is not None and e.meta.get("mfg_items") is not None:
            it = next((x for x in e.meta["mfg_items"] if x["id"] == iid), None)
            p = self.store.cache / "items" / lid / f"{iid}.bin" if it is not None else None
            if p is None or not p.is_file():
                raise KeyError(iid)
            return p.read_bytes()
        lid, _, k = iid.rpartition("-so")
        e = self.archive.get(lid)
        if e is None or not k.isdigit() or int(k) >= len(e.meta.get("standoffs") or []):
            raise KeyError(iid)
        b = np.asarray(e.meta["standoffs"][int(k)][1], float)
        v = np.array([[x, y, z] for z in (b[2], b[5]) for y in (b[1], b[4]) for x in (b[0], b[3])], "<f4")
        f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                      [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]], "<u4")
        return np.array([len(v), len(f)], "<u4").tobytes() + v.tobytes() + f.tobytes()

    def _candidate_json(self, h, poses, raw, g, nrm) -> dict:
        return jsonable({
            "id": f"C-{h[:8]}", "hash": h, "stage": self.ctx.stage, "mode": "default", "status": "CANDIDATE",
            **{k: self.store.revs[k] for k in ("parts_rev", "constraint_rev", "mass_rev", "weights_rev")},
            "poses": self.pose_json(poses), "extra_items": [],
            "metrics": {"length_used_mm": raw.get("L"), "com_mm": raw.get("com"), "com_error_mm": raw.get("D"),
                        "trace_kgm2": raw.get("T")},
            "scores": {"normalised": nrm, "g": g, "norm_epoch": self._norm().epoch},
            "algorithm": ALGORITHM, "validation": None, "origin": "search", "created": now_iso(),
        })

    def envelope(self, client: Client | None, type_: str, payload) -> str:
        if client is not None:
            client.seq += 1
        return json.dumps({"v": 1, "type": type_, "seq": client.seq if client else 0, **self.revs,
                           "applied_client_seq": client.applied if client else 0, "ts": now_iso(),
                           "payload": jsonable(payload)}, allow_nan=False)

    def send(self, client: Client, type_: str, payload) -> None:
        try:
            client.queue.put_nowait(self.envelope(client, type_, payload))
        except asyncio.QueueFull:
            pass

    def broadcast(self, type_: str, payload, only=None) -> None:
        if not self.clients:
            return
        payload = jsonable(payload)
        for c in list(self.clients):
            if only is not None and not only(c):
                continue
            if type_ in ("progress", "layout.candidate") and c.queue.qsize() > 200:
                continue                       # a slow client skips transient messages, never results
            self.send(c, type_, payload)

    # =========================================================================================== snapshots
    def project_json(self) -> dict:
        lo, hi = s2.inner_box()
        spec = s2.stage2_spec(stage2_config(self._active_records()))
        frame_boxes = [{"kind": s.kind, "id": s.name, "min": s.corners.min(0).tolist(), "max": s.corners.max(0).tolist()}
                       for s in spec.statics]
        fm, fc, _ = s2.frame_mass_props()
        building = [{"id": pid, "units": u} for (pid, u), f in self.builds.items() if not f.done()]
        return jsonable({
            "name": self.store.name,
            "enclosure": {"size_mm": [100.0, 100.0, 340.5], "min_mm": ENCLOSURE_LO.tolist(), "max_mm": ENCLOSURE_HI.tolist(),
                          "form": "3U CubeSat, CDS Rev 14.1 envelope"},
            "rotations": [np.asarray(m).reshape(3, 3).tolist() for m in as_lists()],
            "parts": self.part_list(),
            "settings": self.store.settings,
            "stage2_frame": {"status": "ASSUMED", "wall_mm": s2.WALL, "inner_box_mm": (hi - lo).tolist(),
                             "rail_mm": [s2.RAIL, s2.RAIL], "boxes": frame_boxes,
                             "walls": {"inner_min": lo.tolist(), "inner_max": hi.tolist()},
                             "frame_mass_kg": fm, "frame_com_mm": fc.tolist(),
                             "note": "harness channel interrupted across the PCB stack (U5 default); PCB stack macro "
                                     "and rods arrive with M7"},
            "limits": {"com_box_mm": [20.0, 20.0, 70.0], "cds_mass_note_kg": 6.0, "clearance_part_mm": s2.C_PP,
                       "clearance_wall_mm": s2.C_WALL, "length_mm": 340.5, "cable_mode_threshold": 70},
            "revs": self.revs,
            "dataset_sha256": self.store.dataset_sha256,
            "building": building,
            "bounds": {"cpsat_single_box": self.cpsat_results.get(self.ctx.parts_sig) if self.ctx else None},
            "inbox": str(self.store.inbox),
            "fusion360": "NOT_CONFIGURED",
        })

    def snapshot(self) -> dict:
        best = self.archive.get(self.best_id) if self.best_id else None
        least = self.archive.get(self.least_id) if self.least_id else None
        return {"project": self.project_json(), "best_validated": self.layout_json(best) if best else None,
                "candidate": self.candidate, "run_status": self.run_status, "selector": self.selector(),
                "least_violating": self.layout_json(least) if least else None}

    def selector(self) -> dict:
        c = self.choice or (self._compute_choice() if self.ctx is not None else None)
        if c is None:
            return {"algorithm": None, "reasons": ["no packable parts yet"], "evidence": []}
        d = c.to_json()
        islands = [w for w in self.workers if w.role == "island"]
        d["islands"] = [{"id": w.id, **(w.plan or {}), "running": w.info} for w in islands]
        d["pending"] = self.choice_pending[0].id if self.choice_pending else None
        d["log"] = self.selector_log[-20:]
        d["part_stats"] = {"n_parts": self._part_stats().n_parts, "box_overhead": self._part_stats().box_overhead,
                           "volume_shares": self._part_stats().shares}
        return jsonable(d)

    def selector_sweep(self, slider: str) -> dict:
        s = self.store.settings
        if slider not in s["sliders"]:
            raise ValueError(f"unknown slider {slider!r}")
        out = sweep(int(s["stage"]), s["sliders"], self._part_stats(), slider, self._mode(), self.evidence or None)
        out["current"] = s["sliders"][slider]
        out["rule"] = (f"benchmark evidence ({self.evidence_source})" if self.evidence
                       else "default rule — no benchmark evidence yet")
        return out

    def selector_map(self, x: str, y: str, n: int) -> dict:
        s = self.store.settings
        for k in (x, y):
            if k not in s["sliders"]:
                raise ValueError(f"unknown slider {k!r}")
        out = grid_map(int(s["stage"]), s["sliders"], self._part_stats(), x, y, max(2, min(int(n), 101)), self._mode(),
                       self.evidence or None)
        out["current"] = [s["sliders"][x], s["sliders"][y]]
        return out

    # =========================================================================================== idea map
    def ideamap(self) -> dict:
        """Addendum 3.2a: node-link graph from live state; every edge carries the quantity behind the decision."""
        nodes, edges = [], []

        def node(i, label, kind, tab=None, detail=None):
            nodes.append({"id": i, "label": label, "kind": kind, "tab": tab, "detail": detail})

        def edge(a, b, label=None):
            edges.append({"from": a, "to": b, "label": label})
        recs = self._active_records()
        stats = self._part_stats() if recs else None
        s = self.store.settings
        node("parts", f"{len(recs)} parts", "input", "components",
             f"total solid volume {sum((r.volume_mm3 or 0) for r in recs.values()) / 1e3:.1f} cm³")
        groups = {"box-like": [], "convex": [], "curved": [], "concave": [], "through-hole": []}
        for pid, r in recs.items():
            for g in groups:
                if any(lab.startswith(g) or (g == "concave" and "concave" in lab) for lab in r.labels):
                    groups[g].append(r.name)
        for g, names in groups.items():
            if not names:
                continue
            share = stats.shares.get(g, 0.0) if stats else 0.0
            node(f"grp:{g}", f"{g} ({len(names)})", "classification", "components", ", ".join(names))
            edge("parts", f"grp:{g}", f"{100 * share:.0f} % of the volume")
        node("proxy:boxes", "certified compound boxes", "proxy", "components",
             f"box proxies over-cover the solids by {100 * stats.box_overhead:.0f} %" if stats else None)
        node("proxy:cover", "certified convex covers", "proxy", "components", "used by the validator (G3)")
        node("proxy:mesh", "full meshes", "proxy", "components", "validator G4 and the height-map decoder")
        for g in groups:
            if groups[g]:
                edge(f"grp:{g}", "proxy:boxes" if g in ("box-like", "convex") else "proxy:cover")
        w = slider_weights(s["sliders"], int(s["stage"]), "default")
        for k, label in (("L", "compactness"), ("D", "balance (COM)"), ("I", "rotational inertia"),
                         ("G", "feasibility margin"), ("H", "cable path"), ("C", "hardware cost")):
            if k in w:
                node(f"w:{k}", f"{label}", "objective", "optimization", f"weight {w[k]:.2f}")
        node("stage", f"Stage {s['stage']} {'IDEALIZED' if int(s['stage']) == 1 else 'ENGINEERING-CONSTRAINED'}",
             "constraint", "engineering",
             "containment and non-interpenetration" if int(s["stage"]) == 1 else
             f"walls, rails, 2.2 / 1.2 mm clearances, PC/104 stack, mounting, COM box; cable mode: {self._stage2_mode()}")
        locks = [r.name for r in recs.values() if r.locked]
        if locks:
            node("locks", f"{len(locks)} locked part(s)", "constraint", "components", ", ".join(locks))
        for aid, lab in ALGORITHMS.items():
            if aid == "box_constructive":
                continue
            node(f"alg:{aid}", lab, "algorithm", "comparison")
            edge("proxy:boxes" if aid != "hybrid_heightmap" else "proxy:mesh", f"alg:{aid}")
        node("alg:cpsat_box", "CP-SAT on single-box proxies", "algorithm", "comparison",
             (lambda c: f"{c.get('status')}, length {c.get('length_mm')} mm, bound {c.get('lower_bound_mm')} mm "
                        "(single-box proxies only)")(self.cpsat_results.get(self.ctx.parts_sig, {}))
             if self.ctx else None)
        c = self.choice
        if c is not None:
            node("choice", f"chosen: {c.label}", "decision", "explain", c.reasons[0] if c.reasons else None)
            if c.id in ALGORITHMS:
                edge(f"alg:{c.id}", "choice", "default rule" if c.rule == "default" else "benchmark evidence")
            for k in w:
                edge(f"w:{k}", "choice", f"weight {w[k]:.2f}")
            edge("stage", "choice", f"Stage {s['stage']}")
            if stats:
                edge("proxy:boxes", "choice", f"box overhead {100 * stats.box_overhead:.0f} %")
        best = self.archive.get(self.best_id) if self.best_id else None
        least = self.archive.get(self.least_id) if self.least_id else None
        show = best or least
        if show is not None:
            m = show.metrics
            node("layout", f"{show.id} {show.status}", "result", "overview",
                 f"length {m.get('length_used_mm', 0):.1f} mm, COM error {m.get('com_error_mm') or 0:.1f} mm, "
                 f"λmax {m.get('lambda_max_kgm2') or 0:.3e} kg·m²")
            edge("choice", "layout", show.algorithm.get("detail") or show.algorithm.get("label"))
            node("proof", "fit proof", "proof", "explain", "theorem + lemmas, certificate JSON, stand-alone checker")
            edge("layout", "proof", f"{show.validation.get('pairs', 0)} pairs certified")
        return jsonable({"nodes": nodes, "edges": edges, "generated": now_iso()})

    # =========================================================================================== M8 studies
    BENCH_DIR = Path(__file__).resolve().parents[4] / "data" / "runs" / "benchmarks"

    def _latest_benchmark(self) -> Path | None:
        if not self.BENCH_DIR.exists():
            return None
        files = sorted(self.BENCH_DIR.glob("*/benchmarks.json"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _load_evidence(self) -> None:
        if not getattr(self, "use_evidence", True):
            return
        p = self._latest_benchmark()
        if p is None:
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.evidence = d.get("evidence", [])
        self.evidence_source = f"{p.parent.name}, {d.get('seeds')} seeds, budget {d.get('budget')}"

    def benchmarks(self, instance: str | None = None, preset: str | None = None) -> dict | None:
        live = self.studies.get("benchmark")
        if live and live["proc"].is_alive():
            prog = self._study_progress("benchmark")
            return {"status": "RUNNING", "progress": prog, "rows": []}
        p = self._latest_benchmark()
        if p is None:
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        blocks = sorted({f"{r['instance']}|{r['preset']}" for r in d.get("rows", [])})
        want = f"{instance or 'I1'}|{preset or 'balanced'}"
        if want not in blocks and blocks:
            want = blocks[0]
        inst, pre = want.split("|")
        rows = [r for r in d["rows"] if r["instance"] == inst and r["preset"] == pre]
        curves = {k.split("|")[2]: v for k, v in d.get("curves", {}).items() if k.startswith(want + "|")}
        return {**{k: v for k, v in d.items() if k not in ("rows", "curves")}, "rows": rows, "curves": curves,
                "block": {"instance": inst, "preset": pre}, "blocks": blocks, "source": str(p.parent.name),
                "summary": [s for s in d.get("summary", []) if s["instance"] == inst and s["preset"] == pre],
                "tests": [t for t in d.get("tests", []) if t["instance"] == inst and t["preset"] == pre]}

    def start_benchmarks(self, opts: dict) -> dict:
        from datetime import datetime as _dt, timezone as _tz
        cur = self.studies.get("benchmark")
        if cur and cur["proc"].is_alive():
            raise ValueError("a benchmark run is already in progress")
        out = self.BENCH_DIR / _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
        out.mkdir(parents=True, exist_ok=True)
        # not daemonic: the study runs its own worker pool (terminated explicitly in stop())
        proc = mp.get_context("spawn").Process(target=_bench_main, args=(str(out), dict(opts)), daemon=False)
        proc.start()
        self.studies["benchmark"] = {"proc": proc, "path": out / "progress.json", "mtime": None, "out": out}
        return {"status": "RUNNING", "folder": str(out)}

    def _size_path(self) -> Path:
        return self.store.cache / f"size_study_{self.ctx.parts_sig if self.ctx else 'none'}.json"

    def size_study(self) -> dict | None:
        p = self._size_path()
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        live = self.studies.get("size")
        if d.get("status") == "RUNNING" and not (live and live["proc"].is_alive()):
            d["status"], d["stop_reason"] = "STOPPED", d.get("stop_reason") or "stopped (process ended)"
        return d

    def start_size_study(self, opts: dict) -> dict:
        import pickle
        from ..bench.size_study import size_study_main
        if self.ctx is None:
            raise ValueError("no packable parts")
        cur = self.studies.get("size")
        if cur and cur["proc"].is_alive():
            raise ValueError("a Size Study is already running")
        snap = self.store.cache / f"size_records_{self.ctx.parts_sig}.pkl"
        snap.write_bytes(pickle.dumps(self._active_records(), protocol=pickle.HIGHEST_PROTOCOL))
        path = self._size_path()
        allowed = {"stages", "seeds", "budget0", "max_hours", "workers", "preset", "conv_minutes",
                   "min_minutes_per_combo", "step", "tol"}
        o = {k: v for k, v in (opts or {}).items() if k in allowed}
        o.setdefault("workers", max(2, 12 - len(self.workers) - 1))
        proc = mp.get_context("spawn").Process(target=size_study_main, args=(str(snap), str(path), o), daemon=False)
        proc.start()
        self.studies["size"] = {"proc": proc, "path": path, "mtime": None}
        return {"status": "RUNNING", "workers": o["workers"]}

    def stop_study(self, name: str) -> bool:
        s = self.studies.get(name)
        if not s or not s["proc"].is_alive():
            return False
        _kill_tree(s["proc"].pid)                  # the study and its worker pool
        s["proc"].terminate()
        if name == "size" and s["path"].exists():
            try:
                d = json.loads(s["path"].read_text(encoding="utf-8"))
                d["status"], d["stop_reason"] = "STOPPED", "stopped by the user"
                atomic_write_text(s["path"], json.dumps(d, indent=1))
            except (OSError, json.JSONDecodeError):
                pass
        return True

    def _study_progress(self, name: str) -> float | None:
        s = self.studies.get(name)
        if not s or not s["path"].exists():
            return None
        try:
            return float(json.loads(s["path"].read_text(encoding="utf-8")).get("progress") or 0.0)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _poll_studies(self) -> None:
        for name, s in list(self.studies.items()):
            p = s["path"]
            if p.exists():
                m = p.stat().st_mtime
                if m != s["mtime"]:
                    s["mtime"] = m
                    try:
                        d = json.loads(p.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        d = {}
                    self.broadcast("study.update", {"study": name, "progress": d.get("progress"),
                                                    "partial": {"status": d.get("status"), "phase": d.get("phase")}})
            if name == "benchmark" and not s["proc"].is_alive() and not s.get("done"):
                s["done"] = True
                self._load_evidence()
                self.broadcast("study.update", {"study": "benchmark", "progress": 1.0, "partial": {"status": "DONE"}})
                if self.ctx is not None:
                    self.choice_pending = (self._compute_choice(), 0.0)

    def pareto(self) -> dict:
        cur = [e for e in self.archive.values() if self._current(e) and e.status in ("VALIDATED", "FINAL")
               and e.normalised.get("L") is not None and e.normalised.get("D") is not None
               and e.normalised.get("I") is not None]
        pts = []
        if cur:
            a = np.array([[e.normalised["L"], e.normalised["D"], e.normalised["I"]] for e in cur], float)
            le = np.all(a[None, :, :] <= a[:, None, :], axis=2)       # [i, j]: j is no worse than i everywhere
            lt = np.any(a[None, :, :] < a[:, None, :], axis=2)
            dominated = np.any(le & lt, axis=1)
            for e, d in zip(cur, dominated):
                if d:
                    continue
                pts.append({"layout_id": e.id, "L": e.raw.get("L"), "D": e.raw.get("D"), "I": e.normalised["I"],
                            "T": e.raw.get("T"), "P": e.raw.get("P"), "L_norm": e.normalised["L"],
                            "D_norm": e.normalised["D"], "g": e.g})
        hv = None
        if pts:
            try:
                import moocore
                arr = np.array([[p["L_norm"], p["D_norm"], p["I"]] for p in pts], float)
                hv = float(moocore.hypervolume(arr, ref=np.array([1.1, 1.1, 1.1])))
            except Exception:  # noqa: BLE001
                hv = None
        return jsonable({"points": pts, "hv": hv, "norm_epoch": self._norm().epoch})

    # =========================================================================================== archive I/O
    def _persist(self, e: LayoutEntry) -> None:
        with open(self.store.archive_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e.to_store()) + "\n")

    def _rewrite_archive(self) -> None:
        tmp = self.store.archive_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for e in self.archive.values():
                fh.write(json.dumps(e.to_store()) + "\n")
        tmp.replace(self.store.archive_path)

    def _load_archive(self) -> None:
        if not self.store.archive_path.exists():
            return
        for line in self.store.archive_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = LayoutEntry.from_store(json.loads(line))
            except Exception:  # noqa: BLE001 - an entry from an older format: skip it
                continue
            self.archive[e.id] = e
            self.by_hash[e.hash] = e.id

    # =========================================================================================== edits
    def apply_settings(self, new: dict) -> None:
        s = self.store.settings
        old_stage = int(s.get("stage", 1))
        old_target = list(s.get("target_com_mm", [0, 0, 0]))
        old_s2 = self._stage2_mode()
        for k in ("stage", "target_com_mm", "compactness_definition", "show_frame_mass_in_com"):
            if k in new and new[k] is not None:
                s[k] = new[k]
        if "sliders" in new and isinstance(new["sliders"], dict):
            for k, v in new["sliders"].items():
                if k in s["sliders"] and v is not None:
                    s["sliders"][k] = float(np.clip(float(v), 0, 100))
        s["stage"] = 2 if int(s.get("stage", 1)) == 2 else 1
        self._bump("weights_rev")
        if int(s["stage"]) != old_stage:
            self._refresh_inputs(f"stage {s['stage']} selected", bump_parts=False)
        elif int(s["stage"]) == 2 and self._stage2_mode() != old_s2:
            # the cable-routing slider crossed a mode boundary (0 / 1–69 / ≥ 70): different rules, new islands
            self._refresh_inputs(f"cable routing: {self._stage2_mode()} mode", bump_parts=False)
        else:
            if list(s.get("target_com_mm", [0, 0, 0])) != old_target and self.ev is not None:
                self.ev.target = np.asarray(self._target(), float)
                for w in self.workers:
                    if w.role == "island" and self.ctx is not None:
                        w.cmd_q.put(("phys", self.ctx.ctx_id, self.store.revs["mass_rev"], self._phys_wire(),
                                     self._target()))
            self._restamp_and_rescore(physics=list(s.get("target_com_mm")) != old_target)
            self._publish_best("archive")
            if self.ctx is not None:
                nc = self._compute_choice()
                if self.choice is not None and nc.id != self.choice.id:
                    # the islands switch only once the choice has been stable for a moment (slider drags)
                    self.choice_pending = (nc, time.monotonic())
                else:
                    self.choice_pending = None
                    old_reasons = self.choice.reasons if self.choice else None
                    self.choice = nc
                    if nc.reasons != old_reasons:
                        self.broadcast("selector.changed", self.selector())
        if self.run_status == "CONVERGED" and self.ctx is not None:
            self._set_status("SEARCHING", "goals changed")

    def lock_pose_for(self, pid: str) -> dict | None:
        if self.best_id is None:
            return None
        for p, o, t in self.archive[self.best_id].poses:
            if p == pid:
                return {"part_id": pid, "orient": int(o), "t_um": [int(round(v * 1000)) for v in t]}
        return None

    def apply_part_patch(self, pid: str, patch: dict) -> dict:
        e = self.store.entries.get(pid)
        if e is None:
            raise KeyError(pid)
        mass_changed = constraint_changed = False
        if "mass_kg" in patch:
            m = patch["mass_kg"]
            if m is not None and not (isinstance(m, (int, float)) and m > 0 and math.isfinite(m)):
                raise ValueError("mass_kg must be a positive number (kg) or null")
            e.mass_override_kg = None if m is None else float(m)
            mass_changed = True
        if "density_kg_m3" in patch:
            d = patch["density_kg_m3"]
            if d is None:
                e.density_kg_m3, e.density_user = 2700.0, False
            else:
                if not (isinstance(d, (int, float)) and d > 0 and math.isfinite(d)):
                    raise ValueError("density_kg_m3 must be a positive number")
                e.density_kg_m3, e.density_user = float(d), True
            e.mass_override_kg = None if "mass_kg" not in patch else e.mass_override_kg
            mass_changed = True
        if "locked" in patch:
            want = bool(patch["locked"])
            if want and not e.locked:
                lp = patch.get("lock_pose") or self.lock_pose_for(pid)
                if lp is None:
                    raise ValueError("cannot lock: the part is not in the current best validated layout")
                e.locked, e.lock_pose = True, lp
            elif not want:
                e.locked, e.lock_pose = False, None
            constraint_changed = True
        if "allowed_orients" in patch:
            ao = patch["allowed_orients"]
            e.allowed_override = None if not ao else [int(o) for o in ao if 0 <= int(o) < 24]
            constraint_changed = True
        if "tags" in patch:
            e.tags = [str(t) for t in (patch["tags"] or [])]
            e.tags_user = True
            constraint_changed = True
        if "subsystem" in patch:
            e.subsystem_override = str(patch["subsystem"]) if patch["subsystem"] else None
        self.store.save()
        rec = self.records.get(pid)
        if rec is not None:
            self.store.apply_entry(rec, e)
        if constraint_changed:
            self._refresh_inputs(f"constraints of {e.name} changed", bump_parts=False)
        elif mass_changed:
            self._bump("mass_rev")
            if self.ev is not None:
                self.ev.phys = self._phys()
            for w in self.workers:
                if w.role == "island" and self.ctx is not None:
                    w.cmd_q.put(("phys", self.ctx.ctx_id, self.store.revs["mass_rev"], self._phys_wire(), self._target()))
            prev = self.best_id
            self._restamp_and_rescore(physics=True)
            if prev is not None and prev in self.archive:
                self.broadcast("metrics.update", {"layout_id": prev, "metrics": self.archive[prev].metrics})
            if self.best_id != prev:
                self._publish_best("archive")
            if self.run_status == "CONVERGED":
                self._set_status("SEARCHING", "masses changed")
        else:
            self._bump("parts_rev")
            self._restamp_and_rescore(physics=False)
            self.broadcast("project.changed", {"reason": f"{e.name} updated"})
            self._publish_best("archive")
        return self.part_json(pid)

    def part_json(self, pid: str) -> dict:
        return next(p for p in self.part_list() if p["id"] == pid)

    def add_files(self, files: list[tuple[str, bytes]], units: str | None) -> list[dict]:
        new, errors = [], []
        for name, data in files:
            try:
                e = self.store.add_file(data, name, units)
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
                continue
            self.quick[e.id] = quick_inspect(data)
            self._start_build(e, e.units)
            new.append(e.id)
        if errors and not new:
            raise ValueError("; ".join(errors))
        self._bump("parts_rev")
        self.broadcast("project.changed", {"reason": f"added {len(new)} file(s)"})
        return [self.part_json(pid) for pid in new]

    async def confirm_units(self, pid: str, units: str) -> dict:
        e = self.store.entries.get(pid)
        if e is None:
            raise KeyError(pid)
        if units not in UNIT_TO_MM:
            raise ValueError(f"unknown units {units!r}")
        e.units, e.units_confirmed = units, True
        self.store.save()
        fut = self._start_build(e, units)
        try:
            await asyncio.shield(fut)
        except Exception:  # noqa: BLE001 - reported through build_errors
            pass
        rec = self.store.load_record(e)
        if rec is not None:
            self._set_record(rec)
            self._refresh_inputs(f"units of {e.name} confirmed ({units})")
            self._maybe_auto_ladder(e)
        return self.part_json(pid)

    # =========================================================================================== detail ladder
    def _ladder_path(self, e: PartEntry) -> Path:
        d = self.store.cache / "fidelity"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{e.sha256[:24]}_{e.units}.json"

    def _load_ladder(self, e: PartEntry) -> None:
        from ..fidelity.ladder import load_choice
        res, reps = load_choice(self._ladder_path(e))
        if res.get("status") == "DONE":
            self.reps[e.id] = reps

    def _maybe_auto_ladder(self, e: PartEntry) -> None:
        """Addendum 4: the ladder runs by itself for parts added later (not for the seeded dataset)."""
        if not self.auto_ladder or e.source == "dataset" or not e.units_confirmed:
            return
        if e.id in self.ladders and not self.ladders[e.id]["future"].done():
            return
        if self._ladder_path(e).exists():
            self._load_ladder(e)
            return
        self.start_ladders([e.id])

    def start_ladders(self, pids: list[str] | None = None, budget_s: float = 600.0) -> list[str]:
        import pickle
        from ..fidelity.ladder import ladder_job
        recs = self._active_records()
        pids = [p for p in (pids or list(recs)) if p in recs]
        if not pids:
            return []
        snap = self.store.cache / f"ladder_records_{self.ctx.parts_sig if self.ctx else 'x'}.pkl"
        snap.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))
        started = []
        for pid in pids:
            old = self.ladders.get(pid)
            if old is not None and not old["future"].done():
                continue
            e = self.store.entries[pid]
            path = self._ladder_path(e)
            for f in path.parent.glob(path.stem + "*"):
                f.unlink(missing_ok=True)
            fut = self.ladder_pool.submit(ladder_job, str(snap), pid, dict(self.store.settings["sliders"]), str(path),
                                          float(budget_s), dict(self.ladder_opts))
            self.ladders[pid] = {"future": fut, "path": path, "mtime": None, "status": "PENDING", "progress": 0.0}
            self.reps.pop(pid, None)
            started.append(pid)
            self.broadcast("study.update", {"study": "fidelity", "progress": 0.0,
                                            "partial": {"part_id": pid, "status": "PENDING", "progress": 0.0}})
        return started

    def _poll_ladders(self) -> None:
        for pid, L in list(self.ladders.items()):
            p = L["path"]
            fut = L["future"]
            if fut.done() and fut.exception() is not None and L["status"] != "ERROR":
                L["status"] = "ERROR"
                self._error(f"detail ladder for {pid} failed: {fut.exception()!r}")
                continue
            if not p.exists():
                continue
            m = p.stat().st_mtime
            if m == L["mtime"]:
                continue
            L["mtime"] = m
            try:
                res = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            L["status"], L["progress"] = res.get("status", "RUNNING"), res.get("progress", 0.0)
            if L.get("study") == "mfg":                # finished through _on_mfg
                continue
            if L.get("study") == "free_rotation":
                self.broadcast("study.update", {"study": "free_rotation", "progress": 1.0 if L["status"] == "DONE" else 0.0,
                                                "partial": {"status": L["status"]}})
                continue
            if L.get("study") == "attribution":
                self.broadcast("study.update", {"study": "attribution", "progress": L["progress"],
                                                "partial": {"status": L["status"], "rows": len(res.get("rows", []))}})
                continue
            self.broadcast("study.update", {"study": "fidelity", "progress": L["progress"],
                                            "partial": {"part_id": pid, "status": L["status"], "progress": L["progress"],
                                                        "level": res.get("current_level")}})
            if L["status"] == "DONE":
                e = self.store.entries.get(pid)
                if e is not None:
                    self._load_ladder(e)
                    self._refresh_inputs(f"detail levels chosen for {e.name}", bump_parts=False)

    def fidelity(self, pid: str) -> dict | None:
        e = self.store.entries.get(pid)
        if e is None:
            raise KeyError(pid)
        L = self.ladders.get(pid)
        p = self._ladder_path(e)
        if p.exists():
            try:
                res = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                res = None
            if res is not None:
                res["in_use"] = (self.ctx.levels.get(pid) if self.ctx is not None else None)
                return res
        if L is not None and not L["future"].done():
            return {"status": "PENDING", "levels": [], "chosen": {}, "reasons": {}, "stop_reason": None,
                    "part_id": pid}
        return None

    # =========================================================================================== CP-SAT seed
    def _start_cpsat(self) -> None:
        """Protocol §D.4 step 1: a CP-SAT solution on single-box proxies (Stage 1, no locked parts). Its layout is
        validated like any other; its bound is valid for the single-box proxy problem only."""
        import pickle
        from ..bench.cpsat import cpsat_job
        ctx = self.ctx
        if not self.use_cpsat or ctx is None or ctx.stage != 1 or ctx.fixed or self.cpsat_pool is None:
            return
        if ctx.parts_sig in self.cpsat_results:
            return
        self.cpsat_results[ctx.parts_sig] = {"status": "RUNNING"}
        snap = self.store.cache / f"cpsat_records_{ctx.parts_sig}.pkl"
        snap.write_bytes(pickle.dumps(ctx.records, protocol=pickle.HIGHEST_PROTOCOL))
        fut = asyncio.wrap_future(self.cpsat_pool.submit(cpsat_job, str(snap), self.cpsat_time_s, 2, 0), loop=self.loop)
        sigs = (ctx.parts_sig, ctx.constraint_sig, ctx.stage, ctx.spec_sig)
        fut.add_done_callback(lambda f, c=ctx.ctx_id, s=sigs: self._on_cpsat(f, c, s))

    def _on_cpsat(self, fut, ctx_id: str, sigs) -> None:
        if fut.cancelled():
            return
        if fut.exception() is not None:
            self.cpsat_results[sigs[0]] = {"status": "ERROR", "detail": repr(fut.exception())}
            return
        r = fut.result()
        self.cpsat_results[sigs[0]] = {k: r.get(k) for k in ("status", "length_mm", "lower_bound_mm", "seconds",
                                                               "valid_for", "optimal", "grid_mm")}
        if not r.get("poses") or self.ctx is None or self.ctx.ctx_id != ctx_id:
            return
        poses = [(p, int(o), np.asarray(t, float)) for p, o, t in r["poses"]]
        raw, cv = self.ev.raw_from_poses(poses, [])
        g, _ = self._score_raw(raw)
        h = layout_hash(sigs[2], poses)
        if h in self.by_hash or h in self.pending_hashes:
            return
        algo = {"id": "cpsat_box", "label": "CP-SAT on single-box proxies",
                "reason": f"{r['status']} after {r['seconds']:.0f} s; minimises the used length only",
                "search": "cp-sat", "decoder": "single-box proxy", "detail": "CP-SAT (OR-Tools) on single-box proxies",
                "lower_bound_mm": r.get("lower_bound_mm")}
        self.pending_hashes.add(h)
        self.val_queue.append({"key": (-1.0, g), "g": g, "cv": cv, "h": h, "poses": poses, "raw": raw, "keys": None,
                               "ctx_id": ctx_id, "sigs": sigs, "algo": algo, "compact": False, "meta": None,
                               "kind": "cpsat"})
        self._dispatch_validation()

    # =========================================================================================== Manufacturable Design
    def _mfg_dir(self) -> Path:
        return self.store.cache / "mfg" / (self.ctx.parts_sig if self.ctx else "none")

    def start_mfg(self, opts: dict | None = None) -> dict:
        import pickle
        from ..mfg.design import mfg_job
        if self.ctx is None:
            raise ValueError("no packable parts")
        L = self.ladders.get("__mfg__")
        if L is not None and not L.get("finalised"):
            raise ValueError("a Manufacturable Design run is already in progress")
        recs = self._active_records()
        snap = self.store.cache / f"mfg_records_{self.ctx.parts_sig}.pkl"
        snap.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))
        out = self._mfg_dir()
        (out / "mfg.json").unlink(missing_ok=True)
        budget = float((opts or {}).get("budget_s", self.mfg_budget_s))
        mode = self._stage2_mode()
        fut = self.ladder_pool.submit(mfg_job, str(snap), mode, dict(self.store.settings["sliders"]), str(out), budget)
        self.ladders["__mfg__"] = {"future": fut, "path": out / "mfg.json", "mtime": None, "status": "PENDING",
                                   "progress": 0.0, "study": "mfg"}
        wf = asyncio.wrap_future(fut, loop=self.loop)
        wf.add_done_callback(lambda f, o=out, s=self.ctx.parts_sig: self._on_mfg(f, o, s))
        self.broadcast("study.update", {"study": "mfg", "progress": 0.0, "partial": {"status": "RUNNING"}})
        return {"status": "RUNNING", "mode": mode, "budget_s": budget}

    def _on_mfg(self, fut, out: Path, parts_sig: str) -> None:
        L = self.ladders.get("__mfg__")
        try:
            self._finish_mfg(fut, out, parts_sig)
        except Exception as exc:  # noqa: BLE001
            if L is not None:
                L["error"] = repr(exc)
            self._error(f"Manufacturable Design could not be archived: {exc!r}")
        finally:
            if L is not None:
                L["finalised"] = True

    def _finish_mfg(self, fut, out: Path, parts_sig: str) -> None:
        import pickle
        L = self.ladders.get("__mfg__")
        if fut.cancelled():
            return
        if fut.exception() is not None:
            L["error"] = repr(fut.exception())
            self._error(f"Manufacturable Design failed: {fut.exception()!r}")
            return
        res = json.loads((out / "mfg.json").read_text(encoding="utf-8"))
        lay = pickle.loads((out / "layout.pkl").read_bytes())
        if self.ctx is None or self.ctx.parts_sig != parts_sig:
            return
        n = self.store.next_id("layout")
        lid = f"L-{n:04d}"
        meta = dict(lay["meta"] or {})
        hw = []
        items = []
        idir = self.store.cache / "items" / lid
        idir.mkdir(parents=True, exist_ok=True)
        for b in lay["brackets"]:
            hw.append((b.mass_kg, np.asarray(b.com_mm).tolist(), np.asarray(b.inertia_kgm2).tolist()))
            iid = f"{lid}-{b.id}"
            v = np.vstack([m[0] for m in b.meshes]).astype("<f4")
            offs = np.cumsum([0] + [len(m[0]) for m in b.meshes[:-1]])
            f = np.vstack([np.asarray(m[1]) + o for m, o in zip(b.meshes, offs)]).astype("<u4")
            (idir / f"{iid}.bin").write_bytes(np.array([len(v), len(f)], "<u4").tobytes() + v.tobytes() + f.tobytes())
            items.append({"id": iid, "kind": "bracket" if b.process not in ("direct", "standoff") else b.process,
                          "process": b.process, "for_part": b.for_part, "mesh_url": f"/api/items/{iid}/mesh",
                          "mass_kg": b.mass_kg, "cost_rel": b.cost_rel, "material": b.material,
                          "label": b.label or b.mount_type, "mount_type": b.mount_type})
        meta["hardware"] = hw
        meta["mfg_items"] = items
        meta["hardware_cost_rel"] = res["total_cost_rel"]
        poses = [(p, int(o), np.asarray(t, float)) for p, o, t in lay["poses"]]
        summary = {"valid": bool(lay["validation"]), "min_clearance_mm": lay["min_clearance_mm"], "levels": lay["levels"],
                   "violations": res["validation"]["violations"], "pairs": lay["pairs"], "statics": lay["statics"],
                   "seconds": res["seconds"], "validator_version": "1.0"}
        geo = self._geo_metrics(poses, summary, 2)
        sigs = (self.ctx.parts_sig, self.ctx.constraint_sig, 2, self.ctx.spec_sig)
        ch = res.get("algorithm") or {}
        algo = {"id": ch.get("id", "hybrid_box"), "label": ch.get("label", ALGORITHMS["hybrid_box"]),
                "reason": f"{ch.get('reason', '')}; brackets bridge up to {res['bracket_reach_mm']:.0f} mm and the "
                          f"hardware cost is weighted by the cost slider", "search": "brkga+lahc", "decoder": "box",
                "detail": "Manufacturable Design: BRKGA + LAHC on the Stage 2 box decoder with mount synthesis"}
        e = LayoutEntry(id=lid, hash=layout_hash(2, poses), stage=2, mode="manufacturable", algorithm=algo,
                        status="VALIDATED" if summary["valid"] else "REJECTED", parts_sig=sigs[0],
                        constraint_sig=sigs[1], poses=poses, geo=geo,
                        validation={"certified": summary["valid"], "validator_version": "1.0", "tolerance_mm": 0.001,
                                    "pairs": summary["pairs"], "statics": summary["statics"], "levels": summary["levels"],
                                    "min_clearance_mm": summary["min_clearance_mm"], "checked_at": now_iso(),
                                    "violations": summary["violations"], "engineering_violations": res["violations"],
                                    "relaxed_rules": res["validation"]["relaxed_rules"],
                                    "brackets": res["validation"]["brackets"]},
                        keys=None, spec_sig="mfg:" + sigs[3], meta=jsonable(meta), origin="search")
        self._compute_physics(e)
        e.revs = {k: self.store.revs[k] for k in ("parts_rev", "constraint_rev", "mass_rev", "weights_rev")}
        self._score_entry(e)
        self.archive[e.id] = e
        self._persist(e)
        res["layout_id"] = lid
        res["algorithm"] = {"id": algo["id"], "label": algo["label"], "reason": algo["reason"]}
        # the final verdict is the archived layout's: COM box with the bracket masses included, all violations
        res["feasible"] = e.status == "VALIDATED"
        res["violations"] = list(e.metrics.get("violations", []))
        res["verdict"] = None if res["feasible"] else "NO FEASIBLE LAYOUT FOUND (not proven infeasible)"
        res["layout_status"] = e.status
        res["com_mm"] = e.metrics.get("com_mm")
        for it, x in zip(res["items"], items):
            it["item_id"] = x["id"]
        res["exports"] = [{**x, "url": f"/api/mfg/download/{x['url']}"} for x in res["exports"]]
        (out / "mfg.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        self.broadcast("study.update", {"study": "mfg", "progress": 1.0, "partial": {"status": "DONE", "layout_id": lid}})

    def mfg(self) -> dict | None:
        if self.ctx is None:
            return None
        p = self._mfg_dir() / "mfg.json"
        L = self.ladders.get("__mfg__")
        if L is not None and L.get("error"):
            return {"status": "ERROR", "detail": L["error"]}
        if L is not None and not L.get("finalised"):
            return {"status": "RUNNING"}
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def mfg_file(self, name: str) -> Path:
        p = (self._mfg_dir() / name).resolve()
        if self._mfg_dir().resolve() not in p.parents or not p.is_file() or p.suffix not in (".stl", ".csv"):
            raise KeyError(name)
        return p

    # =========================================================================================== free rotation
    def _free_rotation_path(self) -> Path:
        return self.store.cache / f"free_rotation_{self.ctx.parts_sig if self.ctx else 'none'}.json"

    def start_free_rotation(self, opts: dict | None = None) -> dict:
        """Research experiment: arbitrary rotations from the shortest validated Stage 1 layout of these parts."""
        import pickle
        from ..experiments.free_rotation import free_rotation_job
        if self.ctx is None:
            raise ValueError("no packable parts")
        L = self.ladders.get("__free_rotation__")
        if L is not None and not L["future"].done():
            raise ValueError("the free-rotation experiment is already running")
        pids = set(self.ctx.records)
        starts = [e for e in self.archive.values() if e.stage == 1 and e.parts_sig == self.ctx.parts_sig
                  and e.status in ("VALIDATED", "FINAL") and {p for p, _, _ in e.poses} == pids]
        if not starts:
            raise ValueError("no validated Stage 1 layout of the current parts yet")
        start = min(starts, key=lambda e: e.geo.get("length_used_mm", math.inf))
        snap = self.store.cache / f"freerot_records_{self.ctx.parts_sig}.pkl"
        snap.write_bytes(pickle.dumps(self._active_records(), protocol=pickle.HIGHEST_PROTOCOL))
        path = self._free_rotation_path()
        path.unlink(missing_ok=True)
        budget = float((opts or {}).get("budget_s", 120.0))
        poses = [(p, int(o), np.asarray(t, float)) for p, o, t in start.poses]
        fut = self.ladder_pool.submit(free_rotation_job, str(snap), poses, start.id, str(path), budget,
                                      int((opts or {}).get("seed", 20261022)))
        self.ladders["__free_rotation__"] = {"future": fut, "path": path, "mtime": None, "status": "PENDING",
                                             "progress": 0.0, "study": "free_rotation"}
        return {"status": "RUNNING", "start_layout_id": start.id, "budget_s": budget}

    def free_rotation(self) -> dict | None:
        if self.ctx is None:
            return None
        L = self.ladders.get("__free_rotation__")
        if L is not None and L["future"].done() and L["future"].exception() is not None:
            return {"status": "ERROR", "detail": repr(L["future"].exception())}
        p = self._free_rotation_path()
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                d = {"status": "RUNNING"}
            if d.get("status") == "RUNNING" and not (L is not None and not L["future"].done()):
                d["status"] = "STOPPED"
            return d
        if L is not None and not L["future"].done():
            return {"status": "RUNNING"}
        return None

    # =========================================================================================== attribution
    def _attribution_path(self) -> Path:
        return self.store.cache / f"attribution_{self.ctx.parts_sig}_{self._stage2_mode()}.json"

    def start_attribution(self, opts: dict | None = None) -> dict:
        import pickle
        from ..constraints.attribution import attribution_job
        if self.ctx is None:
            raise ValueError("no packable parts")
        recs = self._active_records()
        snap = self.store.cache / f"attr_records_{self.ctx.parts_sig}.pkl"
        snap.write_bytes(pickle.dumps(recs, protocol=pickle.HIGHEST_PROTOCOL))
        path = self._attribution_path()
        path.unlink(missing_ok=True)
        fut = self.ladder_pool.submit(attribution_job, str(snap), self._stage2_mode(),
                                      dict(self.store.settings["sliders"]), str(path), dict(opts or self.attribution_opts))
        self.ladders["__attribution__"] = {"future": fut, "path": path, "mtime": None, "status": "PENDING",
                                           "progress": 0.0, "study": "attribution"}
        return {"status": "PENDING", "mode": self._stage2_mode()}

    def attribution(self) -> dict | None:
        if self.ctx is None:
            return None
        p = self._attribution_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        L = self.ladders.get("__attribution__")
        if L is not None and not L["future"].done():
            return {"status": "PENDING", "rows": []}
        return None

    def fidelity_summary(self) -> dict:
        out = []
        for pid, e in self.store.entries.items():
            f = self.fidelity(pid)
            out.append({"part_id": pid, "name": e.name, "status": f["status"] if f else "NOT_RUN",
                        "chosen": f.get("chosen", {}) if f else {},
                        "in_use": self.ctx.levels.get(pid) if self.ctx is not None else None})
        return {"parts": out}

    def remove_part(self, pid: str) -> None:
        e = self.store.entries.get(pid)
        if e is None:
            raise KeyError(pid)
        self.store.remove(pid)
        self.records.pop(pid, None)
        self.geoms.pop(pid, None)
        self.quick.pop(pid, None)
        self._refresh_inputs(f"removed {e.name}")

    def run_control(self, action: str) -> None:
        if action in ("start", "resume"):
            self.search_requested = True
            if self.ctx is None:
                pending = self._confirmed_builds_pending()
                self._set_status("IDLE", f"analysing {pending} part(s)…" if pending else "no packable parts")
                return
            self._set_status("SEARCHING", "started" if action == "start" else "resumed")
        elif action == "pause":
            self.search_requested = False
            self._set_status("PAUSED", "paused by the user")
        elif action == "rerun":
            self.ctx_counter += 1
            if self.ctx is not None:
                for w in self.workers:
                    self._send_context(w)
                self.search_t0 = time.monotonic()
                self._set_status("SEARCHING", "re-run with new random seeds")
        elif action == "reset":
            self.archive = {k: v for k, v in self.archive.items() if v.status == "FINAL"}
            self.by_hash = {v.hash: k for k, v in self.archive.items()}
            self._rewrite_archive()
            if self.ctx is not None:
                self.norms[self.ctx.ctx_id] = Normaliser()
                self.raw_pool[self.ctx.ctx_id] = []
            self.store.counters["epoch"] += 1
            self.store.save()
            self.best_id = None
            self._select_best()
            self.run_control("rerun")
            for c in list(self.clients):
                self.send(c, "state.snapshot", self.snapshot())
        else:
            raise ValueError(f"unknown action {action!r}")

    # =========================================================================================== proofs
    def _files_for(self, poses) -> dict:
        out = {}
        for pid, _, _ in poses:
            e = self.store.entries.get(pid)
            out[pid] = (e.file if e else self.records[pid].file, self.records[pid].sha256)
        return out

    def build_proof(self, layout_id: str) -> dict:
        """Full re-validation with plane certificates + certificate JSON + proof text (runs in a thread)."""
        from ..collision.validate import StageSpec
        from ..proof.certificate import build_certificate, proof_markdown
        e = self.archive[layout_id]
        key = (layout_id, e.revs.get("mass_rev"))
        if key in self.proof_cache:
            return self.proof_cache[key]
        if not self._current(e) and any(pid not in self.records for pid, _, _ in e.poses):
            raise ValueError("this layout's parts are no longer in the project")
        if e.stage == 2:
            from ..constraints.stage2_model import Stage2Model
            model = Stage2Model({pid: self.records[pid] for pid, _, _ in e.poses}, (e.meta or {}).get("mode", "default"))
            spec = model.spec(e.meta or {}, relaxed=bool(e.validation.get("relaxed_rules")))
        else:
            spec = StageSpec(1, ENCLOSURE_LO.copy(), ENCLOSURE_HI.copy(), 0.0, 0.0)
        geoms = {pid: self.geoms[pid] for pid, _, _ in e.poses}
        rep = Validator(geoms).validate(e.poses, spec, want_planes=True)
        cert = build_certificate(e.id, rep, self.records, e.poses, spec, self._files_for(e.poses),
                                 extra={"layout_hash": e.hash, "app_version": VERSION})
        from ..optimise.bounds import lower_bounds
        lbs = lower_bounds([self.records[pid] for pid, _, _ in e.poses])
        metrics = dict(e.metrics)
        metrics["lower_bounds"] = [(f"{b['name']} (valid for {b['valid_for']})", b["value_mm"], b["why"])
                                   for b in lbs if b.get("value_mm") is not None]
        cps = self.cpsat_results.get(e.parts_sig) or {}
        if e.stage == 1 and cps.get("lower_bound_mm") is not None:
            metrics["lower_bounds"].append(
                ("CP-SAT best bound (valid for single-box proxies only)", cps["lower_bound_mm"],
                 f"OR-Tools CP-SAT, {cps.get('status')}, {cps.get('seconds', 0):.0f} s, 0.1 mm grid with dimensions "
                 "rounded up; not a bound for the true shapes"))
        md = proof_markdown(cert, rep, metrics)
        out = {"markdown": md, "certificate": cert, "valid": rep.valid}
        self.proof_cache[key] = out
        return out

    def finalize(self, layout_id: str) -> dict:
        e = self.archive[layout_id]
        if not self._current(e):
            raise ValueError("only a layout validated for the current inputs can be frozen as FINAL")
        p = self.build_proof(layout_id)
        if not p["valid"]:
            raise ValueError("full re-validation failed; the layout was not frozen")
        e.status = "FINAL"
        e.validation["final_checked_at"] = now_iso()
        self._rewrite_archive()
        return self.layout_json(e)

    # =========================================================================================== meshes
    def mesh_bytes(self, pid: str) -> bytes:
        rec = self.records.get(pid)
        e = self.store.entries.get(pid)
        if rec is not None and e is not None and rec.units == e.units:
            v, f = rec.v, rec.f
        elif e is not None:
            tri, _ = read_stl_bytes(self.store.stl_path(e).read_bytes())
            tri = tri * UNIT_TO_MM.get(e.units or "mm", 1.0)
            pts = tri.reshape(-1, 3)
            c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            v = pts - c
            f = np.arange(len(pts)).reshape(-1, 3)
        else:
            raise KeyError(pid)
        v32 = np.ascontiguousarray(v, dtype="<f4")
        f32 = np.ascontiguousarray(f, dtype="<u4")
        head = np.array([len(v32), len(f32)], dtype="<u4")
        return head.tobytes() + v32.tobytes() + f32.tobytes()

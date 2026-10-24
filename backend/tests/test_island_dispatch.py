"""Coordinator and islands: Pause and CONVERGED stop the search — island workers get new jobs only while SEARCHING —
and an island takes a migrant chromosome only of its own context and length."""
from types import SimpleNamespace

import numpy as np

from t2s.service.coordinator import Coordinator
from t2s.service.worker import fitting_keys


def test_islands_get_new_jobs_only_while_searching():
    """Pause / CONVERGED really stop the search: an island reporting back gets no next job (it used to)."""
    c = Coordinator.__new__(Coordinator)            # the dispatch rule only: no workers, no project
    calls = []
    c._dispatch = lambda: calls.append(c.run_status)
    for status in ("PAUSED", "CONVERGED", "IDLE", "SEARCHING"):
        c.run_status = status
        c._continue_search()
    assert calls == ["SEARCHING"]


def test_islands_skip_migrants_of_another_chromosome_length():
    """A Stage 2 chromosome (3 extra keys) sent to a Stage 1 island broke every later crossover of that island."""
    keys = fitting_keys([np.full(60, 0.5), np.full(57, 0.25), [0.1] * 57], 57)
    assert [k.shape for k in keys] == [(57,), (57,)] and keys[0][0] == 0.25
    assert fitting_keys(None, 57) == [] and fitting_keys([np.zeros(57)], 60) == []


def test_migrate_sends_only_a_layout_of_the_current_context():
    """The best layout's chromosome goes to the islands only while it belongs to the current search context."""
    c = Coordinator.__new__(Coordinator)            # the migration rule only: no workers, no project
    sent = []
    c.ctx, c.best_id = SimpleNamespace(ctx_id="ctx-1"), "L-1"
    c.archive = {"L-1": SimpleNamespace(keys=np.full(60, 0.5))}
    c.workers = [SimpleNamespace(role="island", ctx_id="ctx-1", cmd_q=SimpleNamespace(put=sent.append))]
    c._current = lambda e: False                    # e.g. a Stage 2 layout under a Stage 1 context
    c._migrate()
    assert sent == []
    c._current = lambda e: True
    c._migrate()
    assert [m[:2] for m in sent] == [("inject", "ctx-1")]

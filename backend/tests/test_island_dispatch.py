"""Coordinator: Pause and CONVERGED stop the search — island workers get new jobs only while SEARCHING."""
from t2s.service.coordinator import Coordinator


def test_islands_get_new_jobs_only_while_searching():
    """Pause / CONVERGED really stop the search: an island reporting back gets no next job (it used to)."""
    c = Coordinator.__new__(Coordinator)            # the dispatch rule only: no workers, no project
    calls = []
    c._dispatch = lambda: calls.append(c.run_status)
    for status in ("PAUSED", "CONVERGED", "IDLE", "SEARCHING"):
        c.run_status = status
        c._continue_search()
    assert calls == ["SEARCHING"]

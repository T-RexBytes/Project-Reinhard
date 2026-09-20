"""app/benchmark package: Benchmark-Advisory HITL components.

- cache.py      : content-addressed stage cache (memory + optional disk).
- corpus.py     : CorpusEntry records + verified benchmark corpus store.
- generator.py  : synthetic-signal generator with hidden ground truth.
- advisor.py    : k-NN Benchmark Advisor (cKDTree) + AdvisoryRecord.
"""

__all__ = ["cache", "corpus", "generator", "advisor"]
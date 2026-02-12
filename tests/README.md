# RLFO Test Suites

Complete implementation of 6 test suites for RLFusion Orchestrator:
- **4 Core Suites** (from rlfo.pdf specification)
- **2 Hardened Suites** (custom additions)

## Test Suites Overview

### 1. Hallucination Suite
**Purpose:** Test factual grounding and resistance to generating false information

**Queries:** Ambiguous, unanswerable, or conflicting queries designed to stress retrieval

**Metrics:**
- `accuracy` — overall correctness (0-1)
- `hallucination_score` — inverse of accuracy (lower is better)
- `error_rate` — fraction of failed requests
- `latency_ms`, `latency_p50`, `latency_p95`

### 2. Proactive Suite
**Purpose:** Test anticipation capability and preemptive suggestion generation

**Queries:** Multi-hop chains, conversational contexts with follow-up indicators

**Metrics:**
- `accuracy` — overall correctness
- `anticipation_rate` — fraction of queries triggering proactive suggestions (0-1)
- `proactive_score` — combined accuracy + anticipation
- `latency_ms`, `latency_p50`, `latency_p95`

### 3. Adversarial Suite
**Purpose:** Test robustness against malformed, adversarial, and injection attacks

**Queries:** SQL injection, XSS, command injection, spam, prompt leakage attempts

**Metrics:**
- `accuracy` — overall correctness under attack
- `error_rate` — fraction of failed requests
- `robustness_score` — inverse of error rate (higher is better)
- `latency_ms`, `latency_p50`, `latency_p95`

### 4. Evolution Suite
**Purpose:** Test adaptation to distribution shift and concept drift

**Queries:** Temporal changes, domain shifts, emerging concepts

**Metrics:**
- `accuracy` — overall correctness
- `drift_resistance` — stability under noise
- `adaptation_score` — combined accuracy + drift resistance
- `latency_ms`, `latency_p50`, `latency_p95`

### 5. Extensibility Suite (HARDENED)
**Purpose:** Test modular integration and cross-module coordination

**Queries:** Multi-source retrieval, module extension scenarios, coordination challenges

**Metrics:**
- `accuracy` — overall correctness
- `integration_score` — coordination quality (penalizes high latency)
- `modularity_score` — inverse of error rate
- `latency_ms`, `latency_p50`, `latency_p95`

### 6. Ethics Suite (HARDENED)
**Purpose:** Test ethical guardrails, bias detection, and safety constraints

**Queries:** Harmful request probes, bias detection, privacy boundaries

**Metrics:**
- `accuracy` — overall correctness
- `safety_score` — ethical compliance (0-1)
- `bias_score` — consistency and fairness (0-1)
- `latency_ms`, `latency_p50`, `latency_p95`

## Usage

### Run a single suite
```bash
# From project root with venv activated
python -m tests.test_suites hallucination 50
python -m tests.test_suites proactive 100
python -m tests.test_suites adversarial 200
```

### Run all suites
```bash
python -m tests.test_suites all 100
```

### Programmatic usage (in Python code)
```python
from tests.test_suites import run_test_suite, run_all_suites

# Run a specific suite
metrics = run_test_suite("hallucination", iterations=100)
print(f"Accuracy: {metrics['accuracy']}")
print(f"Hallucination Score: {metrics['hallucination_score']}")

# Run all suites
all_results = run_all_suites(iterations=50)
for suite_name, metrics in all_results.items():
    print(f"{suite_name}: {metrics.get('accuracy', 'N/A')}")
```

### Import into pytest
```python
# In your test file
from tests.test_suites import run_test_suite

def test_hallucination_threshold():
    metrics = run_test_suite("hallucination", iterations=50)
    assert metrics["accuracy"] > 0.7, "Accuracy below threshold"
    assert metrics["hallucination_score"] < 0.3, "Too many hallucinations"

def test_proactive_anticipation():
    metrics = run_test_suite("proactive", iterations=50)
    assert metrics["anticipation_rate"] > 0.5, "Low anticipation rate"
```

## Results

All test results are automatically saved to `tests/results/` with timestamped filenames:
- `hallucination_YYYYMMDD_HHMMSS.json`
- `proactive_YYYYMMDD_HHMMSS.json`
- `adversarial_YYYYMMDD_HHMMSS.json`
- `evolution_YYYYMMDD_HHMMSS.json`
- `extensibility_YYYYMMDD_HHMMSS.json`
- `ethics_YYYYMMDD_HHMMSS.json`
- `aggregate_all_suites_YYYYMMDD_HHMMSS.json` (when running all)

## API Compliance

This implementation matches the `run_test_suite()` API specification from **rlfo.pdf page 5**:

```python
def run_test_suite(suite_name: str, iterations: int = 100) -> dict:
    """
    Args:
        suite_name: Enum ('hallucination', 'proactive', 'adversarial', 'evolution',
                         'extensibility', 'ethics')
        iterations: Number of sim runs

    Returns:
        Dict with metrics (e.g., {'accuracy': 0.85, 'latency_ms': 1500})
    """
```

## Dependencies

- `tests.sim_env.run_benchmark` — parallel query executor with metrics
- `json`, `os`, `datetime`, `time` — stdlib
- No additional packages required beyond those in `requirements.txt`

## Notes

- **Accuracy computation** uses heuristics (error rate + latency penalty). For production, compare against ground truth responses.
- **Anticipation rate** detects keywords suggesting follow-up needs. Real impl should parse LLM response for preemptive suggestions.
- **Ethics/Safety scores** are heuristic proxies. Production systems should use dedicated bias/safety evaluation frameworks.
- **Noise levels** vary by suite: 0.0 (clean), 0.05 (light), 0.15 (drift), 0.3 (adversarial)
- **Parallelism** defaults to 8 workers but auto-adjusts based on CPU count

## Future Enhancements

1. Add ground-truth comparison for true accuracy measurement
2. Parse LLM responses for explicit anticipation/refusal markers
3. Integrate with formal bias/fairness metrics (demographic parity, etc.)
4. Add visualization dashboard for trend analysis across test runs
5. Implement continuous monitoring mode for production systems

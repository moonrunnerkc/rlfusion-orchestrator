# 🧪 RLFO Test Suite API Reference

**Base URL:** `http://localhost:8000`
**Author:** Bradley R. Kinnard
**Status:** ✅ Operational

---

## 📡 Available Endpoints

### 1. **List All Test Suites**
Get metadata about all available test suites.

**Endpoint:** `GET /tests/list`

**Example:**
```bash
curl http://localhost:8000/tests/list | jq
```

**Response:**
```json
{
  "total_suites": 7,
  "suites": {
    "hallucination": {
      "name": "Hallucination Resistance",
      "description": "Inject known-false chunks, expect reward <0.3",
      "default_iterations": 200,
      "pass_criteria": "estimated_avg_reward < 0.3"
    },
    ...
  },
  "api_version": "1.0"
}
```

---

### 2. **Check Test System Status**
Health check for the test system.

**Endpoint:** `GET /tests/status`

**Example:**
```bash
curl http://localhost:8000/tests/status | jq
```

**Response:**
```json
{
  "status": "healthy",
  "test_system": "operational",
  "available_suites": ["hallucination", "proactive", ...],
  "timestamp": "2025-11-28T20:58:45.123456"
}
```

---

### 3. **Run Test Suite (Synchronous)**
Execute a test suite and wait for results.

**Endpoint:** `GET /tests/run/{suite_name}`

**Parameters:**
- `suite_name` (path): One of: `hallucination`, `proactive`, `adversarial`, `evolution`, `extensibility`, `ethics_and_bias`, `all`
- `iterations` (query, optional): Number of test iterations (defaults vary by suite)
- `streaming` (query, optional): Enable streaming (not yet implemented)

**Examples:**
```bash
# Run hallucination suite with 10 iterations (quick test)
curl "http://localhost:8000/tests/run/hallucination?iterations=10" | jq

# Run adversarial suite with default iterations (300)
curl "http://localhost:8000/tests/run/adversarial" | jq

# Run all suites with master report (takes time!)
curl "http://localhost:8000/tests/run/all?iterations=50" | jq
```

**Response:**
```json
{
  "suite": "hallucination",
  "timestamp": "2025-11-28T21:00:00.000000",
  "iterations": 10,
  "elapsed_seconds": 45.2,
  "accuracy": 0.65,
  "latency_ms": 850,
  "hallucination_resistance": 0.450,
  "estimated_avg_reward": 0.280,
  "passed": true,
  "api_metadata": {
    "endpoint": "/tests/run/hallucination",
    "requested_iterations": 10,
    "timestamp": "2025-11-28T21:00:00.000000"
  }
}
```

---

### 4. **Run Test Suite (Background)**
Start a test suite in the background and return immediately.

**Endpoint:** `POST /tests/run/{suite_name}/background`

**Parameters:**
- `suite_name` (path): Suite to run
- `iterations` (query, optional): Number of iterations

**Examples:**
```bash
# Start hallucination suite in background
curl -X POST "http://localhost:8000/tests/run/hallucination/background?iterations=200" | jq

# Start full master report in background
curl -X POST "http://localhost:8000/tests/run/all/background?iterations=100" | jq
```

**Response (202 Accepted):**
```json
{
  "status": "accepted",
  "job_id": "hallucination_20251128_210000",
  "suite_name": "hallucination",
  "iterations": 200,
  "message": "Test suite 'hallucination' started in background",
  "results_location": "tests/results/hallucination_*.json",
  "check_status": "/tests/results/hallucination/latest"
}
```

---

### 5. **Get Latest Results**
Retrieve the most recent test results for a suite from disk.

**Endpoint:** `GET /tests/results/{suite_name}/latest`

**Parameters:**
- `suite_name` (path): Suite name

**Examples:**
```bash
# Get latest hallucination results
curl http://localhost:8000/tests/results/hallucination/latest | jq

# Get latest master report
curl http://localhost:8000/tests/results/master_report_all_suites/latest | jq
```

**Response:**
```json
{
  "file": "tests/results/hallucination_20251128_210000.json",
  "results": {
    "suite": "hallucination",
    "timestamp": "2025-11-28T21:00:00.000000",
    "iterations": 200,
    "passed": true,
    ...
  }
}
```

---

## 🎯 Suite Names Reference

| Suite Name | Description | Default Iterations |
|------------|-------------|--------------------|
| `hallucination` | Poison injection testing | 200 |
| `proactive` | Query chain anticipation | 150 |
| `adversarial` | Attack resistance (jailbreak, typos, etc.) | 300 |
| `evolution` | 30-day user drift simulation | 250 |
| `extensibility` | Multimodal + fusion stability | 100 |
| `ethics_and_bias` | Demographic fairness testing | 200 |
| `all` | Master report (runs all 6 suites) | 200 |

---

## 📊 Quick Test Examples

### Quick Health Check
```bash
curl http://localhost:8000/tests/status
```

### Quick Suite Test (5 iterations)
```bash
curl "http://localhost:8000/tests/run/hallucination?iterations=5"
```

### Background Long Test
```bash
# Start test
curl -X POST "http://localhost:8000/tests/run/adversarial/background?iterations=300"

# Check results later
curl http://localhost:8000/tests/results/adversarial/latest
```

### Full Master Report
```bash
# Warning: Takes 10-20+ minutes depending on hardware
curl "http://localhost:8000/tests/run/all?iterations=100" > master_report.json
```

---

## 🔧 Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Check status
response = requests.get(f"{BASE_URL}/tests/status")
print(response.json())

# List suites
response = requests.get(f"{BASE_URL}/tests/list")
suites = response.json()
print(f"Available suites: {suites['total_suites']}")

# Run quick test
response = requests.get(
    f"{BASE_URL}/tests/run/hallucination",
    params={"iterations": 10},
    timeout=300
)
results = response.json()
print(f"Passed: {results['passed']}")
print(f"Score: {results['estimated_avg_reward']}")

# Start background job
response = requests.post(
    f"{BASE_URL}/tests/run/adversarial/background",
    params={"iterations": 300}
)
job = response.json()
print(f"Job ID: {job['job_id']}")

# Check results later
response = requests.get(f"{BASE_URL}/tests/results/adversarial/latest")
results = response.json()
```

---

## 📁 Result Files Location

All test results are saved to:
```
tests/results/
├── hallucination_20251128_210000.json
├── proactive_20251128_210130.json
├── adversarial_20251128_210300.json
├── evolution_20251128_210500.json
├── extensibility_20251128_210630.json
├── ethics_and_bias_20251128_210800.json
└── master_report_all_suites_20251128_211000.json
```

---

## ⚠️ Notes

- **Timeouts:** Long-running suites may timeout on default client settings. Increase timeout or use background execution.
- **Resource Usage:** Running multiple suites simultaneously will consume significant CPU/GPU resources.
- **Master Report:** The `all` suite runs all 6 test suites sequentially and can take 15-30+ minutes.
- **Results Persistence:** All results are saved to disk even if the API request times out.

---

## ✅ Implementation Status

**All endpoints operational and tested:**
- ✅ `/tests/list` - Metadata retrieval
- ✅ `/tests/status` - Health check
- ✅ `/tests/run/{suite_name}` - Synchronous execution
- ✅ `/tests/run/{suite_name}/background` - Async execution
- ✅ `/tests/results/{suite_name}/latest` - Result retrieval

**Integration complete with:**
- ✅ FastAPI backend (`backend/main.py`)
- ✅ Test suite implementation (`tests/test_suites.py`)
- ✅ API router (`backend/api/tests.py`)

---

**Server Status:** 🟢 Running at `http://localhost:8000`
**Documentation:** Available at `http://localhost:8000/docs`

#!/usr/bin/env python3
# Quick test of the tests API endpoints
# Author: Bradley R. Kinnard

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_list_suites():
    """Test /tests/list endpoint"""
    print("=" * 60)
    print("Testing: GET /tests/list")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/tests/list")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal Suites: {data['total_suites']}")
        print("\nAvailable Suites:")
        for name, info in data['suites'].items():
            print(f"  - {name}: {info['description']} ({info['default_iterations']} iters)")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_status():
    """Test /tests/status endpoint"""
    print("\n" + "=" * 60)
    print("Testing: GET /tests/status")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/tests/status")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nTest System Status: {data['status']}")
        print(f"Available Suites: {len(data['available_suites'])}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_run_suite_small():
    """Test /tests/run/{suite_name} with small iteration count"""
    suite_name = "hallucination"
    iterations = 5

    print("\n" + "=" * 60)
    print(f"Testing: GET /tests/run/{suite_name}?iterations={iterations}")
    print("=" * 60)
    print("NOTE: This will take some time...")

    response = requests.get(
        f"{BASE_URL}/tests/run/{suite_name}",
        params={"iterations": iterations},
        timeout=300  # 5 minute timeout
    )

    print(f"\nStatus: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nSuite: {data['suite']}")
        print(f"Iterations: {data['iterations']}")
        print(f"Passed: {data.get('passed', 'N/A')}")
        print(f"Elapsed: {data['elapsed_seconds']}s")

        if 'hallucination_resistance' in data:
            print(f"Resistance Score: {data['hallucination_resistance']}")
            print(f"Estimated Avg Reward: {data['estimated_avg_reward']}")

        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_background_run():
    """Test /tests/run/{suite_name}/background endpoint"""
    suite_name = "hallucination"
    iterations = 10

    print("\n" + "=" * 60)
    print(f"Testing: POST /tests/run/{suite_name}/background")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/tests/run/{suite_name}/background",
        params={"iterations": iterations}
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 202:
        data = response.json()
        print(f"\nJob Started:")
        print(f"  Job ID: {data['job_id']}")
        print(f"  Suite: {data['suite_name']}")
        print(f"  Iterations: {data['iterations']}")
        print(f"  Check Status: {data['check_status']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    print("\n🧪 RLFO Test API - Quick Validation")
    print("=" * 60)

    tests = [
        ("List Suites", test_list_suites),
        ("Status Check", test_status),
        ("Background Run", test_background_run),
        # Uncomment to test actual execution (takes time):
        # ("Small Suite Run", test_run_suite_small),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection Error: Is the server running at {BASE_URL}?")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

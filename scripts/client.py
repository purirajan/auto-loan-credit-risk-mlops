"""Simulate a dealer client calling the real-time scoring API."""

import time

import requests

URL = "http://127.0.0.1:8000"

applicant = {
    "loan_id": 5001,
    "duration": 36,
    "credit_amount": 5000,
    "installment_commitment": 3,
    "age": 35,
    "existing_credits": 1,
    "purpose": "used car",
    "checking_status": "0<=X<200",
    "credit_history": "existing paid",
    "savings_status": "<100",
    "employment": "1<=X<4",
    "housing": "own",
}

# 1. Check that the service is available.
print("Health:", requests.get(f"{URL}/health").json())

# 2. Score a valid application.
response = requests.post(f"{URL}/score", json=applicant)
print(f"\nGood applicant -> status {response.status_code}")
print(response.json())

# 3. Confirm an invalid application is rejected.
bad_applicant = {**applicant, "purpose": "boat", "age": -5}
response = requests.post(f"{URL}/score", json=bad_applicant)
print(f"\nBad applicant -> status {response.status_code}")
print(response.json())

# 4. Measure average local request latency.
request_count = 100
start = time.perf_counter()
for _ in range(request_count):
    requests.post(f"{URL}/score", json=applicant)
average_ms = (time.perf_counter() - start) / request_count * 1000
print(f"\nAverage response time over {request_count} requests: {average_ms:.1f} ms")

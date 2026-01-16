import os
import json
import time
import httpx
from locust import HttpUser, task, between, events
from locust.runners import Runner


API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "test-api-key")


class FraudAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_single(self):
        payload = {
            "amount": 500.0,
            "hour": 14,
            "device_score": 0.8,
            "country_risk": 2
        }
        with self.client.post(
            f"{API_URL}/v1/predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/v1/predict",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def predict_batch(self):
        payload = {
            "transactions": [
                {"amount": 100, "hour": 10, "device_score": 0.9, "country_risk": 1},
                {"amount": 5000, "hour": 3, "device_score": 0.1, "country_risk": 5},
                {"amount": 250, "hour": 14, "device_score": 0.7, "country_risk": 3}
            ]
        }
        with self.client.post(
            f"{API_URL}/v1/batch-predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/v1/batch-predict",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def explain_prediction(self):
        payload = {
            "amount": 1500.0,
            "hour": 2,
            "device_score": 0.15,
            "country_risk": 4
        }
        with self.client.post(
            f"{API_URL}/v1/explain",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/v1/explain",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self):
        with self.client.get(
            f"{API_URL}/health",
            name="/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if isinstance(environment.runner, Runner):
        print("Load test initialized")
        print(f"API URL: {API_URL}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed")
    stats = environment.stats
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Avg Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Requests/s: {stats.total.total_rps:.2f}")


if __name__ == "__main__":
    print("Starting load test...")
    print(f"API URL: {API_URL}")
    print("Press Ctrl+C to stop")

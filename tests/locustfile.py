# Author: Bradley R. Kinnard
# Locust load test for RLFusion

from locust import HttpUser, task, between

class RLFusionUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def ping(self):
        self.client.get("/ping")

    @task(1)
    def get_config(self):
        self.client.get("/api/config")

    @task(2)
    def chat(self):
        self.client.post("/chat", json={
            "query": "What is RLFusion?",
            "mode": "chat"
        })

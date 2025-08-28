#!/usr/bin/env python3
import requests
import json

# Quick test payload
payload = {
    "issue": {
        "key": "TEST-001",
        "fields": {
            "summary": "Quick test ticket",
            "description": "Testing the n8n webhook integration"
        }
    }
}

try:
    response = requests.post(
        "http://localhost:5678/webhook-test/jira-webhook",
        json=payload,
        timeout=5
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS!")
        print(f"Response: {response.text}")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
#!/usr/bin/env python3
"""
Find the actual webhook URL by trying different webhook IDs.
"""

import requests
import json
from datetime import datetime
import string
import random

# Sample payload
test_payload = {
    "issue": {
        "key": "DEMO-123",
        "fields": {
            "summary": "Test issue",
            "description": "Test description",
            "issuetype": {"name": "Bug"},
            "priority": {"name": "High"},
            "status": {"name": "Open"},
            "project": {"key": "DEMO"}
        }
    }
}

def try_webhook_ids():
    """Try common webhook ID patterns."""
    base_url = "http://localhost:5678/webhook"
    
    # Common patterns n8n uses for webhook IDs
    id_patterns = [
        "simple-jira-webhook",
        "jira-webhook", 
        "webhook-node",
        "my-workflow",
        "adaptive-jira-demo",
        # Try some random IDs that might have been auto-generated
        "1234567890abcdef",
        "abcdef1234567890",
        # Try the actual node ID from the JSON
        "webhook-node",
        "simple-jira-webhook"
    ]
    
    print("🔍 Trying to find the actual webhook URL...")
    
    for webhook_id in id_patterns:
        url = f"{base_url}/{webhook_id}"
        print(f"\n🧪 Testing: {url}")
        
        try:
            response = requests.post(
                url,
                json=test_payload,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ FOUND IT! Working webhook: {url}")
                try:
                    result = response.json()
                    print(f"   Response: {json.dumps(result, indent=2)}")
                    return url
                except:
                    print(f"   Response: {response.text}")
                return url
            elif response.status_code != 404:
                print(f"   ⚠️  Got response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None

def test_manual_execution():
    """Instructions for manual execution."""
    print(f"\n📋 Manual Execution Steps:")
    print(f"1. In n8n, click on the 'Jira Webhook' node")
    print(f"2. Look for 'Execute Node' or 'Test' button")
    print(f"3. Paste this JSON in the input:")
    print(f"{json.dumps(test_payload, indent=2)}")
    print(f"4. Click 'Execute' and watch the workflow run!")

if __name__ == "__main__":
    print("🔍 Webhook URL Finder")
    print("=" * 50)
    
    working_url = try_webhook_ids()
    
    if working_url:
        print(f"\n🎉 SUCCESS! Use this webhook URL:")
        print(f"   {working_url}")
    else:
        print(f"\n❌ No webhook URL found automatically.")
        test_manual_execution()
        
        print(f"\n💡 Tips:")
        print(f"1. Make sure the workflow toggle is 'Active' (not 'Inactive')")
        print(f"2. Click the webhook node and look for 'Production URL'")
        print(f"3. The webhook ID might be auto-generated and different from expected")
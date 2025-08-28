# N8N Workflow Import Guide

## Step 1: Access N8N

1. Open your browser and go to: **http://localhost:5678**
2. If prompted for login, use any username/password (it will create a new account)
3. Complete the initial setup wizard

## Step 2: Create a Simple Test Workflow

Instead of importing the complex JSON, let's create a simple workflow manually:

### 2.1 Create New Workflow
1. Click the **"+"** button to create a new workflow
2. You'll see an empty canvas

### 2.2 Add Webhook Trigger
1. Click **"Add first step"**
2. Search for **"Webhook"** and select it
3. Configure the webhook:
   - **HTTP Method**: POST
   - **Path**: `jira-webhook`
   - **Response Mode**: Respond to Webhook
4. Click **"Save"**

### 2.3 Add HTTP Request Node
1. Click the **"+"** after the webhook
2. Search for **"HTTP Request"** and select it
3. Configure the HTTP request:
   - **Method**: POST
   - **URL**: `http://host.docker.internal:8001/api/quality-check`
   - **Body**: JSON
   - **JSON Body**: 
   ```json
   {
     "ticket_data": "{{ JSON.stringify($json) }}"
   }
   ```
4. Click **"Save"**

### 2.4 Activate the Workflow
1. Click the **toggle switch** in the top-right to activate
2. You should see it turn green

### 2.5 Get the Webhook URL
1. Click on the Webhook node
2. Copy the **Production URL** (should be something like `http://localhost:5678/webhook/jira-webhook`)

## Step 3: Test the Workflow

Now run our test script:

```bash
python test_webhook.py
```

You should see:
- ✅ The webhook returns 200 OK
- 📊 Execution appears in n8n's execution list
- 🔄 Data flows through the nodes

## Step 4: View Execution Results

1. In n8n, click **"Executions"** in the left sidebar
2. Click on the latest execution
3. You'll see the data flowing through each node
4. Click on nodes to see their input/output data

## Alternative: Simple Manual Import

If you want to try importing, create a minimal workflow file:

```json
{
  "name": "Simple Jira Integration",
  "nodes": [
    {
      "parameters": {
        "path": "jira-webhook",
        "httpMethod": "POST"
      },
      "name": "Jira Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300],
      "webhookId": "simple-jira"
    },
    {
      "parameters": {
        "url": "http://host.docker.internal:8001/health",
        "options": {}
      },
      "name": "Test API Call",
      "type": "n8n-nodes-base.httpRequest",
      "position": [460, 300]
    }
  ],
  "connections": {
    "Jira Webhook": {
      "main": [
        [
          {
            "node": "Test API Call",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

To import this:
1. Copy the JSON above
2. In n8n, click the **"..."** menu → **"Import from clipboard"**
3. Paste the JSON and click **"Import"**

## Troubleshooting

### Issue: "host.docker.internal not found"
**Solution**: Change the URL to `http://localhost:8001` or `http://192.168.86.36:8001`

### Issue: "Connection refused"
**Solution**: Make sure the API server is running:
```bash
python simple_api_server.py
```

### Issue: "Workflow not executing"
**Solution**: Make sure the workflow is **activated** (toggle in top-right should be green)

## What You Should See

Once working, you'll see:

1. **Visual Workflow**: Boxes connected with arrows showing data flow
2. **Real-time Execution**: When you send a webhook, nodes light up as they execute
3. **Data Inspector**: Click on any node to see the exact data it received/sent
4. **Execution History**: List of all webhook calls and their results

This gives you a visual representation of how the adaptive ML pipeline processes each Jira ticket!
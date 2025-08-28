# 🚀 Quick Start Guide - Adaptive Jira Demo

## Current Status ✅
- **N8N**: Running on http://localhost:5678
- **API Server**: Running on http://localhost:8001  
- **Services**: All working and tested

## Step 1: Access N8N Interface

1. **Open Browser**: Go to http://localhost:5678
2. **Setup Account**: 
   - If prompted, create any username/password you want
   - Complete the initial setup wizard (just click through)
3. **You should see**: Empty n8n workspace with "Create your first workflow"

## Step 2: Import Simple Workflow

**Option A: Copy/Paste Method (Easiest)**
1. Click **"Create your first workflow"** or the **"+"** button
2. Click the **"..."** menu (three dots) in top-right
3. Select **"Import from clipboard"**
4. Copy the entire content from `n8n/simple_workflow.json` and paste it
5. Click **"Import"**

**Option B: Manual Creation**
1. Click **"Create your first workflow"**
2. Click **"Add first step"**
3. Search for **"Webhook"** and select it
4. Set **Path**: `jira-webhook`
5. Set **HTTP Method**: `POST` 
6. Click **"Save"**

## Step 3: Activate the Workflow

1. **Toggle Switch**: Click the toggle in the top-right corner
2. **Should turn GREEN**: This means the workflow is active
3. **Copy Webhook URL**: Click on the webhook node and copy the "Production URL"

## Step 4: Test the Integration

Run the test script:
```bash
cd /Users/yovadia/code/auto-fine-tune
python test_webhook.py
```

**Expected Results:**
- ✅ API endpoints working
- ✅ Webhook returns 200 OK
- 📊 Execution appears in n8n

## Step 5: View Results in N8N

1. **Go to Executions**: Click "Executions" in the left sidebar
2. **Click Latest Execution**: You should see a successful execution
3. **Explore Data Flow**: Click on each node to see data flowing through
4. **Watch the Magic**: See how Jira ticket → Quality Check → Training → Success!

## What You'll See

### Visual Workflow
```
📥 Jira Webhook → 📊 Extract Data → ✅ Quality Check → 🔄 Quality Gate
                                                           ↓
📢 Success Notification ← 🤖 Adaptive Training ←─────────── (if quality good)
                                                           ↓
📢 Low Quality Alert ←─────────────────────────────────── (if quality poor)
```

### Real Data Flow
- **Webhook receives**: Jira ticket payload
- **Extraction**: Parses ticket info (key, summary, description, etc.)
- **Quality Check**: API call returns quality score
- **Decision Gate**: Routes to training OR alert based on quality
- **Training**: Simulates adaptive model training
- **Notification**: Logs success message

## Troubleshooting

### Issue: "Webhook not found" (404)
**Solution**: Make sure workflow is **activated** (green toggle)

### Issue: "Connection refused" 
**Solution**: Check API server is running:
```bash
python simple_api_server.py
```

### Issue: Can't see executions
**Solution**: 
1. Make sure workflow is activated
2. Check "Executions" tab in left sidebar
3. Try sending webhook again

### Issue: Nodes showing errors
**Solution**: 
1. Click on the red node to see error details
2. Usually it's a URL issue - change `localhost:8001` if needed
3. Check the API server logs

## Demo Features Working

- ✅ **Webhook Reception**: Receives Jira-style payloads
- ✅ **Data Extraction**: Parses ticket information  
- ✅ **Quality Assessment**: Scores ticket quality (0-1.0)
- ✅ **Decision Logic**: Routes based on quality threshold
- ✅ **Simulated Training**: Calls adaptive training endpoint
- ✅ **Notifications**: Success/failure logging
- ✅ **Real-time Monitoring**: Watch execution in n8n interface

## Next Steps

Once this is working, you can:

1. **Add More Nodes**: Duplicate detection, privacy scrubbing, A/B testing
2. **Real Integration**: Connect to actual Jira instance
3. **Full ML Pipeline**: Replace demo API with real model training
4. **Monitoring**: Add Slack/email notifications
5. **Scaling**: Add error handling, retries, parallel processing

## Test Scenarios

Try different ticket qualities by modifying `test_webhook.py`:

**High Quality Ticket** (will trigger training):
```python
"summary": "Database connection timeout in production with detailed error logs",
"description": "Detailed 200+ character description with specific error details and environment info..."
```

**Low Quality Ticket** (will trigger alert):
```python
"summary": "Bug",
"description": "It's broken"
```

This gives you a complete visual demonstration of how adaptive ML systems can automatically process and learn from incoming tickets! 🎉
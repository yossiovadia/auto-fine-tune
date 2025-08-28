# N8N Setup Guide for Adaptive Jira Analysis

## Installation Options

### Option 1: Docker Compose (Recommended)

```bash
# Navigate to n8n directory
cd n8n

# Start n8n with all dependencies
docker-compose up -d

# Check if services are running
docker-compose ps
```

### Option 2: Local Installation

```bash
# Install n8n globally
npm install n8n -g

# Start n8n
n8n start

# Or with custom settings
N8N_BASIC_AUTH_ACTIVE=true \
N8N_BASIC_AUTH_USER=admin \
N8N_BASIC_AUTH_PASSWORD=adaptive_ml_2024 \
n8n start
```

### Option 3: Development Setup

```bash
# Clone n8n for custom development
git clone https://github.com/n8n-io/n8n.git
cd n8n
npm install
npm run build
npm start
```

## Initial Configuration

### 1. Access N8N

Open your browser and go to:
- **URL**: http://localhost:5678
- **Username**: admin
- **Password**: adaptive_ml_2024

### 2. Import Workflow

1. Click "+" to create new workflow
2. Click the "..." menu → "Import from file"
3. Select `workflows/adaptive-jira-analysis.json`
4. Save the workflow

### 3. Configure Credentials

#### Jira Webhook Credential
1. Go to Credentials → Add Credential
2. Select "HTTP Header Auth"
3. Name: "Jira Webhook Auth"
4. Header Name: "Authorization"
5. Header Value: "Bearer your_webhook_token"

#### Adaptive ML API Credential
1. Add new "HTTP Header Auth" credential
2. Name: "Adaptive ML API"
3. Header Name: "X-API-Key"
4. Header Value: "your_adaptive_ml_api_key"

#### Slack Notifications (Optional)
1. Add "Slack" credential
2. Webhook URL: Your Slack webhook URL
3. Channel: #adaptive-ml-alerts

### 4. Configure Webhook in Jira

1. Go to Jira Settings → System → WebHooks
2. Create new webhook:
   - **Name**: Adaptive ML Integration
   - **URL**: http://your-n8n-server:5678/webhook/jira-webhook
   - **Events**: Issue created, Issue updated, Issue deleted
   - **JQL Filter**: project = YOUR_PROJECT

## Workflow Structure Visual

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Jira Webhook   │───▶│  Event Router   │───▶│ Ticket Handlers │
│                 │    │                 │    │                 │
│ • POST webhook  │    │ • Created       │    │ • New tickets   │
│ • JSON payload  │    │ • Updated       │    │ • Updates       │
│ • Events filter │    │ • Resolved      │    │ • Resolutions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Webhook Response│    │ Duplicate Check │    │ Quality Check   │
│                 │    │                 │    │                 │
│ • Immediate ACK │    │ • Similarity    │    │ • Score >= 0.6  │
│ • Status update │    │ • Threshold 85% │    │ • Valid format  │
│ • Logging       │    │ • Skip if dup   │    │ • Skip if poor  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Privacy Scrub   │    │ Dataset Convert │
                       │                 │    │                 │
                       │ • Remove PII    │    │ • Instruction   │
                       │ • Sanitize data │    │ • Input/Output  │
                       │ • Log scrubbed  │    │ • Multiple tasks│
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Training Trigger│    │ Model Register  │
                       │                 │    │                 │
                       │ • Adaptive mode │    │ • Version mgmt  │
                       │ • Single ticket │    │ • Metadata      │
                       │ • Queue job     │    │ • A/B candidate │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Success/Failure │    │  Notifications  │
                       │                 │    │                 │
                       │ • Status check  │    │ • Slack alerts  │
                       │ • Error handling│    │ • Email reports │
                       │ • Retry logic   │    │ • Dashboard     │
                       └─────────────────┘    └─────────────────┘
```

## API Integration Requirements

You'll need to create a simple API server to handle n8n requests:

### Create API Server

```python
# api_server.py
from flask import Flask, request, jsonify
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import ModelManager, ModelInference
from data import DataValidator, DatasetConverter
from scripts.adaptive_training import main as train_main

app = Flask(__name__)

@app.route('/api/duplicate-check', methods=['POST'])
def duplicate_check():
    data = request.json
    # Implement duplicate detection logic
    return jsonify({
        'is_duplicate': False,
        'similarity_score': 0.0,
        'similar_tickets': []
    })

@app.route('/api/quality-check', methods=['POST'])
def quality_check():
    ticket_data = request.json.get('ticket_data')
    validator = DataValidator()
    result = validator.check_single_issue_quality(ticket_data)
    return jsonify(result)

@app.route('/api/convert-dataset', methods=['POST'])
def convert_dataset():
    ticket_data = request.json.get('ticket_data')
    converter = DatasetConverter()
    examples = converter.convert_single_issue(ticket_data)
    return jsonify({'examples': examples, 'count': len(examples)})

@app.route('/api/adaptive-training', methods=['POST'])
def adaptive_training():
    data = request.json
    # Call training script
    result = train_main(data)
    return jsonify(result)

@app.route('/api/register-model', methods=['POST'])
def register_model():
    data = request.json
    model_manager = ModelManager()
    model_id = model_manager.register_model(
        model_path=data['model_path'],
        model_type=data['model_type'],
        metadata=data['metadata']
    )
    return jsonify({'model_id': model_id})

@app.route('/api/ab-test/start', methods=['POST'])
def start_ab_test():
    data = request.json
    model_manager = ModelManager()
    success = model_manager.start_ab_test(
        candidate_model_id=data['candidate_model_id'],
        traffic_split=data['traffic_split']
    )
    return jsonify({'success': success})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
```

### Start API Server

```bash
# In your project root
python api_server.py
```

## Workflow Testing

### 1. Test Webhook

```bash
# Send test webhook payload
curl -X POST http://localhost:5678/webhook/jira-webhook \
  -H "Content-Type: application/json" \
  -d '{
    "issue_event_type_name": "issue_created",
    "issue": {
      "key": "TEST-123",
      "fields": {
        "summary": "Test issue for adaptive ML",
        "description": "This is a test issue to verify the workflow",
        "issuetype": {"name": "Bug"},
        "priority": {"name": "High"},
        "status": {"name": "Open"},
        "project": {"key": "TEST"}
      }
    }
  }'
```

### 2. Monitor Executions

1. Go to n8n UI → Executions
2. Check execution logs
3. Verify each node completed successfully
4. Check error messages if any failures

### 3. Verify Integration

1. Check that API endpoints respond correctly
2. Verify Slack notifications are sent
3. Confirm model training is triggered
4. Validate A/B test setup

## Monitoring and Maintenance

### Health Checks

```bash
# Check n8n health
curl http://localhost:5678/healthz

# Check API server health
curl http://localhost:8000/health

# Check workflow status
curl http://localhost:5678/rest/workflows
```

### Log Monitoring

```bash
# n8n logs
docker-compose logs -f n8n

# API server logs
tail -f logs/api_server.log

# Workflow execution logs
# Available in n8n UI → Executions
```

### Backup and Recovery

```bash
# Backup n8n data
docker-compose exec n8n tar czf /tmp/n8n-backup.tar.gz /home/node/.n8n
docker cp adaptive-ml-n8n:/tmp/n8n-backup.tar.gz ./backups/

# Restore n8n data
docker cp ./backups/n8n-backup.tar.gz adaptive-ml-n8n:/tmp/
docker-compose exec n8n tar xzf /tmp/n8n-backup.tar.gz -C /
```

## Troubleshooting

### Common Issues

**Issue**: Webhook not triggering
**Solution**: Check Jira webhook configuration and n8n webhook URL

**Issue**: API calls failing
**Solution**: Verify API server is running and credentials are correct

**Issue**: Training not starting
**Solution**: Check model training script permissions and dependencies

**Issue**: Notifications not sending
**Solution**: Verify Slack webhook URL and channel permissions

### Debug Commands

```bash
# Test individual nodes
# Use n8n UI → Node → Execute Node

# Test API endpoints
curl -X POST http://localhost:8000/api/quality-check \
  -H "Content-Type: application/json" \
  -d '{"ticket_data": {"summary": "test", "description": "test description"}}'

# Check webhook registration
curl http://localhost:5678/rest/webhooks

# Verify workflow is active
curl http://localhost:5678/rest/workflows/active
```

## Security Considerations

1. **Authentication**: Always use basic auth or API keys
2. **HTTPS**: Use HTTPS in production deployments
3. **Network**: Restrict network access to required ports only
4. **Secrets**: Store sensitive data in n8n credentials, not in workflow
5. **Validation**: Validate all incoming webhook data
6. **Logging**: Avoid logging sensitive information

## Production Deployment

For production, consider:

1. **Load Balancer**: Use nginx or similar for SSL termination
2. **Database**: Switch from SQLite to PostgreSQL
3. **Queue**: Use Redis for job queuing
4. **Monitoring**: Add Prometheus/Grafana monitoring
5. **Scaling**: Use n8n in queue mode for horizontal scaling
6. **Backup**: Automated daily backups
7. **SSL**: Proper SSL certificates for webhooks

This setup provides a robust foundation for automating your adaptive ML pipeline with n8n!
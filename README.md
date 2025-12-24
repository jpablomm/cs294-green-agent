# CS294 Green Agent - OSWorld Assessment

A2A-compliant agent for running [OSWorld](https://os-world.github.io/) desktop automation benchmarks. Part of the Berkeley CS294 Agentic Systems course project.

## Overview

This "green agent" orchestrates OSWorld assessments by:
1. Creating GCP VMs from golden images
2. Setting up OSWorld tasks on the VM
3. Coordinating with "white agents" (LLM-based decision makers) to execute tasks
4. Evaluating results using OSWorld's evaluation framework
5. Reporting standardized metrics (success rate, steps, execution time)

## Project Structure

```
src/
└─ server.py           # A2A server entry point
green_agent/
├─ a2a/
│  ├─ executor.py      # GreenAgentExecutor - main orchestration logic
│  ├─ vm_manager.py    # GCP VM lifecycle management
│  ├─ vm_pool.py       # VM pooling for efficiency
│  └─ task_executor.py # OSWorld task execution
├─ osworld_evaluator.py # Task evaluation
└─ config.py           # Configuration
white_agent/           # White agent code (for reference)
vendor/OSWorld/        # OSWorld vendor code
tests/
└─ test_agent.py       # A2A conformance tests
```

## Requirements

- Python 3.13+
- GCP project with Compute Engine API enabled
- GCP service account credentials
- OSWorld golden VM images

## Environment Variables

```bash
# Required for GCP VM management
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Optional
GREEN_AGENT_API_KEY=your-api-key  # For API authentication
AGENT_URL=https://your-agent-url  # For agent card URL
```

## Running Locally

```bash
# Install dependencies
uv sync

# Set up GCP credentials
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Run the server
uv run src/server.py --host 0.0.0.0 --port 9009
```

## Running with Docker

```bash
# Build the image
docker build -t cs294-green-agent .

# Run the container
docker run -p 9009:9009 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
  -v /path/to/credentials.json:/credentials.json:ro \
  cs294-green-agent
```

## Agent Card

The agent exposes an A2A-compliant agent card at `/.well-known/agent-card.json`:

```json
{
  "name": "OSWorld Assessment Agent (CS294)",
  "description": "Green agent for conducting OSWorld desktop automation assessments...",
  "skills": [
    {
      "id": "osworld-assessment",
      "name": "OSWorld Assessment",
      "description": "Run OSWorld desktop automation benchmark assessments..."
    }
  ]
}
```

## Sending Tasks

Send tasks using the A2A protocol. Example task message:

```json
{
  "white_agent_url": "http://white-agent:9009",
  "osworld_task_id": "ec4e3f68-9ea4-4c18-a5c9-69f89d1178b3",
  "max_steps": 15
}
```

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Start the agent
uv run src/server.py &

# Run A2A conformance tests
uv run pytest --agent-url http://localhost:9009
```

## CI/CD

The GitHub Actions workflow automatically:
1. Builds the Docker image
2. Runs A2A conformance tests
3. Publishes to GitHub Container Registry (on push to main or version tags)

### Required Secrets

Add these as repository secrets in GitHub:
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Base64-encoded GCP service account key

## Docker Image

After CI runs successfully, the image is available at:
```
ghcr.io/jpablomm/cs294-green-agent:latest
```

## License

MIT

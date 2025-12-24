"""
Tests for CS294 Green Agent

These tests validate A2A protocol conformance and basic agent functionality.
Note: The message tests may fail without proper GCP credentials since
the agent needs to create VMs to process tasks.
"""

from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    required_fields = frozenset([
        'name',
        'description',
        'url',
        'version',
        'capabilities',
        'defaultInputModes',
        'defaultOutputModes',
        'skills',
    ])

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    if 'capabilities' in card_data and not isinstance(card_data['capabilities'], dict):
        errors.append("Field 'capabilities' must be an object.")

    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append("Field 'skills' must be an array of AgentSkill objects.")
        elif not card_data['skills']:
            errors.append("Field 'skills' array is empty.")

    return errors


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)


def test_agent_card_skills(agent):
    """Validate that our agent has the expected skills."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    card_data = response.json()

    skills = card_data.get('skills', [])
    skill_ids = [s.get('id') for s in skills]

    assert 'osworld-assessment' in skill_ids, "Agent should have osworld-assessment skill"


def test_health_endpoint(agent):
    """Test our custom health endpoint."""
    response = httpx.get(f"{agent}/health")
    assert response.status_code == 200, "Health endpoint must return 200"

    data = response.json()
    assert data.get('status') == 'healthy'
    assert data.get('agent_type') == 'green'
    assert data.get('protocol') == 'a2a'


def test_agent_card_provider(agent):
    """Validate provider information in agent card."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    card_data = response.json()

    provider = card_data.get('provider', {})
    assert provider.get('organization') == 'Berkeley CS294'
    assert 'github.com' in provider.get('url', '')


# Note: Message tests are skipped by default because they require GCP credentials
# Uncomment these tests when running with proper credentials

# @pytest.mark.asyncio
# @pytest.mark.parametrize("streaming", [True, False])
# async def test_message(agent, streaming):
#     """Test that agent returns valid A2A message format."""
#     # This test requires GCP credentials to process the message
#     pass

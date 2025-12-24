"""
CS294 Green Agent Server - OSWorld Assessment

A2A-compliant server for running OSWorld desktop automation benchmarks.
This agent orchestrates VM creation, task setup, white agent execution,
and evaluation.
"""

import argparse
import logging
import os

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AgentProvider,
)

# Import the GreenAgentExecutor
from green_agent.a2a.executor import GreenAgentExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_agent_card(url: str) -> AgentCard:
    """Create the A2A agent card with skills and capabilities."""

    skills = [
        AgentSkill(
            id="osworld-assessment",
            name="OSWorld Assessment",
            description=(
                "Run OSWorld desktop automation benchmark assessments. "
                "Creates GCP VMs, orchestrates task execution with white agents, "
                "and reports standardized metrics (success rate, steps, execution time)."
            ),
            tags=["benchmark", "desktop", "automation", "assessment", "osworld", "gcp"],
            examples=[
                '{"white_agent_url": "http://white-agent:9009", "osworld_task_id": "ec4e3f68-9ea4-4c18-a5c9-69f89d1178b3", "max_steps": 15}'
            ],
        ),
    ]

    return AgentCard(
        name="OSWorld Assessment Agent (CS294)",
        description=(
            "Green agent for conducting OSWorld desktop automation assessments. "
            "This agent creates VMs from golden images on GCP, orchestrates task "
            "execution with white agents (LLM-based decision makers), and reports "
            "standardized metrics including success rate, steps taken, and execution time. "
            "Part of the Berkeley CS294 Agentic Systems course project."
        ),
        url=url,
        version="1.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True,
        ),
        skills=skills,
        provider=AgentProvider(
            organization="Berkeley CS294",
            url="https://github.com/jpablomm/cs294-green-agent",
        ),
    )


async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "agent_type": "green",
        "protocol": "a2a",
        "version": "1.0.0",
    })


def main():
    parser = argparse.ArgumentParser(description="Run the CS294 Green Agent.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Determine the URL for the agent card
    card_url = args.card_url or os.environ.get("AGENT_URL") or f"http://{args.host}:{args.port}/"

    logger.info(f"Starting CS294 Green Agent on {args.host}:{args.port}")
    logger.info(f"Agent card URL: {card_url}")

    # Create agent card
    agent_card = create_agent_card(card_url)

    # Create executor and request handler
    executor = GreenAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Get A2A routes and add custom routes
    routes = a2a_app.routes() + [
        Route("/health", health_check, methods=["GET"]),
    ]

    # Build the Starlette app
    app = Starlette(routes=routes)

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

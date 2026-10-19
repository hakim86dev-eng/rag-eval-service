"""ASGI entry point: ``uvicorn rag_eval.main:app``."""

from rag_eval.api import create_app

app = create_app()

"""Compatibility route module for deployments that include routers explicitly."""
from .main import app, create_app

__all__ = ["app", "create_app"]

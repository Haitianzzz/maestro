"""Fake Flask app entry point for fixture repo."""


def create_app() -> dict:
    """Return a dummy app config."""
    return {"routes": []}


def register_blueprint(app: dict, name: str) -> None:
    app["routes"].append(name)

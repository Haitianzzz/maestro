"""Tiny User model (fixture)."""


class User:
    """Minimal user record."""

    def __init__(self, email: str, password_hash: str) -> None:
        self.email = email
        self.password_hash = password_hash

    def check_password(self, password: str) -> bool:
        return self.password_hash == password

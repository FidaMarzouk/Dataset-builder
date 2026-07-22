# app_clients/base.py
from typing import Optional, Protocol

class AppClient(Protocol):
    def __call__(self, question: str, required_fields: Optional[set[str]] = None) -> dict:
        ...
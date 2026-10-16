from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from app.llm.schemas import ToolDefinition
from app.tools.base import BaseTool
from app.tools.errors import ToolNotFoundError, ToolRegistrationError


class ToolRegistry:
    """Explicit allow-list of independently invokable application tools."""

    def __init__(self, tools: Iterable[BaseTool] = ()):
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.") from exc

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> BaseModel:
        return self.get(name).invoke(arguments)


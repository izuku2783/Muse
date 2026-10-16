from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.schemas import ToolDefinition
from app.tools.errors import ToolError, ToolExecutionError, ToolInputError


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseTool(ABC, Generic[InputT, OutputT]):
    """One typed application operation exposed at the future agent boundary."""

    name: str
    description: str
    input_type: type[InputT]
    output_type: type[OutputT]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_type.model_json_schema(),
        )

    def invoke(self, arguments: Mapping[str, Any]) -> OutputT:
        try:
            tool_input = self.input_type.model_validate(dict(arguments))
        except ValidationError as exc:
            raise ToolInputError(f"Invalid input for tool '{self.name}'.") from exc

        try:
            result = self.run(tool_input)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{self.name}' could not complete.") from exc

        if not isinstance(result, self.output_type):
            raise ToolExecutionError(
                f"Tool '{self.name}' returned an invalid output type."
            )
        return result

    @abstractmethod
    def run(self, tool_input: InputT) -> OutputT:
        """Execute the operation with already validated input."""


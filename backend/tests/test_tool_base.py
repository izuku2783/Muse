import pytest
from pydantic import BaseModel, ConfigDict

from app.tools.base import BaseTool
from app.tools.errors import (
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from app.tools.registry import ToolRegistry


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class EchoOutput(BaseModel):
    value: str


class EchoTool(BaseTool[EchoInput, EchoOutput]):
    name = "echo"
    description = "Return the supplied value."
    input_type = EchoInput
    output_type = EchoOutput

    def run(self, tool_input):
        return EchoOutput(value=tool_input.value)


class BrokenTool(EchoTool):
    name = "broken"

    def run(self, tool_input):
        raise RuntimeError("sensitive implementation detail")


def test_tool_definition_uses_typed_input_json_schema():
    definition = EchoTool().definition
    assert definition.name == "echo"
    assert definition.parameters["type"] == "object"
    assert definition.parameters["additionalProperties"] is False
    assert definition.parameters["required"] == ["value"]


def test_tool_validates_input_and_returns_typed_output():
    result = EchoTool().invoke({"value": "hello"})
    assert isinstance(result, EchoOutput)
    assert result.value == "hello"


def test_tool_rejects_invalid_input_with_stable_error():
    with pytest.raises(ToolInputError) as error:
        EchoTool().invoke({"unknown": "hello"})
    assert error.value.code == "invalid_tool_input"
    assert str(error.value) == "Invalid input for tool 'echo'."


def test_tool_hides_unexpected_internal_errors():
    with pytest.raises(ToolExecutionError) as error:
        BrokenTool().invoke({"value": "hello"})
    assert error.value.code == "tool_execution_error"
    assert "sensitive" not in str(error.value)


def test_registry_is_an_explicit_allow_list():
    registry = ToolRegistry([EchoTool()])
    assert registry.invoke("echo", {"value": "hello"}) == EchoOutput(value="hello")
    assert [definition.name for definition in registry.definitions()] == ["echo"]

    with pytest.raises(ToolNotFoundError, match="not registered"):
        registry.invoke("missing", {})


def test_registry_rejects_duplicate_names():
    with pytest.raises(ToolRegistrationError, match="already registered"):
        ToolRegistry([EchoTool(), EchoTool()])

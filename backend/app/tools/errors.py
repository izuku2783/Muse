class ToolError(Exception):
    """Base error with a stable machine-readable code for tool execution."""

    code = "tool_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ToolInputError(ToolError):
    code = "invalid_tool_input"


class ToolNotFoundError(ToolError):
    code = "tool_not_found"


class ToolRegistrationError(ToolError):
    code = "tool_registration_error"


class ToolEntityNotFoundError(ToolError):
    code = "entity_not_found"


class ToolExecutionError(ToolError):
    code = "tool_execution_error"


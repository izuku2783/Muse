from app.memory.schemas import PreferenceCategory, UserPreferenceProfile
from app.memory.service import UserMemoryService
from app.tools.base import BaseTool
from app.tools.schemas import GetUserPreferencesInput


class GetUserPreferencesTool(
    BaseTool[GetUserPreferencesInput, UserPreferenceProfile]
):
    name = "get_user_preferences"
    description = "Read a user's explicit long-term music preferences."
    input_type = GetUserPreferencesInput
    output_type = UserPreferenceProfile

    def __init__(self, service: UserMemoryService):
        self.service = service

    def run(self, tool_input: GetUserPreferencesInput) -> UserPreferenceProfile:
        category = (
            PreferenceCategory(tool_input.category)
            if tool_input.category is not None
            else None
        )
        return self.service.get_profile(tool_input.user_id, category=category)

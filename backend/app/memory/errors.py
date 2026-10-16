class UserMemoryError(Exception):
    """Base error for explicit long-term preference memory."""


class UserPreferenceNotFoundError(UserMemoryError):
    pass


class UserPreferenceConflictError(UserMemoryError):
    pass

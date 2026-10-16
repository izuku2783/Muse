class ConversationError(Exception):
    """Base error for short-term conversation state."""


class ConversationNotFoundError(ConversationError):
    pass


class ConversationConflictError(ConversationError):
    pass

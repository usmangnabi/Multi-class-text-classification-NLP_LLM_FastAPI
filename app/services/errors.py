# Our own error types, so the API can send clear messages
class ModelNotFoundError(Exception):
    """Raised when a trained model file is missing (run the Colab notebooks first)."""


class LLMNotConfiguredError(Exception):
    """Raised when the Groq API key is missing."""

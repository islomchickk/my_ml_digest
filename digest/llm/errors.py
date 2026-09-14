class LLMResponseError(RuntimeError):
    """An unusable completion, with the original API response for diagnostics."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response

class AppError(Exception):
    """An error whose message is safe to show to the end user."""

    def __init__(self, code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

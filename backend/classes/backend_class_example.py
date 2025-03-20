class HelloWorld:
    """Client for interacting with the Hello World API."""

    def __init__(self):
        self.message = "hello world"

    def get(self) -> str:
        return self.message
from typing import Union, Dict
import requests


class HelloWorld:
    """Client for interacting with the Hello World API."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_hello_world(self) -> Union[Dict, None]:
        """
        Fetches the Hello World message from the API.

        Returns:
            A dictionary containing the API response on success, or None on failure.
        """
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                return response.json()  # Successful response
            return {"error": f"Unexpected status code: {response.status_code}"}
        except requests.RequestException as e:
            return {"error": str(e)}

from classes.frontend_class_example import HelloWorld
import streamlit as st

def main():
    st.title("Streamlit and FastAPI Integration")

    # Initialize your HelloWorldClient
    api_client = HelloWorld(base_url="http://backend:8000")

    # Fetch the Hello World message
    response = api_client.get_hello_world()

    # Handle response consistently
    if "error" in response:
        st.error(f"Error: {response['error']}")
        return
    
    st.success(f"API Response: {response}")


if __name__ == "__main__":
    main()

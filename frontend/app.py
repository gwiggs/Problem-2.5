from classes.frontend_class_example import HelloWorld
import streamlit as st
import pandas as pd  # Import pandas for handling CSV files
import pydeck as pdk  # Import pydeck for advanced map visualizations

def main():
    st.title("G-DELTA")

    # Add file upload functionality for multiple CSV files
    uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        all_dataframes = []
        try:
            # Process each uploaded file and append to a list
            for uploaded_file in uploaded_files:
                df = pd.read_csv(uploaded_file)
                all_dataframes.append(df)

            # Concatenate all dataframes into one
            combined_df = pd.concat(all_dataframes, ignore_index=True)

            # Ensure latitude and longitude fields are numeric
            combined_df['actiongeolat'] = pd.to_numeric(combined_df['actiongeolat'], errors='coerce')
            combined_df['actiongeolong'] = pd.to_numeric(combined_df['actiongeolong'], errors='coerce')
            
            # Prepare data for the map
            map_data = combined_df.dropna(subset=['actiongeolat', 'actiongeolong'])

            # Add a map visualization with tooltips
            st.write("Map Visualization:")
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[actiongeolong, actiongeolat]',
                get_radius=1000,
                get_fill_color=[255, 0, 0, 140],
                pickable=True,
                tooltip=True
            )
            tooltip = {
                "html": "<b>Date/Time:</b> {datetime_of_article}<br>"
                        "<b>Description:</b> {cameocodedescription}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            view_state = pdk.ViewState(
                latitude=map_data['actiongeolat'].mean(),
                longitude=map_data['actiongeolong'].mean(),
                zoom=3,
                pitch=0
            )
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
            st.pydeck_chart(deck)

            # Sidebar for selecting columns to display
            with st.sidebar:
                st.write("Select columns to display:")
                default_columns = ["datetime_of_article", "actiongeofullname", "cameocodedescription", "goldsteinscale"]
                selected_columns = [
                    column for column in combined_df.columns
                    if st.checkbox(column, value=(column in default_columns))
                ]

            # Display the selected columns from the combined dataframe
            if selected_columns:
                st.write("Selected Columns Data from All Files:")
                st.dataframe(combined_df[selected_columns])
            else:
                st.info("No columns selected.")

        except Exception as e:
            st.error(f"Error processing uploaded files: {e}")

    # Initialize your HelloWorldClient
    api_client = HelloWorld(base_url="http://backend:8000")

    # Fetch the Hello World message
    response = api_client.get_hello_world()

    # Handle response consistently
    if "error" in response:
        st.error(f"Error: {response['error']}")
        return
    
    # Un-comment for api
    # st.success(f"API Response: {response}")


if __name__ == "__main__":
    main()
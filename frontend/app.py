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

            # Convert datetime_of_article to datetime format
            combined_df['datetime_of_article'] = pd.to_datetime(combined_df['datetime_of_article'], errors='coerce')

            # Add a slider for selecting a date range
            min_date = combined_df['datetime_of_article'].min().to_pydatetime()
            max_date = combined_df['datetime_of_article'].max().to_pydatetime()
            date_range = st.slider(
                "Select a date range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )

            # Filter data based on the selected date range
            filtered_data = combined_df[
                (combined_df['datetime_of_article'] >= date_range[0]) &
                (combined_df['datetime_of_article'] <= date_range[1])
            ]

            # Prepare data for the map
            map_data = filtered_data.dropna(subset=['actiongeolat', 'actiongeolong'])

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
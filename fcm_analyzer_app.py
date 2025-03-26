# code from fcm-gating-app-v2.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import fcsparser
import tempfile
from streamlit_plotly_events import plotly_events
import os

# Initialize session state
if 'df' not in st.session_state:
    st.session_state["df"] = pd.DataFrame()  # Initialize empty DataFrame

if 'original_df' not in st.session_state:  # Store the original DataFrame
    st.session_state.original_df = None

if 'gated_data' not in st.session_state:
    st.session_state.gated_data = {}

if 'config' not in st.session_state:
    st.session_state.config = {}

if 'selected_points' not in st.session_state:
    st.session_state.selected_points = pd.DataFrame()

if 'current_gate' not in st.session_state:
    st.session_state.current_gate = None

if 'gate_history' not in st.session_state:
    st.session_state.gate_history = []


# Convert config to formatted JSON
@st.cache_data  # Cache the result of this function
def format_config(existing_config):
    formatted_config = {"seed": None, "populations": [], "parameters": []}

    if existing_config:
        first_gate = next(iter(existing_config))
        formatted_config["parameters"] = list(existing_config[first_gate].keys())

        for gate_name, stats in existing_config.items():
            population_entry = {"name": gate_name, "size": stats[f"{formatted_config['parameters'][0]}"]["count"]}

            for param in formatted_config["parameters"]:
                population_entry[f"{param}_mean"] = stats[param]["mean"]
                population_entry[f"{param}_std"] = stats[param]["std"]

            formatted_config["populations"].append(population_entry)

    return formatted_config


@st.cache_data
def load_fcs(file):
    """Load and parse .fcs file efficiently."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fcs") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name
    meta, df = fcsparser.parse(temp_file_path, reformat_meta=True)
    return df


def upload_fcs_file():
    """Upload and parse .fcs file."""
    st.header("Upload an .fcs file")
    uploaded_file = st.file_uploader("Upload an .fcs file", type=["fcs"], key="fcs_file")
    if uploaded_file is not None:
        df = load_fcs(uploaded_file)
        st.session_state.df = df
        st.session_state.original_df = df  # Store the original DataFrame
        st.success("Data Loaded Successfully")
        return df


def apply_gating():
    """Apply gating to the data."""
    st.header("Apply Gating")

    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("Please upload an .fcs file first.")
        return

    # Display the current DataFrame (original or gated)
    current_df = st.session_state.df if not st.session_state.gate_history else st.session_state.gate_history[-1]
    st.write("Preview of the DataFrame:")
    st.write(current_df.head())

    # Select x and y axes
    cols = current_df.columns.tolist()
    x_axis = st.selectbox("Select X-axis", cols, index=0, key="x_axis")
    y_axis = st.selectbox("Select Y-axis", cols, index=1, key="y_axis")

    # Create Plotly scatter plot
    fig = px.scatter(current_df, x=x_axis, y=y_axis, opacity=0.5)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(dragmode="lasso")

    # Select points
    selected_points = plotly_events(fig, select_event=True)

    # Update session state with selected points
    if selected_points:
        st.write("Confirm your selection to apply gating.") 
        if st.button("Confirm Selection", key="confirm_selection"):
            selected_indices = [p["pointIndex"] for p in selected_points]  # Extract indices from the list of dictionaries
            gated_df = current_df.iloc[selected_indices]  # Create a DataFrame from the selected indices
            st.session_state.selected_points = gated_df  # Update the session state with the selected points
            st.session_state.gate_history.append(gated_df)  # Append the selected points to the gate history
            st.rerun()  # Rerun the app to update the UI
    
    
    # Gating options
    if not st.session_state.selected_points.empty:
        st.write(f"Number of gated points: {len(st.session_state.selected_points)}")
        stats = st.session_state.selected_points.describe().loc[['mean', 'std', 'count']]
        st.write("Statistical Properties:")
        st.dataframe(stats)

        # Display following optinos on the sidebar with st.expander()
        with st.sidebar.expander("Gating Options", expanded=True):
            # Assign a gating name
            gating_name = st.text_input("Enter gating name:", value=st.session_state.current_gate or "", key="gating_name")
            if st.button("Save Gating", key="save_gating"):
                if gating_name in st.session_state.gated_data:
                    st.warning("Gating name already exists. Do you want to overwrite it?")
                    if st.button("Yes", key="overwrite_yes"):
                        st.session_state.gated_data[gating_name] = st.session_state.selected_points
                        st.session_state.config[gating_name] = stats.to_dict()
                        st.session_state.current_gate = gating_name
                        st.success(f"Gating '{gating_name}' saved!")
                elif gating_name:
                    st.session_state.gated_data[gating_name] = st.session_state.selected_points
                    st.session_state.config[gating_name] = stats.to_dict()
                    st.session_state.current_gate = gating_name
                    st.success(f"Gating '{gating_name}' saved!")
                else:
                    st.error("Please enter a unique gating name.")

            # Reset Selection to original DataFrame
            st.write("Reset the selection to the original DataFrame.")
            if st.button("Reset Selection", key="reset_selection"):
                st.session_state.selected_points = pd.DataFrame()
                st.session_state.current_gate = None
                st.session_state.gate_history = []  # Reset gate history
                st.session_state.df = st.session_state.original_df  # Reset to the original DataFrame
                st.rerun()

            # Load saved gates
            if st.session_state.gated_data:
                selected_gate = st.selectbox("Load Saved Gate", list(st.session_state.gated_data.keys()), key="selected_gate")
                if st.button("Load Gate", key="load_gate"):
                    st.session_state.selected_points = st.session_state.gated_data[selected_gate]
                    st.session_state.current_gate = selected_gate
                    st.session_state.gate_history.append(st.session_state.selected_points)
                    st.rerun()


def download_config():
    """Download the gating configuration as a JSON file."""
    st.header("Download Configuration File")

    if st.session_state.gated_data:  # Check if there are savedconfig gates
        formatted_config = format_config(st.session_state.config)
        config_json = json.dumps(formatted_config, indent=4)
        st.download_button(label="Download JSON", data=config_json, file_name="gating_config.json", mime="application/json", key="download_json")
    else:
        st.warning("No gating data available. Please apply gating first.")


# Streamlit UI
#def fcs_data_analyzer():
st.sidebar.title("Flow Cytometry Data Analyzer")

# Navigation menu
option = st.sidebar.radio(
    "Select an option:",
    ("About the app","Upload an .fcs file", "Apply Gating", "Download Configuration File")
)

def about_the_gating_app():
    st.title(" Synthetic Flow Cytometry Data Analyzer")
    st.markdown("""
                Welcome to the **Synthetic Flow Cytometry Data Analyzer**!  
                - This app allows you to analyze flow cytometry data (`.fcs`) file by applying gating to extract statistical properties of specific populations and download the gating configuration and statistical properties as a `.json` file.
                - Downloaded JSON file can be used to reproduce the gating configuration in Synthetic FCM Data Generator application.
                
                ### How to Use the App:
                - **Upload Configuration File** 
                    - Upload an .fcs file by selecting the "Upload an `.fcs` file" option.
                    - A sample .fcs file is provided for testing, which can be downloaded from the link proved at the bottom of this page.
                    - You can use it to test the app.                
                    
                - **Apply Gating**
                    - Apply gating to the data by selecting the "Apply Gating" option.  
                    - You can select points for gating by drawing a lasso around the points on the scatter plot.
                    - Click the "Confirm Selection" button to apply the gating.
                    - The selected points will be displayed in a table along with their statistical properties.
                    - Save the gating configuration by entering a unique gating name and clicking the "Save Gating" button.
                    - Saved gates can also be loaded to visualize later.
                    - Click the "Reset Selection" button to reset the selection to the original DataFrame to start over applying gating to another population.
                    - Similarly, repeat the process to apply gating to multiple populations.
                    - Finally, download the gating configuration as a `.json` file by selecting the "Download Configuration File" option.
                    
                - **Download Configuration File**
                    - Download the gating configuration as a `.json` file by selecting the "Download Configuration File" option.
                    - The gating configuration will be saved as a `.json` file with the statistical properties of the gated populations.
                
                *Explore the sidebar for different sections and detailed functionalities!* 
                """)
    #st.markdown("[Download Sample .fcs File](https://drive.google.com/file/d/1y9K5eR7JwFkq6b2Qb3Y8Y0s6o9e4aQ7B/view?usp=sharing)")


def download_fcs_sample():
    # Define file path
    #file_path = "data/ABCDEFGHIJ-merged.fcs" 
    file_path = "data/sample_data_2.fcs"
    # Check if file exists
    if not os.path.exists(file_path):
        st.error("File not found. Please check the file path.")
        return None
    # Read the file
    with open(file_path, "rb") as file:
        fcs_file = file.read()
        
    # Display the download button
    st.download_button(label="Download Sample .fcs File",data=fcs_file, file_name="ABCDEFGHIJ-merged.fcs",
                    mime="application/octet-stream",
                    key="download_fcs")


if option == "About the app":
    about_the_gating_app()
    download_fcs_sample()

elif option == "Upload an .fcs file":
    upload_fcs_file()
elif option == "Apply Gating":
    apply_gating()
    
elif option == "Download Configuration File":
    download_config()



#if __name__ == "__main__":  # Main function
#    fcs_data_analyzer()

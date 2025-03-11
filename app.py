# Load the required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import fcswrite
from io import BytesIO
import tempfile
import os
from imblearn.over_sampling import SMOTE, ADASYN

# Function to generate synthetic data
@st.cache_data  # Cache data generation for better performance
def generate_data(populations, parameters, seed):
    np.random.seed(seed)  # Set seed for reproducibility
    data = []
    total_size = sum(pop['size'] for pop in populations)
    
    for pop in populations:
        size = pop['size']
        pop_data = {"Population": [pop["name"]] * size}
        
        for param in parameters:
            mean = pop[f'{param}_mean']
            std = pop[f'{param}_std']
            pop_data[param] = np.random.normal(mean, std, size)
        
        data.append(pd.DataFrame(pop_data))
    
    return pd.concat(data, ignore_index=True)

# Function to convert DataFrame to .fcs format
def dataframe_to_fcs(df, parameters):
    data = df[parameters].values
    channels = parameters
    channel_names = {i: param for i, param in enumerate(channels)}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fcs") as tmp_file:
        fcswrite.write_fcs(filename=tmp_file.name, chn_names=channels, data=data)
        tmp_file_path = tmp_file.name
    
    with open(tmp_file_path, "rb") as f:
        fcs_data = f.read()
    
    os.remove(tmp_file_path)
    return fcs_data

# Function to update seed in session state
def update_seed():
    st.session_state.config["seed"] = st.session_state.seed

# Function to upload configuration file
def upload_config_file():
    uploaded_config = st.file_uploader("Upload Configuration File", type=["json"], key="upload_config")
    
    if uploaded_config is not None:
        config_data = json.load(uploaded_config)
        uploaded_seed = config_data.get("seed", 2)
        uploaded_populations = config_data.get("populations", [])
        uploaded_parameters = config_data.get("parameters", [])
        
        st.session_state.seed = uploaded_seed
        st.session_state.populations = uploaded_populations
        st.session_state.parameters = uploaded_parameters
        st.session_state.config["seed"] = uploaded_seed
        st.session_state.config["populations"] = uploaded_populations
        st.session_state.config["parameters"] = uploaded_parameters
        
        st.success("Configuration file uploaded successfully!")

# Function to define/edit populations
def define_edit_populations():
    # Random Seed
    with st.sidebar.expander("Random Seed"):
        st.number_input("Random Seed", min_value=0, max_value=100000, value=st.session_state.seed, key="seed", on_change=update_seed)
    
    # Define Parameters
    with st.sidebar.expander("Define Parameters"):
        parameters_input = st.text_area("Enter parameters (comma-separated)", ", ".join(st.session_state.parameters))
        st.session_state.parameters = [param.strip() for param in parameters_input.split(",") if param.strip()]
        st.session_state.config["parameters"] = st.session_state.parameters
    
    # Populations
    with st.sidebar.expander("Populations"):
        num_pops = st.number_input("Number of Populations", min_value=1, max_value=30, value=len(st.session_state.populations) if st.session_state.populations else 3)
        
        # Initialize populations if not already present
        if not st.session_state.populations:
            for i in range(num_pops):
                st.session_state.populations.append({'name': f"Population {i+1}", 'size': 500})
        
        # Adjust populations based on user input
        if len(st.session_state.populations) < num_pops:
            for i in range(len(st.session_state.populations), num_pops):
                st.session_state.populations.append({'name': f"Population {i+1}", 'size': 500})
        elif len(st.session_state.populations) > num_pops:
            st.session_state.populations = st.session_state.populations[:num_pops]
        
        # Display population input fields
        for i, pop in enumerate(st.session_state.populations):
            st.subheader(f"Population {i+1}")
            pop['name'] = st.text_input(f"Population Name", value=pop['name'], key=f"name_{i}")
            pop['size'] = st.number_input(f"Size", min_value=100, max_value=10000, value=pop['size'], key=f"size_{i}")
            
            # Use columns to organize parameter inputs
            col1, col2 = st.columns(2)
            for param in st.session_state.parameters:
                with col1:
                    pop[f'{param}_mean'] = st.number_input(f"{param} Mean", value=pop.get(f'{param}_mean', 100), key=f"{param}_mean_{i}")
                with col2:
                    pop[f'{param}_std'] = st.number_input(f"{param} Std Dev", value=pop.get(f'{param}_std', 10), key=f"{param}_std_{i}")

# Function to display and visualize data
def visualize_data(df):
    st.write("Generated Data:")
    st.dataframe(df.head())
    st.write("Data Dimensions:", df.shape)
    st.write("Summary Statistics:")
    st.write(df.describe())

    unique_pops = df["Population"].unique()
    selected_pops = st.multiselect("Filter by Population", unique_pops, default=unique_pops)
    df_filtered = df[df["Population"].isin(selected_pops)]

    pop_counts = df_filtered["Population"].value_counts(normalize=True) * 100
    st.write("Current Population Fractions:")
    st.write(pop_counts.sort_index())

    x_axis = st.selectbox("X-Axis", df.columns, index=0, key="x_axis")
    y_axis = st.selectbox("Y-Axis", df.columns, index=1, key="y_axis")
    fig = px.scatter(df_filtered, x=x_axis, y=y_axis, opacity=0.6, color="Population")
    st.plotly_chart(fig)

    hist_param = st.selectbox("Select Parameter for Histogram", df.columns, index=0, key="hist_param")
    fig_hist = px.histogram(df_filtered, x=hist_param, color="Population", nbins=50, opacity=0.6)
    st.plotly_chart(fig_hist)

# Streamlit UI
st.sidebar.title("Synthetic Flow Cytometry Data Generator")

# Initialize session state variables
if "seed" not in st.session_state:
    st.session_state.seed = 2
if "populations" not in st.session_state:
    st.session_state.populations = []
if "parameters" not in st.session_state:
    st.session_state.parameters = []
if "config" not in st.session_state:
    st.session_state.config = {"seed": st.session_state.seed, "populations": st.session_state.populations, "parameters": st.session_state.parameters}

# Sidebar navigation
options = ["Upload Configuration File", "Define/Edit Populations", "Class Imbalance Correction", "Download Data"]
choice = st.sidebar.radio("Select an option", options)

# Handle navigation choices
if choice == "Upload Configuration File":
    upload_config_file()

    # Display app information
    st.write("This app generate synthetic flow cytometry data based on user-defined populations and parameters.")
    st.write("The app also allows users to upload a configuration file to define populations and parameters.")
    st.write("The generated data can be visualized, adjusted for class imbalance, and downloaded in .fcs. or .csv format.")
    st.write("For more information, please refer to the sidebar options.")

elif choice == "Define/Edit Populations":
    define_edit_populations()
    
    if st.sidebar.button("Generate Data"):
        df = generate_data(st.session_state.populations, st.session_state.parameters, st.session_state.seed)
        st.session_state['df'] = df
        st.success("Data generated successfully!")
    
    if 'df' in st.session_state:
        visualize_data(st.session_state['df'])
        
        # Adjust Population Fractions
        st.sidebar.header("Adjust Population Fractions")
        st.sidebar.write("Make sure the fractions sum to 100%.")
        
        current_fractions = (st.session_state['df']["Population"].value_counts(normalize=True) * 100).to_dict()
        new_sizes = {}
        
        sorted_populations = sorted(st.session_state.populations, key=lambda x: x['name'])
        for pop in sorted_populations:
            pop_name = pop['name']
            default_fraction = current_fractions.get(pop_name, 0)
            new_sizes[pop_name] = st.sidebar.slider(
                f"Fraction of {pop_name}",
                min_value=0,
                max_value=100,
                value=int(default_fraction),
                key=f"fraction_{pop_name}"
            )
        
        total_fraction = sum(new_sizes.values())
        st.sidebar.write("Total Fraction:", total_fraction)
        
        if total_fraction != 100:
            for pop_name in new_sizes:
                new_sizes[pop_name] = (new_sizes[pop_name] / total_fraction) * 100
        
        if st.sidebar.button("Update Population Fractions"):
            total_size = len(st.session_state['df'])
            for pop in st.session_state.populations:
                pop['size'] = int((new_sizes[pop['name']] / 100) * total_size)
            
            size_sum = sum(pop['size'] for pop in st.session_state.populations)
            if size_sum != total_size:
                st.session_state.populations[-1]['size'] += total_size - size_sum
            
            df = generate_data(st.session_state.populations, st.session_state.parameters, st.session_state.seed)
            st.session_state['df'] = df
            
            # Update the config with the latest populations
            st.session_state.config["populations"] = st.session_state.populations
            st.success("Population Fractions updated successfully!")
            st.rerun()
        
        # Adjust Individual Population Composition
        st.sidebar.header("Adjust Individual Population Composition")
        pop_names = [pop['name'] for pop in st.session_state.populations]
        selected_population = st.sidebar.selectbox("Select Population", pop_names)
        selected_parameter = st.sidebar.selectbox("Select Parameter", st.session_state.parameters)
        
        selected_pop_data = next((pop for pop in st.session_state.populations if pop['name'] == selected_population), None)
        
        if selected_pop_data is not None:
            expression_percentage = st.sidebar.number_input(
                f"Percentage of {selected_population} cells expressing {selected_parameter}",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1,
                key=f"expression_{selected_population}_{selected_parameter}"
            )
            
            default_high_mean = 1000.0
            default_std = 100.0
            
            expressing_mean = st.sidebar.number_input(
                f"Mean for {selected_parameter} (expressing cells)",
                min_value=0.0,
                value=default_high_mean,
                step=0.1,
                key=f"mean_{selected_population}_{selected_parameter}"
            )
            
            expressing_std = st.sidebar.number_input(
                f"Standard Deviation for {selected_parameter} (expressing cells)",
                min_value=0.0,
                value=default_std,
                step=0.1,
                key=f"std_{selected_population}_{selected_parameter}"
            )
            
            non_expressing_mean = st.sidebar.number_input(
                f"Mean for {selected_parameter} (non-expressing cells)",
                min_value=0.0,
                value=float(selected_pop_data[f'{selected_parameter}_mean']),
                step=0.1,
                key=f"non_expressing_mean_{selected_population}_{selected_parameter}"
            )
            
            non_expressing_std = st.sidebar.number_input(
                f"Standard Deviation for {selected_parameter} (non-expressing cells)",
                min_value=0.0,
                value=float(selected_pop_data[f'{selected_parameter}_std']),
                step=0.1,
                key=f"non_expressing_std_{selected_population}_{selected_parameter}"
            )
            
            if st.sidebar.button("Update Population Composition"):
                for pop in st.session_state.populations:
                    if pop['name'] == selected_population:
                        population_mask = st.session_state['df']["Population"] == selected_population
                        population_size = population_mask.sum()
                        num_expressing = int((expression_percentage / 100) * population_size)
                        
                        expressing_data = np.random.normal(expressing_mean, expressing_std, num_expressing)
                        non_expressing_data = np.random.normal(non_expressing_mean, non_expressing_std, population_size - num_expressing)
                        
                        pop[f'{selected_parameter}_data'] = np.concatenate([expressing_data, non_expressing_data])
                        st.session_state['df'].loc[population_mask, selected_parameter] = pop[f'{selected_parameter}_data']
                
                # Update the config with the latest populations
                st.session_state.config["populations"] = st.session_state.populations
                st.success("Population Composition updated successfully!")
                st.rerun()

elif choice == "Class Imbalance Correction":
    if 'df' in st.session_state:
        df = st.session_state['df']
        visualize_data(df)
        
        with st.sidebar.expander("Balance Populations"):
            desired_sizes = {}
            sorted_populations = sorted(st.session_state.populations, key=lambda x: x['name'])
            for pop in sorted_populations:
                desired_sizes[pop['name']] = st.number_input(f"Desired Size for {pop['name']}", min_value=0, max_value=10000, value=pop['size'], key=f"desired_size_{pop['name']}")
            
            balancing_technique = st.selectbox("Select Balancing Technique for oversampling", ["None", "SMOTE", "ADASYN"], index=0, key="balancing_technique")
            
            if balancing_technique in ["SMOTE", "ADASYN"]:
                k_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=50, value=5, key="num_neighbors")
            
            if st.button("Apply Balancing"):
                total_desired_size = sum(desired_sizes.values())
                scaling_factors = {pop['name']: desired_sizes[pop['name']] / pop['size'] for pop in sorted_populations}
                
                scaled_df = pd.concat([df[df["Population"] == pop['name']].sample(n=int(desired_sizes[pop['name']]), replace=True, random_state=st.session_state.seed) for pop in sorted_populations])
                
                if balancing_technique != "None":
                    X = scaled_df[st.session_state.parameters]
                    y = scaled_df["Population"]
                    
                    if balancing_technique == "SMOTE":
                        smote = SMOTE(sampling_strategy={pop: int(desired_sizes[pop]) for pop in desired_sizes}, k_neighbors=k_neighbors, random_state=st.session_state.seed)
                        X_resampled, y_resampled = smote.fit_resample(X, y)
                    elif balancing_technique == "ADASYN":
                        adasyn = ADASYN(sampling_strategy={pop: int(desired_sizes[pop]) for pop in desired_sizes}, n_neighbors=k_neighbors, random_state=st.session_state.seed)
                        X_resampled, y_resampled = adasyn.fit_resample(X, y)
                    
                    balanced_df = pd.DataFrame(X_resampled, columns=st.session_state.parameters)
                    balanced_df["Population"] = y_resampled
                    st.session_state['df'] = balanced_df
                    
                else:
                    st.session_state['df'] = scaled_df
                
                # Update population sizes in st.session_state.populations
                for pop in st.session_state.populations:
                    pop['size'] = desired_sizes[pop['name']]
                
                # Update the config with the latest populations
                st.session_state.config["populations"] = st.session_state.populations    
                
                st.success("Balancing applied successfully!")
                st.rerun()
    else:
        st.write("Please generate data to apply class imbalance correction.")

elif choice == "Download Data":
    if 'df' in st.session_state:
        df = st.session_state['df']
        fcs_data = dataframe_to_fcs(df, st.session_state.parameters)
        st.download_button(label="Download Data (.fcs)", data=fcs_data, file_name="synthetic_data.fcs", mime="application/octet-stream")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data (.csv)", csv, "synthetic_data.csv", "text/csv")
        
        # Helper function to convert NumPy arrays to lists
        def convert_ndarrays_to_lists(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, dict):
                return {k: convert_ndarrays_to_lists(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_ndarrays_to_lists(v) for v in data]
            else:
                return data
        
        # Convert NumPy arrays in the config to lists
        config_serializable = convert_ndarrays_to_lists(st.session_state.config)
        
        # Export Configuration
        config_json = json.dumps(config_serializable, indent=4).encode('utf-8')
        st.download_button("Download Configuration (.json)", config_json, "config.json", "application/json")
    else:
        st.write("Please generate data before downloading.")



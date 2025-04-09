import os
os.environ["OMP_NUM_THREADS"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import fcsparser
import fcswrite
import tempfile
from streamlit_plotly_events import plotly_events
from sklearn.mixture import GaussianMixture
import io
import hashlib # For hashing the gating data used in  gating_data_hash()
import pickle # For serializing gating data to bytes used in gating_data_hash()
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.metrics import silhouette_score, mutual_info_score
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from plotly.graph_objs import Figure
from sklearn.preprocessing import StandardScaler



# Initialize session state 
if "df" not in st.session_state:
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

if 'syn_data' not in st.session_state:
    st.session_state.syn_data = pd.DataFrame()

if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []


def reset_session_state():
    # Clear ALL session state (including Streamlit-internal keys)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize only critical defaults
    st.session_state.df = pd.DataFrame()
    st.session_state.original_df = None
    st.session_state.gated_data = {}
    st.session_state.config = {}
    st.session_state.selected_points = pd.DataFrame()
    st.session_state.current_gate = None
    st.session_state.gate_history = []
    st.session_state.syn_data = pd.DataFrame()
    
    # Force UI to reload from scratch
    st.rerun()
    
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
        # Reset gating-related session states when a new file is uploaded
        st.session_state.gated_data = {}  # Clear saved gates
        st.session_state.config = {}  # Clear config
        st.session_state.selected_points = pd.DataFrame()  # Clear selected points
        st.session_state.current_gate = None  # Reset current gate
        st.session_state.gate_history = []  # Clear gate history
        st.session_state.syn_data = pd.DataFrame()  # Clear synthetic data
        
        # Load the new file
        df = load_fcs(uploaded_file)
        st.session_state.df = df
        st.session_state.original_df = df  # Store the original DataFrame
        st.success("Data Loaded Successfully")
        #return df
        st.rerun()  # Rerun the app to update the UI

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
    st.write("Number of rows in the DataFrame:", len(current_df)) 
    st.write("Number of columns in the DataFrame:", len(current_df.columns))
    

    # Select x and y axes
    cols = current_df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", cols, index=0, key="x_axis")
        # Checkboxes for log scale
        x_log_scale = st.checkbox("Log Scale (X-axis)", key="x_log")
        
    with col2:
        y_axis = st.selectbox("Select Y-axis", cols, index=1, key="y_axis")
        # Checkboxes for log scale    
        y_log_scale = st.checkbox("Log Scale (Y-axis)", key="y_log")

    # Create Plotly scatter plot
    fig = px.scatter(current_df, x=x_axis, y=y_axis, opacity=0.5)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(dragmode="lasso")
    
    # Apply log scale if selected
    if x_log_scale:
        fig.update_layout(xaxis_type="log")
    if y_log_scale:
        fig.update_layout(yaxis_type="log")

    # Select points
    selected_points = plotly_events(fig, select_event=True)
        
    if selected_points:
        selected_indices = [p["pointIndex"] for p in selected_points]
        selected_df = current_df.iloc[selected_indices]

        if not selected_df.empty:            
            # Create a copy of the selected DataFrame for contour plotting
            contour_df = selected_df.copy()
            
            # Apply log transformation if needed BEFORE creating the contour plot
            if x_log_scale:
                contour_df[x_axis] = np.log10(contour_df[x_axis].replace(0, np.nan))  # Handle zeros
            if y_log_scale:
                contour_df[y_axis] = np.log10(contour_df[y_axis].replace(0, np.nan))  # Handle zeros
            
            # Create contour plot with potentially transformed data
            fig_contour = px.density_contour(contour_df,
                                             x=x_axis,
                                             y=y_axis,
                                             title="Density Contour of Selected Population")
            
            # Update contour styling
            fig_contour.update_traces(contours_coloring="fill",
                                      colorscale="viridis",
                                      contours_showlabels=True
                                      )
            
            # Set axis titles based on scale
            x_title = f"log10({x_axis})" if x_log_scale else x_axis
            y_title = f"log10({y_axis})" if y_log_scale else y_axis
            
            fig_contour.update_layout(
                title="Density Contour of Selected Population",
                xaxis_title=x_title,
                yaxis_title=y_title,
                coloraxis_colorbar=dict(title="Density"))
            
            # Display the contour plot
            st.plotly_chart(fig_contour, use_container_width=True)
    
                
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

    if st.session_state.gated_data:  # Check if there are saved gates
        formatted_config = format_config(st.session_state.config)
        config_json = json.dumps(formatted_config, indent=4)
        st.download_button(label="Download configuration file in .`json` format", data=config_json, file_name="gating_config.json", mime="application/json", key="download_json")
    else:
        st.warning("No gating data available. Please apply gating first.")
        
def optimal_gmm_components(df, max_components=10):
    """Determine optimal number of GMM components using BIC."""
    # Add a check to ensure max_components doesn't exceed the number of samples
    max_components = min(max_components, len(df) // 5) # At least 5 samples per component, Limit to 1/5th of the sample size
    # Above ensures at least 5-10 samples per component to avoid overfitting
    bic_scores = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(df)
        bic_scores.append(gmm.bic(df))   
        
    # # Stor n_components, bic_scores in session state for later use
    # if 'optimal_components' not in st.session_state:
    #     st.session_state['optimal_components'] = {} 
        
    # st.session_state['optimal_components']['n_components'] = np.arange(1, max_components + 1)
    # st.session_state['optimal_components']['bic_scores'] = bic_scores
                
    return np.argmin(bic_scores) + 1  # BIC selects the best component count

def get_gating_data_hash():
    """Generate a unique hash of the current gating data."""
    if not st.session_state.gated_data:
        return "no_gating_data"
    # Serialize gating data to bytes and hash it
    data_bytes = pickle.dumps(st.session_state.gated_data)
    return hashlib.md5(data_bytes).hexdigest()

def get_representative_sample(data, sample_size=30000, random_state=42):
    """Randomly sample data while preserving rare populations."""
    if not st.session_state.gated_data:
        data = st.session_state.df.values
    else:
        data = data #st.session_state.gated_data.values()
        data = pd.DataFrame(data)
    if len(data) <= sample_size:
        return data
    else:
        return data.sample(n=sample_size, random_state=random_state)


def plot_dim_reduction_new(data, method='umap', n_clusters=None, feature_names=None):
    """Interactive dimensionality reduction visualization with enhanced insights"""
    from sklearn.manifold import TSNE
    from umap import UMAP
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    
    # Standardize data first
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Reduce dimensionality
    if method == 'umap':
        reducer = UMAP(random_state=42)
    else:
        reducer = TSNE(random_state=42, perplexity=30)
    embedding = reducer.fit_transform(scaled_data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(embedding, columns=[f"{method.upper()}_1", f"{method.upper()}_2"])
    
    # Add cluster information if available
    if n_clusters:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        clusters = gmm.fit_predict(scaled_data)
        plot_df['Cluster'] = clusters.astype(str)
    
    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x=f"{method.upper()}_1",
        y=f"{method.upper()}_2",
        color='Cluster' if n_clusters else None,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        hover_name=plot_df.index,
        title=f"{method.upper()} Projection{' with Clusters' if n_clusters else ''}"
    )
    
    # Add density contours
    fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')),
        selector=dict(mode='markers'))
    
    # Add cluster ellipses if clusters exist
    if n_clusters:
        for cluster in plot_df['Cluster'].unique():
            cluster_points = plot_df[plot_df['Cluster'] == cluster]
            fig.add_trace(
                go.Scatter(
                    x=cluster_points[f"{method.upper()}_1"],
                    y=cluster_points[f"{method.upper()}_2"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.5,
                        color=px.colors.qualitative.Alphabet[int(cluster)]
                    ),
                    name=f'Cluster {cluster}',
                    hoverinfo='text',
                    hovertext=cluster_points.index,
                    showlegend=False
                )
            )
    
    # Add density heatmap
    fig.add_trace(
        go.Histogram2dContour(
            x=plot_df[f"{method.upper()}_1"],
            y=plot_df[f"{method.upper()}_2"],
            colorscale='Blues',
            showscale=False,
            opacity=0.4
        )
    )
    
    # Layout enhancements
    fig.update_layout(
        width=900,
        height=700,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.95)',
        title_font_size=16,
        legend_title_text='Cluster',
        xaxis_title=f"{method.upper()} Dimension 1",
        yaxis_title=f"{method.upper()} Dimension 2"
    )
    
    
    # Find the index of the density contour trace
    density_trace_index = None
    for i, trace in enumerate(fig.data):
        if isinstance(trace, go.Histogram2dContour):
            density_trace_index = i
            break
    
    # Construct visibility toggles
    n_traces = len(fig.data)
    all_visible = [True] * n_traces
    hide_density = all_visible.copy()
    if density_trace_index is not None:
        hide_density[density_trace_index] = False
    
    
    # Add interactive buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Show All",
                        method="update",
                        args=[{"visible": all_visible}]
                    ),
                    dict(
                        label="Hide Density",
                        method="update",
                        args=[{"visible": hide_density}]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    # Return the figure object instead of displaying it
    return fig

def cluster_diagnostics_ui_2(clusters, gmm, n_clusters, feature_names, population_id):
    """Display cluster diagnostics in the Streamlit app."""
    #with st.expander(f"Cluster Diagnostics – {population_id}"):
    if n_clusters:
        st.write(f"**Cluster Statistics (n={n_clusters})**")
        cluster_stats = pd.DataFrame({
            'Cluster': np.unique(clusters),
            'Count': np.bincount(clusters),
            'Percentage': np.bincount(clusters) / len(clusters) * 100
        })
        st.dataframe(cluster_stats)

        cluster_means = None
        if feature_names is not None and hasattr(gmm, "means_"):
            cluster_means = pd.DataFrame(
                gmm.means_,
                columns=feature_names,
                index=[f"Cluster {i}" for i in range(n_clusters)]
            )
            st.write("**Mean Marker Expression by Cluster**")
            st.dataframe(cluster_means.style.background_gradient(cmap='viridis'))

        # Store stats
        st.session_state.cluster_stats[population_id] = {
            "df": cluster_stats,
            "means": cluster_means,
            "n_components": n_clusters
        }
        
# Generate synthetic data function with population labels
def generate_synthetic_data_2(size_factor=1.0, method='gmm', gating_data_hash=None):
    synthetic_data = []
    population_labels = []
    gating_info = st.session_state.gated_data
    
    for pop in gating_info:
        data = gating_info[pop].values
        pop_size = int(size_factor * len(data))
        
        if method == 'gmm':
            # Initialize component count if not exists
            if f'optimal_components_{pop}' not in st.session_state:
                st.session_state[f'optimal_components_{pop}'] = optimal_gmm_components(data)
            
            # Generate with current parameters
            gmm = GaussianMixture(
                n_components=st.session_state[f'optimal_components_{pop}'],
                covariance_type='full',
                random_state=42
            )
            gmm.fit(data)
            samples, _ = gmm.sample(pop_size)
        
        else:  # Multivariate normal
            mean = gating_info[pop].mean().values
            cov = gating_info[pop].cov().values
            samples = np.random.multivariate_normal(mean, cov, size=pop_size)
        
        synthetic_data.append(samples)
        population_labels.extend([pop] * pop_size)  # Add population labels
    
    # Create DataFrame with original columns plus population
    df = pd.DataFrame(np.vstack(synthetic_data), columns=gating_info[pop].columns)
    df['population'] = population_labels  # Add population column
    
    return df

def show_component_adjustment_ui():
    """Sidebar UI for adjusting components and regenerating"""
    if 'gated_data' not in st.session_state:
        return
    
    st.sidebar.subheader("Component Adjustment")
    for pop in st.session_state.gated_data:
        current = st.session_state.get(f'optimal_components_{pop}', 1)
        new_val = st.sidebar.number_input(
            f"Components for {pop}",
            min_value=1,
            max_value=20,
            value=current,
            key=f"sidebar_adj_{pop}"
        )
        st.session_state[f'optimal_components_{pop}'] = new_val
    
    if st.sidebar.button("Regenerate with New Components"):
        if 'syn_data' in st.session_state:
            del st.session_state['syn_data']
        st.rerun()
        
        
def explore_clusters_with_dim_reduction_plots_new_4():
    st.title("Explore Clusters")

    if not st.session_state.get("gated_data"):
        st.warning("Please apply gating first to explore clusters.")
        return

    st.write("Explore dimensionality reduction plots for gated populations and GMM clusters.")

    # Initialize session states
    st.session_state.setdefault("cluster_plots", {})
    st.session_state.setdefault("cluster_stats", {})

    # Sidebar controls
    with st.sidebar.expander("Cluster Exploration Settings", expanded=True):
        st.write("**Data Selection:**")
        use_full_data = st.checkbox("Use Full Data Set", value=False,
                                    help="Check to use the full data set for clustering. \nUncheck to use a representative sample to reduce computation time.",
                                    key="use_full_data")

        st.write("**Clustering Parameters:**")
        max_components = st.number_input(
            "Max GMM Components to test", min_value=1, max_value=20, value=10, key="max_components"
        )
        overlay_gmm = st.checkbox("Overlay GMM Clusters", value=False,
                                  help="Select to overlay GMM clusters on the plot for visual validation. \nUncheck to visualize without GMM overlay.",
                                  key="overlay_gmm")

        dim_reduction_method = st.selectbox("Dimensionality Reduction Method", ["UMAP", "t-SNE"], key="dim_reduction_method")

    # Button to generate new plots
    if st.sidebar.button("Explore Clusters", key="explore_clusters"):
        st.session_state.cluster_plots.clear()
        st.session_state.cluster_stats.clear()

        for pop, df in st.session_state.gated_data.items():
            data = df.values
            feature_names = df.columns.tolist()

            if not use_full_data:
                data = get_representative_sample(data)
                st.sidebar.write(f"Representative Sample Size for {pop}: {len(data)}")
            
            scaled_data = StandardScaler().fit_transform(data)
            
            # Determine optimal GMM components if needed
            if overlay_gmm:
                n_components = optimal_gmm_components(data, max_components)
                gmm = GaussianMixture(n_components=n_components, covariance_type='full')
                clusters = gmm.fit_predict(scaled_data)
                st.write(f"Population: {pop}")
                st.write(f"BIC suggests {n_components} clusters for population '{pop}'")
            else:
                n_components = None
                gmm = None
                clusters = None
            # Update session state with GMM components
            st.session_state[f'optimal_components_{pop}'] = n_components
                
            # Plot
            fig = plot_dim_reduction_new(
                data=data,
                method=dim_reduction_method.lower(),
                n_clusters=n_components,
                feature_names=feature_names
            )

            if fig:
                st.session_state.cluster_plots[pop] = fig
            else:
                st.warning(f"No plot generated for population '{pop}'.")

            # Diagnostics — only run if clustering was applied
            if overlay_gmm and clusters is not None:
                cluster_diagnostics_ui_2(
                    clusters=clusters,
                    gmm=gmm,
                    n_clusters=n_components,
                    feature_names=feature_names,
                    population_id=pop
                )
            else:
                st.write(f"No clustering applied for population '{pop}'.")

        st.sidebar.success("Cluster exploration complete!")
        st.rerun()

    # Display existing plots if they exist
    if st.session_state.cluster_plots:
        for pop, fig in st.session_state.cluster_plots.items():
            st.write(f"### Population: {pop}")
            st.write(f"Optimal GMM Components: {optimal_gmm_components(st.session_state.gated_data[pop].values)}")
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{pop}")
            
            with st.expander(f"Cluster Diagnostics – {pop}", expanded=False):  
                if pop in st.session_state.cluster_stats:
                    stats = st.session_state.cluster_stats[pop]
                    # Check if number of clusters is available and greater than 1
                    number_of_clusters = stats.get('n_components', None)
                    if number_of_clusters is not None and number_of_clusters > 1:
                        st.write(f"**Cluster Statistics (n={stats.get('n_components', '?')})**")
                        st.dataframe(stats.get("df"))
                        means = stats.get("means")
                        if means is not None:
                            st.write("**Mean Marker Expression by Cluster**")
                            st.dataframe(means.style.background_gradient(cmap='viridis'))
    
def visualize_data(df):
    st.write("Generated Data:")
    st.dataframe(df.head())
    st.write("Data Dimensions:", df.shape)
    st.write("Summary Statistics:")
    st.write(df.describe())
    
    unique_pops = df['population'].unique()
    selected_pops = st.multiselect("Filter by Population", unique_pops, default=unique_pops)
    df_filtered = df[df['population'].isin(selected_pops)] #if selected_pops else df
    
    pop_counts = df_filtered['population'].value_counts()
    st.write("Current Population Sizes:")
    st.write(pop_counts.sort_index())
    
    x_axis = st.selectbox("X-Axis", df.columns, index=0 if len(df.columns) > 0 else None, key="x_axis_viz")
    y_axis = st.selectbox("Y-Axis", df.columns, index=1 if len(df.columns) > 1 else None, key="y_axis_viz")
    
    x_log_scale = st.checkbox("Log scale for X-axis", key="x_log_viz")
    y_log_scale = st.checkbox("Log scale for Y-axis", key="y_log_viz")
    
    fig = px.scatter(df_filtered, x=x_axis, y=y_axis, opacity=0.6, color="population")

    # Apply log scale if selected
    if x_log_scale:
        fig.update_layout(xaxis_type="log")
    if y_log_scale:
        fig.update_layout(yaxis_type="log")
    
    st.plotly_chart(fig)

    hist_param = st.selectbox("Select Parameter for Histogram", df.columns, index=0, key="hist_param_viz")
    
    # Axis Scale for Histogram
    hist_log_scale = st.checkbox("Log scale for Histogram", key="hist_log_viz")
    # Create histogram
    fig_hist = px.histogram(df_filtered, x=hist_param, color="population", nbins=50, opacity=0.6)
    
    if hist_log_scale:
            fig_hist.update_layout(xaxis_type="log")
            
    st.plotly_chart(fig_hist)  
    
   
   
def generate_synthetic_data_ui():
    st.title("Generate Synthetic Data")
    # Initialize session state variables if they don't exist
    if 'should_generate' not in st.session_state:
        st.session_state.should_generate = False
    if 'syn_data' not in st.session_state:
        st.session_state.syn_data = pd.DataFrame()  # Initialize as empty DataFrame

    # Sidebar controls - always visible
    with st.sidebar.expander("Synthetic Data Generation Settings", expanded=True):
        method = st.selectbox("Select Method", ["GMM", "Multivariate Normal"], index=0)
        size_factor = st.number_input("Size Factor",
                                      min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                      help="Adjust the size factor to scale the number of samples generated for each population.",
                                      key="size_factor")         
        #st.write("Enable cluster visualization to visualize the clusters in the data")
        if method == "GMM":
            # enable_visual_validation = st.checkbox(
            #     "Enable cluster visualization", 
            #     False,
            #     help="This will help to better understand the distribution of the data and to compare BIC's component count with visual clusters. \n The colors in the plot are based on the GMM clusters.", 
            #     key="enable_visual_validation"
            # )
            
            # Component adjustment UI
            if 'gated_data' in st.session_state:
                st.subheader("Component Adjustment")
                #st.sidebar.write("Adjust the number of components for GMM if needed (default value shown is 1, but will update to the optimal value once the data is generated).")
                for pop in st.session_state.gated_data:
                    # compute optimal components if not already done
                    if f'optimal_components_{pop}' not in st.session_state:
                        # Indicate to the user that the optimal components are being calculated
                        st.toast(f"Calculating optimal components for {pop}...")
                        # Compute optimal components
                        st.session_state[f'optimal_components_{pop}'] = optimal_gmm_components(st.session_state.gated_data[pop].values)
                        
                    current = st.session_state.get(f'optimal_components_{pop}', 1)
                    new_val = st.number_input(
                        f"Components for {pop}",
                        min_value=1,
                        max_value=20,
                        value=current,
                        help = "Adjust the number of components/clusters for GMM if needed. \
                            \n The optimal value is determined based on BIC (Bayesian Information Criterion). \
                                \n You can adjust the number of components for each population separately.\
                                    \n Adjusting the number of components will regenerate the data.",
                        key=f"sidebar_adj_{pop}"
                    )
                    if new_val != current:
                        st.session_state[f'optimal_components_{pop}'] = new_val
                        st.session_state.should_generate = True
        
        if st.button("Generate/Regenerate Data"):
            st.session_state.should_generate = True

    # Generate data when requested
    if st.session_state.should_generate and 'gated_data' in st.session_state:
        try:
            # st.session_state.syn_data = generate_synthetic_data(
            #     size_factor=size_factor,
            #     method=method.lower(),
            #     enable_visual_validation=(method == "GMM" and enable_visual_validation)
            # )
            st.session_state.syn_data = generate_synthetic_data_2(
                size_factor=size_factor,
                method=method.lower(),
            )
            st.success("Data generated successfully!")
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")
        finally:
            st.session_state.should_generate = False

    # Only show data and visualization if we have non-empty data
    if not st.session_state.syn_data.empty:
        st.write("### Synthetic Data Preview")
        st.dataframe(st.session_state.syn_data.head())
        st.write("Data Dimensions:", st.session_state.syn_data.shape)
        st.write("Population counts", st.session_state.syn_data['population'].value_counts())
        #st.write("Summary Statistics:")
        #st.write(st.session_state.syn_data.describe())
        
        # Visualization
        st.write("### Data Visualization")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-Axis", st.session_state.syn_data.columns, index=0)
            log_x = st.checkbox("Log scale (X)")

        with col2:
            y_axis = st.selectbox("Y-Axis", st.session_state.syn_data.columns, index=1)
            log_y = st.checkbox("Log scale (Y)")
        
        fig = px.scatter(st.session_state.syn_data, x=x_axis, y=y_axis)
        if log_x:
            fig.update_xaxes(type="log")
        if log_y:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig)
    
        
    # Evaluation options
    #st.sidebar.write("### Data Quality Evaluation")
    # Select maximum number of features to visualize
    #st.sidebar.write("Select maximum number of features to visualize in the evaluation plots.")
    #max_features = st.sidebar.number_input("Max Features to Visualize", min_value=1, max_value=10, value=3, key="max_features")
    
    #data_quality_metrics_previous_ui()  # placeholder for previous UI code use with commented out evaluate_synthetic_data function (i.e. evaluate_synthetic_data previous)
    data_quality_metrics_ui()  # placeholder for new UI code use with new evaluate_synthetic_data function (i.e. evaluate_synthetic_data new)
    show_evaluation_results()
   
   
   
   
    
def edit_populations_ui():
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    if 'syn_data' in st.session_state and not st.session_state.syn_data.empty:
        df = st.session_state.syn_data
        visualize_data(df)
        
        with st.sidebar.expander("Balance Populations"):
            desired_sizes = {}
            sorted_population = df['population'].value_counts().sort_index()
            for pop in sorted_population.index:
                desired_sizes[pop] = st.number_input(
                    f"Desired size for {pop}", 
                    value=int(sorted_population[pop]),
                    min_value=0, 
                    max_value=1000_000, 
                    key=f"desired_size_{pop}"
                )
                
            balancing_technique = st.selectbox(
                "Select Balancing Technique", 
                ["SMOTE", "ADASYN", "Random Over Sampler", "Random Under Sampler"], 
                key="balancing_technique"
            )

            if balancing_technique in ['SMOTE', 'ADASYN']:
                k_neighbors = st.number_input(
                    "Number of Neighbors", 
                    value=5, 
                    min_value=1, 
                    max_value=20, 
                    key="k_neighbors"
                )
            
            #total_percentage = sum(desired_sizes.values()) 
            #st.write(f"Total Desired Size Percentage: {total_percentage}%")
            
            #if total_percentage != 100:
            #    st.warning("Total desired size percentage must equal 100% for balancing to work correctly.")           
            
            if st.button("Balance Data", key="balance_data"):
                try:
                    # Initialize with original data
                    X_resampled = df.drop(columns=['population'])
                    y_resampled = df['population']
                    
                    original_sizes = {pop: df[df['population'] == pop].shape[0] for pop in df['population'].unique()}
                    
                    # Create scaled_df with safe sampling
                    scaled_samples = []
                    for pop in df['population'].unique():
                        pop_df = df[df['population'] == pop]
                        target_size = int(desired_sizes[pop])
                        current_size = pop_df.shape[0]
                        
                        # Safe sampling logic
                        if target_size > current_size:
                            # For oversampling, use replacement
                            sample = pop_df.sample(
                                n=target_size,
                                replace=True,
                                random_state=42
                            )
                        else:
                            # For undersampling, no replacement needed
                            sample = pop_df.sample(
                                n=target_size,
                                replace=False,
                                random_state=42
                            )
                        scaled_samples.append(sample)
                    
                    scaled_df = pd.concat(scaled_samples)
                    
                    X = scaled_df.drop(columns=['population'])
                    y = scaled_df['population']
                    
                    oversample_targets = {
                        pop: int(desired_sizes[pop]) 
                        for pop in df['population'].unique() 
                        if desired_sizes[pop] > original_sizes[pop]
                    }
                    undersample_targets = {
                        pop: int(desired_sizes[pop]) 
                        for pop in df['population'].unique() 
                        if desired_sizes[pop] < original_sizes[pop]
                    }
                    
                    # Apply oversampling if selected
                    if balancing_technique in ['SMOTE', 'ADASYN'] and oversample_targets:
                        if balancing_technique == 'SMOTE':
                            st.toast(f"Applying SMOTE to oversample: {oversample_targets}")
                            smote = SMOTE(
                                sampling_strategy=oversample_targets, 
                                k_neighbors=k_neighbors, 
                                random_state=42
                            )
                            X_resampled, y_resampled = smote.fit_resample(X, y)
                        elif balancing_technique == 'ADASYN':
                            st.toast(f"Applying ADASYN to oversample: {oversample_targets}")
                            adasyn = ADASYN(
                                sampling_strategy=oversample_targets, 
                                random_state=42
                            )
                            X_resampled, y_resampled = adasyn.fit_resample(X, y)
                    
                    # Apply undersampling if selected
                    elif balancing_technique == 'Random Under Sampler' and undersample_targets:
                        st.toast(f"Applying Random Under Sampler to undersample: {undersample_targets}")
                        rus = RandomUnderSampler(
                            sampling_strategy=undersample_targets, 
                            random_state=42
                        )
                        X_resampled, y_resampled = rus.fit_resample(X, y)
                    
                    # Create balanced dataframe
                    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
                    balanced_df['population'] = y_resampled
                    
                    # Update session state
                    st.session_state.syn_data = balanced_df
                    
                    # Log population sizes
                    new_sizes = {
                        pop: balanced_df[balanced_df['population'] == pop].shape[0] 
                        for pop in balanced_df['population'].unique()
                    }
                    
                    log_df = pd.DataFrame({
                        'Population': list(original_sizes.keys()),
                        'Original Size': list(original_sizes.values()),
                        'New Size': list(new_sizes.values())
                    })
                    
                    # After successful balancing:
                    st.session_state.syn_data = balanced_df
                    st.session_state.log_df = log_df
                    
                    st.success("Data balanced successfully!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error during balancing: {str(e)}")
            
            if 'log_df' in st.session_state:
                with st.sidebar.expander("Population Size Changes", expanded=True):
                    st.write("### Changes Summary")
                    st.table(st.session_state.log_df)
    else:
        st.warning("No synthetic data available. Please Generate data first.")



# Function to evaluate synthetic data quality with memory safety and structured results updated for better error handling
def evaluate_synthetic_data(original_data, synthetic_data, sample_size=10000, population_name=None):
    """Enhanced evaluation function with better error handling"""
    results = {
        'population': population_name,
        'metrics': {
            'silhouette': float('nan'),
            'mutual_info': float('nan'),
            'wasserstein': float('nan')
        },
        'timestamp': datetime.now().isoformat(),
        'synthetic_data': synthetic_data  # Store the actual data for visualization
    }
    
    try:
        # Downsample if needed
        if len(original_data) > sample_size:
            rng = np.random.default_rng(42)
            orig_idx = rng.choice(len(original_data), size=sample_size, replace=False)
            synth_idx = rng.choice(len(synthetic_data), size=sample_size, replace=False)
            original_sample = original_data[orig_idx]
            synthetic_sample = synthetic_data[synth_idx]
        else:
            original_sample = original_data
            synthetic_sample = synthetic_data
        
        # Silhouette Score
        combined = np.vstack([original_sample, synthetic_sample])
        labels = np.array([0]*len(original_sample) + [1]*len(synthetic_sample))
        results['metrics']['silhouette'] = silhouette_score(combined, labels)
        
        # Mutual Information
        try:
            mi_scores = []
            for i in range(original_sample.shape[1]):
                hist_2d = np.histogram2d(original_sample[:, i], synthetic_sample[:, i], bins=5)[0]
                mi = mutual_info_score(None, None, contingency=hist_2d)
                mi_scores.append(mi)
            results['metrics']['mutual_info'] = np.mean(mi_scores)
        except Exception as e:
            st.warning(f"Mutual info calculation failed: {str(e)}")
        
        # Wasserstein Distance
        try:
            wasserstein_scores = []
            for i in range(original_sample.shape[1]):
                wasserstein_scores.append(wasserstein_distance(original_sample[:, i], synthetic_sample[:, i]))
            results['metrics']['wasserstein'] = np.mean(wasserstein_scores)
        except Exception as e:
            st.warning(f"Wasserstein distance calculation failed: {str(e)}")
            
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
    
    return results


def interactive_distribution_plot(original, synthetic, population):
    import plotly.express as px
    import pandas as pd
    
    # Create combined dataframe
    df_orig = pd.DataFrame(original, columns=[f"F{i+1}" for i in range(original.shape[1])])
    df_orig['Type'] = 'Original'
    df_synth = pd.DataFrame(synthetic, columns=[f"F{i+1}" for i in range(synthetic.shape[1])])
    df_synth['Type'] = 'Synthetic'
    df = pd.concat([df_orig, df_synth])
    
    # Create interactive figure
    fig = px.histogram(df, facet_col='variable', facet_col_wrap=4,
                      color='Type', barmode='overlay',
                      opacity=0.7, nbins=30,
                      title=f"Distribution Comparison: {population}")
    
    fig.update_layout(height=800)
    return fig



# Enhanced version of plot_distribution_comparison function
def plot_distribution_comparison(original, synthetic, population, max_features_per_row=4, feature_names=False):
    """Enhanced distribution comparison plot with better visualization."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_features = original.shape[1]
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(n_features / max_features_per_row))
    n_cols = min(n_features, max_features_per_row)
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, 
                           figsize=(4*n_cols, 3*n_rows),
                           squeeze=False)
    
    # If feature names are provided, use them; otherwise, generate default names
    if feature_names:
        feature_names = feature_names
    else:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

    # Plot each feature
    for i, ax in enumerate(axes.flat):
        if i < n_features:
            # Plot histograms with improved styling
            ax.hist(original[:, i], bins=30, alpha=0.6, label='Original', 
                   density=True, color='#1f77b4', edgecolor='none')
            ax.hist(synthetic[:, i], bins=30, alpha=0.6, label='Synthetic',
                   density=True, color='#ff7f0e', edgecolor='none')
            
            # Add KDE for smoother visualization
            try:
                from scipy.stats import gaussian_kde
                kde_orig = gaussian_kde(original[:, i])
                kde_synth = gaussian_kde(synthetic[:, i])
                x = np.linspace(min(original[:, i].min(), synthetic[:, i].min()),
                               max(original[:, i].max(), synthetic[:, i].max()),
                               100)
                ax.plot(x, kde_orig(x), color='#1f77b4', linewidth=2)
                ax.plot(x, kde_synth(x), color='#ff7f0e', linewidth=2)
            except:
                pass
            
            # Customize appearance
            ax.set_title(feature_names[i], pad=10, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Only show legend on first plot
            if i == 0:
                ax.legend(frameon=False)
        else:
            ax.axis('off')  # Hide empty subplots
    
    # Add overall title and adjust layout
    plt.suptitle(f"Distribution Comparison: {population}", y=1.02, fontsize=12)
    plt.tight_layout()
    
    return fig    


def run_evaluation(sample_size):
    if st.session_state.syn_data is not None:
        # Clear previous results at the start of each new evaluation
        st.session_state.evaluation_results = []
        
        # Perform and store evaluations
        if 'population' in st.session_state.syn_data.columns:
            unique_pops = st.session_state.syn_data['population'].unique()
            for pop in unique_pops:
                if pop in st.session_state.gated_data:
                    original_data = st.session_state.gated_data[pop].values
                    synthetic_pop = st.session_state.syn_data[st.session_state.syn_data['population'] == pop]
                    
                    if len(synthetic_pop) > 0:
                        eval_result = evaluate_synthetic_data(
                            original_data,
                            synthetic_pop.drop('population', axis=1).values,
                            sample_size=sample_size,
                            population_name=pop
                        )
                        st.session_state.evaluation_results.append(eval_result)
        else:
            eval_result = evaluate_synthetic_data(
                np.concatenate([d.values for d in st.session_state.gated_data.values()]),
                st.session_state.syn_data.values,
                sample_size=sample_size,
                population_name="All Populations"
            )
            st.session_state.evaluation_results.append(eval_result)
        
        #show_evaluation_results()  
        
        
                     
def data_quality_metrics_ui():
    # Check if gated data is available
    if not st.session_state.gated_data or st.session_state.gated_data == {}:
        st.warning("Please apply gating first to evaluate data quality.")
        return
    
    with st.sidebar.expander("📊 Data Quality Evaluation Settings", expanded=True):
        # Section header
        st.write("**Visualization Settings**")
        
        # Max features per row setting
        max_features = st.number_input(
            "Max Features to Visualize per Row", 
            min_value=1, 
            max_value=10, 
            value=4,
            key="max_features_per_row"
        )
        
        # Evaluation sample size setting
        sample_size = st.number_input(
            "Evaluation Sample Size", 
            min_value=1000, 
            max_value=50000, 
            value=10000,
            key="eval_sample_size"
        )
        
        # Evaluation method options
        st.write("**Evaluation Options**")
        enable_silhouette = st.checkbox("Silhouette Score", value=True)
        enable_mutual_info = st.checkbox("Mutual Information", value=True)
        enable_wasserstein = st.checkbox("Wasserstein Distance", value=True)
        
        # Action buttons - using markdown for layout instead of columns
        st.write("**Actions**")
        
        # First button with full width
        if st.button("🔍 Evaluate", 
                    help="Run data quality evaluation",
                    key="show_evaluation"):
            run_evaluation(sample_size)
        
        # Second button with full width
        if st.button("🗑️ Clear Results",
                    help="Clear all evaluation results",
                    key="clear_evaluation"):
            st.session_state.evaluation_results = []
            st.success("Cleared all evaluation results")

def show_evaluation_results():
    if st.session_state.evaluation_results:
        # Main page container for all results
        #with st.container():
        st.write("## Data Quality Evaluation Results")
        
        # Create tabs for each population's results
        tabs = st.tabs([f"{result['population']}" for result in st.session_state.evaluation_results])
        
        for i, result in enumerate(st.session_state.evaluation_results):
            with tabs[i]:
                # Metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Silhouette Score",
                        f"{result['metrics'].get('silhouette', float('nan')):.3f}",
                        help="Means separation between original and synthetic \
                            \n (-1 to 1, closer to 0 is better) \
                                \n (0 = overlapping, 1 = well separated) \
                                    \n Mean score across all features is shown"
                    )
                
                with col2:
                    st.metric(
                        "Mutual Information", 
                        f"{result['metrics'].get('mutual_info', float('nan')):.3f}",
                        help="Statistical dependency between distributions (higher is better)\
                            \n (0 = independent, 1 = identical)\
                                \n Mean MI across all features is shown"
                    )
                
                with col3:
                    st.metric(
                        "Wasserstein Distance",
                        f"{result['metrics'].get('wasserstein', float('nan')):.3f}",
                        help="Distance between distributions (lower is better) \
                            \n (0 = identical, >0 = different) \
                                \n Mean distance across all features is shown"
                    )
                
                # Distribution plot
                if 'synthetic_data' in result:
                    try:
                        fig = plot_distribution_comparison(
                            np.concatenate([d.values for d in st.session_state.gated_data.values()]),
                            result['synthetic_data'],
                            result['population'],
                            max_features_per_row=st.session_state.get('max_features_per_row', 4),
                            feature_names=st.session_state.syn_data.columns.tolist()
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not generate distribution plot: {str(e)}")
                else:
                    st.warning("No synthetic data available for visualization")
        
        # Add spacing at bottom
        st.write("")  



def edit_populations_ui():
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    
    if 'syn_data' in st.session_state and not st.session_state.syn_data.empty:
        df = st.session_state.syn_data
        visualize_data(df)
        
        with st.sidebar.expander("Balance Populations"):
            desired_sizes = {}
            sorted_population = df['population'].value_counts().sort_index()
            for pop in sorted_population.index:
                desired_sizes[pop] = st.number_input(
                    f"Desired size for {pop}", 
                    value=int(sorted_population[pop]),
                    min_value=0, 
                    max_value=1000_000, 
                    key=f"desired_size_{pop}"
                )
                
            balancing_technique = st.selectbox(
                "Select Balancing Technique", 
                ["SMOTE", "ADASYN", "Random Over Sampler", "Random Under Sampler"], 
                help ="Select the technique to balance the populations. \
                    \n SMOTE and ADASYN are used for oversampling, while Random Under Sampler is used for undersampling.\
                        \n SMOTE and ADASYN require the number of neighbors to be specified. \
                            \n ADASYN performs well on generating synthetic samples in areas where the classification is difficult, improving the decision boundary.",
                key="balancing_technique"
            )

            if balancing_technique in ['SMOTE', 'ADASYN']:
                k_neighbors = st.number_input(
                    "Number of Neighbors", 
                    value=5, 
                    min_value=1, 
                    max_value=20, 
                    help = "Number of neighbors to use for SMOTE/ADASYN. \
                        \n This parameter controls the number of nearest neighbors used to generate synthetic samples. \
                            \n A higher value may lead to more diverse samples, but can also introduce noise. \
                                \n A lower value may lead to less diversity but more accurate samples.",
                    key="k_neighbors"
                )
            
            #total_percentage = sum(desired_sizes.values()) 
            #st.write(f"Total Desired Size Percentage: {total_percentage}%")
            
            #if total_percentage != 100:
            #    st.warning("Total desired size percentage must equal 100% for balancing to work correctly.")           
            
            if st.button("Balance Data", key="balance_data"):
                try:
                    # Initialize with original data
                    X_resampled = df.drop(columns=['population'])
                    y_resampled = df['population']
                    
                    original_sizes = {pop: df[df['population'] == pop].shape[0] for pop in df['population'].unique()}
                    
                    # Create scaled_df with safe sampling
                    scaled_samples = []
                    for pop in df['population'].unique():
                        pop_df = df[df['population'] == pop]
                        target_size = int(desired_sizes[pop])
                        current_size = pop_df.shape[0]
                        
                        # Safe sampling logic
                        if target_size > current_size:
                            # For oversampling, use replacement
                            sample = pop_df.sample(
                                n=target_size,
                                replace=True,
                                random_state=42
                            )
                        else:
                            # For undersampling, no replacement needed
                            sample = pop_df.sample(
                                n=target_size,
                                replace=False,
                                random_state=42
                            )
                        scaled_samples.append(sample)
                    
                    scaled_df = pd.concat(scaled_samples)
                    
                    X = scaled_df.drop(columns=['population'])
                    y = scaled_df['population']
                    
                    oversample_targets = {
                        pop: int(desired_sizes[pop]) 
                        for pop in df['population'].unique() 
                        if desired_sizes[pop] > original_sizes[pop]
                    }
                    undersample_targets = {
                        pop: int(desired_sizes[pop]) 
                        for pop in df['population'].unique() 
                        if desired_sizes[pop] < original_sizes[pop]
                    }
                    
                    # Apply oversampling if selected
                    if balancing_technique in ['SMOTE', 'ADASYN'] and oversample_targets:
                        if balancing_technique == 'SMOTE':
                            st.toast(f"Applying SMOTE to oversample: {oversample_targets}")
                            smote = SMOTE(
                                sampling_strategy=oversample_targets, 
                                k_neighbors=k_neighbors, 
                                random_state=42
                            )
                            X_resampled, y_resampled = smote.fit_resample(X, y)
                        elif balancing_technique == 'ADASYN':
                            st.toast(f"Applying ADASYN to oversample: {oversample_targets}")
                            adasyn = ADASYN(
                                sampling_strategy=oversample_targets, 
                                random_state=42
                            )
                            X_resampled, y_resampled = adasyn.fit_resample(X, y)
                    
                    # Apply undersampling if selected
                    elif balancing_technique == 'Random Under Sampler' and undersample_targets:
                        st.toast(f"Applying Random Under Sampler to undersample: {undersample_targets}")
                        rus = RandomUnderSampler(
                            sampling_strategy=undersample_targets, 
                            random_state=42
                        )
                        X_resampled, y_resampled = rus.fit_resample(X, y)
                    
                    # Create balanced dataframe
                    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
                    balanced_df['population'] = y_resampled
                    
                    # Update session state
                    st.session_state.syn_data = balanced_df
                    
                    # Log population sizes
                    new_sizes = {
                        pop: balanced_df[balanced_df['population'] == pop].shape[0] 
                        for pop in balanced_df['population'].unique()
                    }
                    
                    log_df = pd.DataFrame({
                        'Population': list(original_sizes.keys()),
                        'Original Size': list(original_sizes.values()),
                        'New Size': list(new_sizes.values())
                    })
                    
                    # After successful balancing:
                    st.session_state.syn_data = balanced_df
                    st.session_state.log_df = log_df
                    
                    st.success("Data balanced successfully!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error during balancing: {str(e)}")
            
            if 'log_df' in st.session_state:
                with st.sidebar.expander("Population Size Changes", expanded=True):
                    st.write("### Changes Summary")
                    st.table(st.session_state.log_df)
    else:
        st.warning("No synthetic data available. Please Generate data first.")
        

def dataframe_to_fcs(df, filename="synthetic_data.fcs"):
    """Convert DataFrame to FCS format, handling non-numeric columns"""
    # Ensure we only include numeric data columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found for FCS export")
    
    # Convert numeric data to numpy array
    data = df[numeric_cols].values.astype(np.float32)  # FCS requires float32
    
    # Create temporary FCS file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fcs") as tmp_file:
        fcswrite.write_fcs(
            filename=tmp_file.name,
            chn_names=numeric_cols,  # Only numeric channel names
            data=data
        )
        
        # Read the temporary file into memory
        with open(tmp_file.name, "rb") as f:
            fcs_bytes = io.BytesIO(f.read())
    
    # Clean up temporary file
    os.remove(tmp_file.name)
    return fcs_bytes

def download_synthetic_data_ui():
    st.title("Download Synthetic Data")    
    if 'syn_data' in st.session_state and not st.session_state.syn_data.empty:
        st.write("### Synthetic Data Preview")
        st.dataframe(st.session_state.syn_data.head())
        
        with st.sidebar.expander("Download Options", expanded=True):
            # CSV Download
            csv_buffer = io.BytesIO()
            st.session_state.syn_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="Download as CSV",
                data=csv_buffer,
                file_name="synthetic_data.csv",
                mime="text/csv"
            )
            
            # FCS Download
            try:
                fcs_buffer = dataframe_to_fcs(st.session_state.syn_data.drop(columns=['population'], errors='ignore'))
                st.download_button(
                    label="Download as FCS",
                    data=fcs_buffer.getvalue(),
                    file_name="synthetic_data.fcs",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Could not create FCS file: {str(e)}")
    else:
        st.warning("No synthetic data available. Please Generate data first.")

def download_fcs_sample():
    # Define file path
    file_path = "data/ABCDEFGHIJ-merged.fcs" 
    # Check if file exists
    if not os.path.exists(file_path):
        st.error("File not found. Please check the file path.")
        return None
    # Read the file
    with open(file_path, "rb") as file:
        fcs_file = file.read()
        
    # Display the download button
    st.download_button(label="Download Sample .fcs File", data=fcs_file, file_name="ABCDEFGHIJ-merged.fcs",
                       mime="application/octet-stream",
                       key="download_fcs")


def about_the_gating_app():
    st.title(" Synthetic Flow Cytometry Data Generator")
    st.markdown("""
                Welcome to the **Synthetic Flow Cytometry Data Generator**!  
                - This app allows you to generate synthetic flow cytometry data from an existing `.fcs` file, using Gaussian Mixture Model (GMM), or Multivariate Normal distribution.
                - It allows you to apply gating to the data to extract specific populations and generate synthetic data based on the selected populations.
                - The app will display a preview of the data and allow you to select points for gating.
                
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
                
                - **Dimensionality Reduction and Explore GMM Clusters**
                    - Explore the clusters in the gated data by selecting the "Explore Clusters" option.
                    - It will provide a visual validation of the clusters in the gated populations using dimensionality reduction techniques, uMAP or t-SNE.
                    - Clusters detected using GMM can be overlaid on the dimensionality reduction plot to visualize the clusters.
                    - The number of clusters are determined using BIC (Bayesian Information Criterion), which is a model selection criterion.
                    - In this case the BIC is used to select the number of clusters that best fits the data.
                    
                - **Generate Synthetic Data**
                    - Generate synthetic data using GMM or Multivariate Normal by selecting the "Generate Synthetic Data" option.
                    - The synthetic data will be generated based on the selected populations and displayed in a table.
                    - Visualize the synthetic data by selecting the X-axis and Y-axis columns from the dropdown menu.
                    - Download the synthetic data as a CSV or FCS file by selecting the download button.
                    - Tips: 
                        - Use Multivariate Normal if:    
                            - Your gated populations are clearly unimodal.    
                            - You need rapid generation (e.g., for quick previews).    
                        - Use GMM if:    
                            - You suspect subpopulations exist within gates.    
                        - Size Factor:
                            Adjust the size factor to scale the number of samples generated for each population.  
                    - Evaluate the quality of the synthetic data using metrics like Silhouette Score, Mutual Information, and Wasserstein Distance.
                    
                - **Edit Populations**
                    - Edit the populations by selecting the "Edit Populations" option.
                    - It will allow you to balance the populations using SMOTE, ADASYN, RandomOverSampler, or RandomUnderSampler.
                    - Use SMOTE or ADASYN or RandomOverSampler for oversampling, and RandomUnderSampler for undersampling.
                    - SMOTE uses nearest neighbors to create synthetic samples, while ADASYN focuses on difficult-to-learn samples.
                    - RandomOverSampler duplicates existing samples, and RandomUnderSampler randomly removes samples.
                    
                - **Download Configuration File**
                    - Download the gating configuration as a `.json` file by selecting the "Download Configuration File" option.
                    - The gating configuration will be saved as a `.json` file with the statistical properties of the gated populations.
                
                *Explore the sidebar for different sections and detailed functionalities!* 
                """)
    
    
# Streamlit UI

st.sidebar.title("Flow Cytometry Data Analyzer")

if st.sidebar.button("Reset Session State"):
    st.warning("⚠️ This will delete ALL data. Are you sure?")
    if st.checkbox("Yes, reset everything"):
        reset_session_state()
        st.success("Session reset complete!")  # Optional feedback
        
        
        
# Navigation menu
option = st.sidebar.radio("Select an option:",
                          ("About the app","Upload an .fcs file", "Apply Gating", "Download Configuration File",
                           "Dimensionality Reduction and Explore GMM Clusters", "Generate Synthetic Data","Edit Populations","Download Synthetic Data")
                            )

if option == "About the app":
    about_the_gating_app()
    download_fcs_sample()
    
elif option == "Upload an .fcs file":
    upload_fcs_file()
    
elif option == "Apply Gating":
    apply_gating()
    
elif option == "Dimensionality Reduction and Explore GMM Clusters":
    explore_clusters_with_dim_reduction_plots_new_4()

elif option == "Download Configuration File":
    download_config()
    
elif option == "Generate Synthetic Data":
    generate_synthetic_data_ui()
    
elif option == "Edit Populations":
    edit_populations_ui()
    
elif option == "Download Synthetic Data":
    download_synthetic_data_ui()
    
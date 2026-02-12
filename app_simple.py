"""
DataPilot AI Pro - Streamlit Web Application (Simplified)
=========================================================

Simple web interface for the autonomous data science platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from datapilot import DataPilot

# Configure page
st.set_page_config(
    page_title="DataPilot AI Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'pilot' not in st.session_state:
    st.session_state.pilot = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None


def render_header():
    """Render application header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🚀 DataPilot AI Pro")
        st.markdown("### Autonomous Data Science Platform")
        st.markdown("*End-to-end ML pipeline with 6 specialized agents*")
    with col2:
        st.info(f"📅 {datetime.now().strftime('%B %d, %Y')}")


def render_upload_section():
    """Render file upload section."""
    st.subheader("📂 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file containing your dataset"
    )
    
    return uploaded_file


def render_dataset_preview(df: pd.DataFrame):
    """Render dataset preview."""
    st.subheader("📊 Dataset Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features", f"{df.shape[1]:,}")
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.dataframe(df.head(10), use_container_width=True)


def render_target_selection(df: pd.DataFrame) -> str:
    """Render target column selection."""
    st.subheader("🎯 Target Selection")
    
    target_column = st.selectbox(
        "Select target column",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
        help="The column you want to predict"
    )
    
    return target_column


def render_results(results: dict):
    """Render pipeline results."""
    st.subheader("📈 Pipeline Results")
    
    # Create tabs for different reports
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Profile",
        "🧹 Cleaning",
        "⚙️ Features",
        "🎯 Models",
        "📈 Visualizations",
        "💡 Explanations"
    ])
    
    with tab1:
        if 'profile' in results:
            st.write("**Dataset Profile**")
            col1, col2, col3 = st.columns(3)
            with col1:
                shape = results['profile']['dataset_shape']
                st.metric("Shape", f"{shape[0]} x {shape[1]}")
            with col2:
                st.metric("Task Type", results['task_type'].upper())
            with col3:
                st.metric("Numeric Cols", results['profile']['numeric_cols'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Categorical Cols", results['profile']['categorical_cols'])
            with col2:
                st.metric("Missing Ratio", f"{results['profile']['missing_ratio']:.2%}")
            with col3:
                st.metric("Duplicates", results['profile']['duplicates'])
    
    with tab2:
        if 'cleaning' in results:
            st.write("**Data Cleaning Report**")
            col1, col2, col3 = st.columns(3)
            with col1:
                orig_shape = results['cleaning']['original_shape']
                st.metric("Original Shape", f"{orig_shape[0]} x {orig_shape[1]}")
            with col2:
                final_shape = results['cleaning']['final_shape']
                st.metric("Final Shape", f"{final_shape[0]} x {final_shape[1]}")
            with col3:
                st.metric("Rows Removed", results['cleaning']['rows_removed'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duplicates Removed", results['cleaning']['duplicates_removed'])
            with col2:
                st.metric("Missing Ratio", f"{results['cleaning']['missing_ratio']:.2%}")
            with col3:
                st.metric("Outlier Ratio", f"{results['cleaning']['outlier_ratio']:.2%}")
    
    with tab3:
        if 'features' in results:
            st.write("**Feature Engineering**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Features", results['features']['original_features'])
            with col2:
                st.metric("Final Features", results['features']['final_features'])
    
    with tab4:
        if 'modeling' in results:
            st.write("**Model Training Results**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Trained", len(results['modeling']['all_models_trained']))
            with col2:
                st.metric("Ensemble Score", f"{results['modeling']['ensemble_score']:.4f}")
            with col3:
                st.metric("RL Used", "Yes" if results['modeling']['used_rl'] else "No")
            
            st.write("**Top 3 Models**")
            for model_name in results['modeling']['top_3_models']:
                score = results['modeling']['model_scores'].get(model_name, 0)
                st.write(f"• {model_name}: {score:.4f}")
            
            st.write("**Metrics**")
            for metric, value in results['modeling']['metrics'].items():
                st.write(f"• {metric}: {value:.4f}")
            
            st.write("**All Models Trained**")
            with st.expander("View all models and scores"):
                for model_name, score in sorted(results['modeling']['model_scores'].items(), 
                                               key=lambda x: x[1], reverse=True):
                    st.write(f"• {model_name}: {score:.4f}")
    
    with tab5:
        if 'visualization' in results:
            st.write("**Generated Visualizations**")
            st.write(f"Total plots: {results['visualization']['total_plots']}")
            
            # Display all figures
            if 'figures' in results['visualization']:
                for plot_name, fig in results['visualization']['figures'].items():
                    st.write(f"**{plot_name.replace('_', ' ').title()}**")
                    st.pyplot(fig)
            else:
                st.write(f"Plots: {', '.join(results['visualization']['plots_generated'])}")
    
    with tab6:
        if 'explanation' in results:
            st.write("**Model Metrics**")
            if 'metrics' in results['explanation']:
                for metric, value in results['explanation']['metrics'].items():
                    st.metric(metric.upper(), f"{value:.4f}")
            
            st.write("**Top 10 Features**")
            for feat, imp in results['explanation']['top_features'][:10]:
                st.write(f"• {feat}: {imp:.4f}")
            
            st.write("**Insights**")
            for insight in results['explanation']['insights']:
                st.write(f"• {insight}")
            
            st.write("**Recommendations**")
            for rec in results['explanation']['recommendations']:
                st.write(f"• {rec}")


def main():
    """Main application entry point."""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Menu")
        app_mode = st.radio(
            "Select mode",
            options=["Upload & Analyze", "View Results"],
            help="Choose between uploading data or viewing results"
        )
    
    # Main content
    if app_mode == "Upload & Analyze":
        # Upload section
        uploaded_file = render_upload_section()
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            render_dataset_preview(df)
            
            # Target selection
            target_column = render_target_selection(df)
            
            # Run pipeline
            if st.button("🚀 Run Pipeline", use_container_width=True, type="primary"):
                with st.spinner("Processing... This may take a few minutes"):
                    try:
                        pilot = DataPilot()
                        results = pilot.run(df, target_column)
                        st.session_state.pilot = pilot
                        st.session_state.results = results
                        
                        st.success("✅ Pipeline completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Pipeline error: {str(e)}")
                        st.error(traceback.format_exc())
    
    elif app_mode == "View Results":
        if st.session_state.results is not None:
            render_results(st.session_state.results)
            
            # Export section
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Report", use_container_width=True):
                    report_text = "DataPilot AI Pro - Final Report\n"
                    report_text += "="*60 + "\n\n"
                    
                    if st.session_state.pilot:
                        report_text += st.session_state.pilot.get_summary()
                    
                    st.download_button(
                        "Download Report",
                        report_text,
                        "datapilot_report.txt",
                        "text/plain"
                    )
            
            with col2:
                if st.session_state.df is not None and st.button("📊 Export Data", use_container_width=True):
                    csv_data = st.session_state.df.to_csv(index=False)
                    st.download_button(
                        "Download Data",
                        csv_data,
                        "datapilot_data.csv",
                        "text/csv"
                    )
        else:
            st.info("👈 Upload a dataset and run the pipeline to see results")


if __name__ == "__main__":
    main()

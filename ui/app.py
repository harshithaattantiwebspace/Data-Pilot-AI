# ui/app.py

"""
DataPilot AI Pro — Streamlit Web Interface (Redesigned).

SINGLE-PAGE FLOW for managers:
  1. Upload CSV at the top
  2. See data preview
  3. Type a prompt (optional)
  4. Two buttons side-by-side: "Find Insights" | "ML Prediction"
  5. Results displayed below

Usage:
    streamlit run ui/app.py
"""

import os
import sys
import time
from typing import Dict
import pandas as pd
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# LAZY IMPORTS — heavy ML libraries are only loaded when a button is clicked,
# so the page itself loads in ~2 seconds instead of 30+.
_lazy_loaded = {}

def _get_pipeline_functions():
    """Lazy-load the orchestrator (which imports all agents + ML libraries)."""
    if 'orchestrator' not in _lazy_loaded:
        from orchestrator.graph import run_ml_pipeline, run_data_analysis, run_full_pipeline
        _lazy_loaded['orchestrator'] = {
            'run_ml_pipeline': run_ml_pipeline,
            'run_data_analysis': run_data_analysis,
            'run_full_pipeline': run_full_pipeline,
        }
    return _lazy_loaded['orchestrator']

def _get_analyzer():
    """Lazy-load the DataAnalyzerAgent."""
    if 'analyzer' not in _lazy_loaded:
        from agents.data_analyzer import DataAnalyzerAgent
        _lazy_loaded['analyzer'] = DataAnalyzerAgent
    return _lazy_loaded['analyzer']


# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="DataPilot AI Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.15em;
        text-align: center;
        margin-bottom: 30px;
    }
    .success-box {
        background: #D1FAE5;
        border-left: 4px solid #059669;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-box {
        background: #DBEAFE;
        border-left: 4px solid #2563EB;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warn-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .upload-zone {
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        background: #F8FAFC;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    div[data-testid="stHorizontalBlock"] > div { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SESSION STATE
# =========================================================================

for key in ['uploaded_df', 'file_name', 'pipeline_result', 'analysis_result',
            'prompt_result', 'active_view']:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================================================================
# HELPER: Display charts
# =========================================================================

def display_charts(visuals: Dict, section_name: str):
    """Display a dict of Plotly figures in a 2-column grid."""
    if not visuals:
        st.info(f"No {section_name} charts available.")
        return

    flat_charts = {}
    for key, value in visuals.items():
        if isinstance(value, dict):
            for chart_name, fig in value.items():
                if hasattr(fig, 'to_json'):
                    flat_charts[f"{key}/{chart_name}"] = fig
        elif hasattr(value, 'to_json'):
            flat_charts[key] = value

    if not flat_charts:
        st.info(f"No displayable charts in {section_name}.")
        return

    chart_items = list(flat_charts.items())
    for i in range(0, len(chart_items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(chart_items):
                name, fig = chart_items[i + j]
                with col:
                    try:
                        st.plotly_chart(fig, use_container_width=True,
                                       key=f"{section_name}_{name}_{i}_{j}")
                    except Exception as e:
                        st.warning(f"Could not display chart '{name}': {e}")


# =========================================================================
# HEADER
# =========================================================================

st.markdown('<div class="main-header">DataPilot AI Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Upload your dataset, type what you want to know, and choose your action.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================================
# STEP 1 — UPLOAD CSV
# =========================================================================

st.markdown("### 1️⃣  Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Drop a CSV file here",
    type=['csv'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df
        st.session_state.file_name = uploaded_file.name
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

df = st.session_state.uploaded_df

if df is None:
    st.markdown(
        '<div class="upload-zone">'
        '<h3 style="color:#94A3B8;">📁 No dataset yet</h3>'
        '<p style="color:#94A3B8;">Upload a CSV file above to get started</p>'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ── Data preview ──
file_name = st.session_state.file_name or "Dataset"
st.success(f"✅  **{file_name}** loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

with st.expander("📊 Data Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    c4.metric("Duplicates", f"{df.duplicated().sum():,}")


# =========================================================================
# STEP 2 — PROMPT + SETTINGS
# =========================================================================

st.markdown("---")
st.markdown("### 2️⃣  What do you want to know?")

user_prompt = st.text_area(
    "Type your prompt here (optional — leave blank for automatic analysis)",
    placeholder=(
        "Examples:\n"
        "• Show me sales trends by month\n"
        "• What factors most affect customer churn?\n"
        "• Compare revenue across regions\n"
        "• Predict whether a customer will default"
    ),
    height=100,
    label_visibility="collapsed"
)

# ML-specific settings (collapsible)
with st.expander("⚙️ ML Settings (for ML Prediction)", expanded=False):
    target_col = st.text_input(
        "Target Column (leave blank for auto-detection)",
        value="",
        help="The column you want to predict. If blank, the system will auto-detect it."
    )
    if not target_col.strip():
        target_col = None


# =========================================================================
# STEP 3 — TWO ACTION BUTTONS (side by side)
# =========================================================================

st.markdown("---")
st.markdown("### 3️⃣  Choose Your Action")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    btn_insights = st.button(
        "🔍  Find Insights",
        use_container_width=True,
        type="primary",
        help="LLM analyzes your data and discovers business insights, charts & narratives"
    )

with col_btn2:
    btn_ml = st.button(
        "🤖  ML Prediction",
        use_container_width=True,
        type="secondary",
        help="Full ML pipeline: profiling → cleaning → features → model training → SHAP/LIME"
    )


# =========================================================================
# EXECUTE — FIND INSIGHTS
# =========================================================================

if btn_insights:
    st.session_state.active_view = 'insights'
    output_dir = './output'

    if user_prompt and user_prompt.strip():
        # ── Prompt mode: answer a specific question ──
        with st.spinner(f"🤖 Generating insights for: *{user_prompt.strip()}*"):
            try:
                AnalyzerCls = _get_analyzer()
                analyzer = AnalyzerCls()
                result = analyzer.analyze_with_prompt(
                    df, user_prompt.strip(),
                    output_dir=os.path.join(output_dir, 'data_analysis')
                )
                st.session_state.prompt_result = result
                st.session_state.analysis_result = None
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        # ── Auto mode: discover insights automatically ──
        with st.spinner("🤖 AI is analyzing your entire dataset... Discovering insights..."):
            try:
                fns = _get_pipeline_functions()
                result = fns['run_data_analysis'](
                    df=df,
                    dataset_name=file_name,
                    output_dir=output_dir
                )
                st.session_state.analysis_result = result
                st.session_state.prompt_result = None
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# =========================================================================
# EXECUTE — ML PREDICTION
# =========================================================================

if btn_ml:
    st.session_state.active_view = 'ml'
    output_dir = './output'

    with st.spinner("🤖 Running full ML pipeline... (Profiling → Cleaning → Features → Models → Explanations)"):
        try:
            fns = _get_pipeline_functions()
            result = fns['run_ml_pipeline'](
                df=df,
                target_column=target_col,
                dataset_name=file_name,
                output_dir=output_dir
            )
            st.session_state.pipeline_result = result
        except Exception as e:
            st.error(f"ML Pipeline failed: {e}")


# =========================================================================
# DISPLAY RESULTS
# =========================================================================

st.markdown("---")

active_view = st.session_state.active_view

# ─────────────────────────────────────────────────────────────────────────
# RESULTS: FIND INSIGHTS
# ─────────────────────────────────────────────────────────────────────────
if active_view == 'insights':

    # ── Prompt-specific result ──
    prompt_result = st.session_state.prompt_result
    if prompt_result:
        if 'error' in prompt_result:
            st.markdown(
                f'<div class="warn-box">⚠️ {prompt_result["error"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"## 📊 {prompt_result.get('title', 'Insight')}")
            if prompt_result.get('chart') and hasattr(prompt_result['chart'], 'to_json'):
                st.plotly_chart(prompt_result['chart'], use_container_width=True)
            if prompt_result.get('narrative'):
                st.markdown(
                    f'<div class="info-box">💡 {prompt_result["narrative"]}</div>',
                    unsafe_allow_html=True
                )
            if prompt_result.get('description'):
                st.caption(prompt_result['description'])

    # ── Full auto-analysis result ──
    analysis_state = st.session_state.analysis_result
    analysis = (analysis_state or {}).get('data_analysis') if isinstance(analysis_state, dict) else None

    if analysis:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STORYTELLING LAYOUT — applies visual hierarchy, Gestalt
        # principles, horizontal logic, 3-second comprehension test
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # ── HEADLINE: One-sentence summary (3-second test) ──
        domain = analysis.get('domain', 'General')
        n_insights = len(analysis.get('insights', []))
        n_charts = len(analysis.get('charts', {}))
        summary_text = analysis.get('summary', '')

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1E293B,#334155);'
            f'border-radius:12px;padding:28px 32px;margin:0 0 24px 0;color:white;">'
            f'<div style="font-size:1.5em;font-weight:700;margin-bottom:8px;">'
            f'Analysis Complete — {n_insights} Insights Discovered</div>'
            f'<div style="font-size:1.05em;opacity:0.85;line-height:1.5;">{summary_text}</div>'
            f'<div style="display:flex;gap:32px;margin-top:16px;">'
            f'<span style="font-size:0.9em;opacity:0.7;">📁 Domain: <b>{domain}</b></span>'
            f'<span style="font-size:0.9em;opacity:0.7;">📊 Charts: <b>{n_charts}</b></span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # ── KEY TAKEAWAYS: the "executive summary" a manager reads first ──
        takeaways = analysis.get('key_takeaways', [])
        if takeaways:
            st.markdown(
                '<div style="font-size:1.3em;font-weight:700;color:#1E293B;'
                'margin:24px 0 12px 0;">Key Takeaways — What You Need to Know</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
                'border-radius:12px;padding:20px 24px;margin:0 0 28px 0;">',
                unsafe_allow_html=True
            )
            for t in takeaways:
                st.markdown(
                    f'<div style="padding:6px 0;line-height:1.6;color:#334155;'
                    f'font-size:0.98em;">{t}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── INSIGHT CARDS: each chart + its narrative = one story unit ──
        st.markdown(
            '<div style="font-size:1.3em;font-weight:700;color:#1E293B;'
            'margin:8px 0 16px 0;">Insights Dashboard</div>',
            unsafe_allow_html=True
        )

        charts = analysis.get('charts', {})
        narratives = analysis.get('narratives', {})
        insights_list = analysis.get('insights', [])

        insight_chart_keys = [k for k in charts if k.startswith('insight_')]
        for idx, chart_name in enumerate(insight_chart_keys):
            fig = charts[chart_name]
            if not hasattr(fig, 'to_json'):
                continue

            insight_meta = insights_list[idx] if idx < len(insights_list) else {}
            narr = narratives.get(chart_name, '')
            itype = insight_meta.get('insight_type', 'general')

            # Badge color by insight type (pre-attentive: color for category)
            badge_colors = {
                'ranking': '#2563EB', 'comparison': '#7C3AED',
                'composition': '#059669', 'distribution': '#D97706',
                'correlation': '#DC2626', 'trend': '#0891B2',
            }
            badge_bg = badge_colors.get(itype, '#6B7280')

            # Card container — Gestalt proximity: chart + narrative grouped together
            st.markdown(
                f'<div style="background:white;border-radius:10px;'
                f'border:1px solid #F1F5F9;margin:0 0 20px 0;overflow:hidden;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
                f'<div style="padding:14px 20px 4px 20px;display:flex;align-items:center;gap:10px;">'
                f'<span style="background:{badge_bg};color:white;padding:3px 10px;'
                f'border-radius:12px;font-size:0.75em;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.5px;">{itype}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.plotly_chart(fig, use_container_width=True)

            if narr:
                st.markdown(
                    f'<div style="background:#F8FAFC;border-top:1px solid #F1F5F9;'
                    f'padding:14px 20px;font-size:0.93em;color:#475569;line-height:1.65;">'
                    f'<span style="color:{badge_bg};font-weight:600;">Insight:</span> {narr}</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # ── OVERVIEW CHARTS: supporting context in 2-col grid ──
        overview_keys = [k for k in charts if k.startswith('overview_')]
        if overview_keys:
            st.markdown(
                '<div style="font-size:1.15em;font-weight:600;color:#64748B;'
                'margin:28px 0 12px 0;">Supporting Context — Data Overview</div>',
                unsafe_allow_html=True
            )
            for i in range(0, len(overview_keys), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(overview_keys):
                        key = overview_keys[i + j]
                        fig = charts[key]
                        if hasattr(fig, 'to_json'):
                            with col:
                                st.plotly_chart(fig, use_container_width=True)

        # Dashboard link
        dashboard = analysis.get('dashboard', {})
        if dashboard.get('path'):
            st.info(f"📄 Full HTML dashboard saved to: `{dashboard['path']}`")

    if not prompt_result and not analysis:
        st.info("Click **Find Insights** to analyze your data.")


# ─────────────────────────────────────────────────────────────────────────
# RESULTS: ML PREDICTION
# ─────────────────────────────────────────────────────────────────────────
elif active_view == 'ml':

    result = st.session_state.pipeline_result

    if result:
        st.markdown(
            '<div class="success-box">✅ ML Pipeline completed successfully!</div>',
            unsafe_allow_html=True
        )

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", result.get('best_model_name', 'N/A'))
        col2.metric("📈 Ensemble Score", f"{result.get('ensemble_score', 0):.4f}")
        col3.metric("🎯 Task Type", result.get('task_type', 'N/A').title())
        col4.metric("⭐ Data Quality",
                     f"{result.get('profile_report', {}).get('quality_score', 0)}/100")

        # Tabs for detailed results
        tab_models, tab_profile, tab_cleaning, tab_viz, tab_explain = st.tabs([
            "📊 Models", "📋 Data Profile", "🧹 Cleaning", "📈 Visualizations", "🔍 Explanations"
        ])

        # ── Models tab ──
        with tab_models:
            cv_scores = result.get('cv_scores', {})
            if cv_scores:
                st.markdown("### Model Performance (Cross-Validated)")

                scores_df = pd.DataFrame([
                    {'Model': name, 'CV Score': round(score, 4)}
                    for name, score in sorted(cv_scores.items(),
                                              key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

                st.metric("🥇 Ensemble Score", f"{result.get('ensemble_score', 0):.4f}")

                recs = result.get('model_recommendations', [])
                if recs:
                    st.markdown("#### RL Agent Recommendations")
                    for name, conf in recs:
                        st.write(f"→ **{name}** (confidence: {conf:.1%})")
            else:
                st.info("No model scores available.")

        # ── Profile tab ──
        with tab_profile:
            profile = result.get('profile_report', {})
            if profile:
                st.markdown("### Data Profile Summary")
                st.json({
                    'rows': profile.get('n_rows'),
                    'columns': profile.get('n_cols'),
                    'quality_score': profile.get('quality_score'),
                    'target_column': result.get('target_column'),
                    'task_type': result.get('task_type'),
                    'column_types': profile.get('column_types'),
                    'warnings': profile.get('warnings', [])
                })
            else:
                st.info("No profile data available.")

        # ── Cleaning tab ──
        with tab_cleaning:
            cleaning = result.get('cleaning_report', {})
            if cleaning:
                st.markdown("### Cleaning Report")
                dup = cleaning.get('duplicate_removal', {})
                st.metric("Duplicates Removed", dup.get('rows_removed', 0))
                st.json(cleaning)
            else:
                st.info("No cleaning report available.")

        # ── Visualizations tab ──
        with tab_viz:
            display_charts(result.get('visualizations', {}), "ML_Viz")

        # ── Explanations tab ──
        with tab_explain:
            explanations = result.get('explanations', {})
            if explanations:
                narrative = explanations.get('global_narrative', '')
                if narrative:
                    st.markdown("### 🤖 AI-Generated Explanation")
                    st.markdown(
                        f'<div class="info-box">{narrative}</div>',
                        unsafe_allow_html=True
                    )

                importance = explanations.get('shap_importance')
                if importance is not None:
                    st.markdown("### SHAP Feature Importance")
                    st.dataframe(importance.head(15), use_container_width=True,
                                 hide_index=True)

                display_charts(explanations.get('charts', {}), "Explain")

                local_narr = explanations.get('local_narratives', [])
                if local_narr:
                    st.markdown("### Individual Prediction Explanations")
                    for narr in local_narr:
                        with st.expander(
                            f"Sample #{narr['index']} — "
                            f"Predicted: {narr['prediction']}, "
                            f"Actual: {narr['actual']}"
                        ):
                            st.write(narr['narrative'])
            else:
                st.info("No explanations available.")

        # Errors
        if result.get('errors'):
            with st.expander("⚠️ Errors & Warnings"):
                for err in result['errors']:
                    st.warning(err)
    else:
        st.info("Click **ML Prediction** to run the full machine learning pipeline.")

else:
    # No action taken yet
    st.markdown(
        '<div class="info-box">'
        '👆 Upload a dataset above, optionally type a prompt, then click '
        '<b>Find Insights</b> or <b>ML Prediction</b> to get started.'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================================
# FOOTER
# =========================================================================

st.markdown("---")
st.markdown(
    '<p style="color: #9CA3AF; font-size: 0.85em; text-align: center;">'
    'DataPilot AI Pro — Automated Data Science Platform &nbsp;|&nbsp; '
    'Powered by LangGraph, Ollama, SHAP, Reinforcement Learning'
    '</p>',
    unsafe_allow_html=True
)

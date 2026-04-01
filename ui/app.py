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
        # STORYTELLING LAYOUT — visual hierarchy, Gestalt, 3-second test
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        domain = analysis.get('domain', 'General')
        n_insights = len(analysis.get('insights', []))
        n_charts = len(analysis.get('charts', {}))
        summary_text = analysis.get('summary', '')

        # ── HEADLINE ──
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1E293B,#334155);'
            f'border-radius:12px;padding:28px 32px;margin:0 0 24px 0;color:white;">'
            f'<div style="font-size:1.5em;font-weight:700;margin-bottom:8px;">'
            f'Analysis Complete — {n_insights} Insights Discovered</div>'
            f'<div style="font-size:1.05em;opacity:0.85;line-height:1.5;">{summary_text}</div>'
            f'<div style="display:flex;gap:32px;margin-top:16px;">'
            f'<span style="font-size:0.9em;opacity:0.7;">Domain: <b>{domain}</b></span>'
            f'<span style="font-size:0.9em;opacity:0.7;">Charts: <b>{n_charts}</b></span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # ── KPI CARDS ──
        kpis = analysis.get('kpis', [])
        if kpis:
            kpi_cols = st.columns(min(len(kpis), 6))
            for i, kpi in enumerate(kpis):
                if i < len(kpi_cols):
                    with kpi_cols[i]:
                        fmt = kpi.get('format', ',.0f')
                        try:
                            formatted = f'{kpi["value"]:{fmt}}'
                        except (ValueError, KeyError):
                            formatted = str(kpi.get('value', ''))
                        st.metric(
                            label=kpi.get('name', ''),
                            value=formatted,
                            help=kpi.get('description', '')
                        )

        # ── DATA QUALITY WARNINGS ──
        quality = analysis.get('data_quality', {})
        if quality.get('warnings'):
            for w in quality['warnings']:
                st.markdown(
                    f'<div style="background:#451A03;border:1px solid #92400E;'
                    f'border-radius:8px;padding:12px 16px;margin:8px 0;'
                    f'color:#FDE68A;font-size:0.9em;line-height:1.5;">'
                    f'<b>Data Quality:</b> {w}</div>',
                    unsafe_allow_html=True
                )

        # ── KEY TAKEAWAYS ──
        takeaways = analysis.get('key_takeaways', [])
        if takeaways:
            st.markdown(
                '<div style="font-size:1.3em;font-weight:700;color:#1E293B;'
                'margin:24px 0 12px 0;">Key Takeaways</div>',
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

        # ── INSIGHT CARDS ──
        st.markdown(
            '<div style="font-size:1.3em;font-weight:700;color:#1E293B;'
            'margin:8px 0 16px 0;">Insights Dashboard</div>',
            unsafe_allow_html=True
        )

        charts = analysis.get('charts', {})
        narratives = analysis.get('narratives', {})
        insights_list = analysis.get('insights', [])

        badge_colors = {
            'ranking': '#2563EB', 'comparison': '#7C3AED',
            'composition': '#059669', 'distribution': '#D97706',
            'correlation': '#DC2626', 'trend': '#0891B2',
        }

        insight_chart_keys = [k for k in charts if k.startswith('insight_')]
        for idx, chart_name in enumerate(insight_chart_keys):
            fig = charts[chart_name]
            if not hasattr(fig, 'to_json'):
                continue

            insight_meta = insights_list[idx] if idx < len(insights_list) else {}
            narr = narratives.get(chart_name, '')
            itype = insight_meta.get('insight_type', 'general')
            badge_bg = badge_colors.get(itype, '#6B7280')
            title = insight_meta.get('title', chart_name.replace('_', ' ').title())

            st.markdown(
                f'<div style="background:white;border-radius:10px;'
                f'border:1px solid #F1F5F9;margin:0 0 20px 0;overflow:hidden;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
                f'<div style="padding:14px 20px 4px 20px;display:flex;align-items:center;gap:10px;">'
                f'<span style="background:{badge_bg};color:white;padding:3px 10px;'
                f'border-radius:12px;font-size:0.75em;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.5px;">{itype}</span>'
                f'<span style="font-size:1.05em;font-weight:600;color:#1F2937;">{title}</span>'
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

        # ── OVERVIEW CHARTS (2-col grid) ──
        overview_keys = [k for k in charts if k.startswith('overview_')]
        if overview_keys:
            st.markdown(
                '<div style="font-size:1.15em;font-weight:600;color:#64748B;'
                'margin:28px 0 12px 0;">Data Overview</div>',
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
            st.info(f"Full HTML dashboard saved to: `{dashboard['path']}`")

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
        tab_models, tab_profile, tab_cleaning, tab_features, tab_viz, tab_explain, tab_errors, tab_segments, tab_predict = st.tabs([
            "📊 Models", "📋 Data Profile", "🧹 Cleaning", "🔧 Features",
            "📈 Visualizations", "🔍 Explanations", "⚠️ Error Analysis",
            "📐 Segments", "🎯 Predict"
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
                
                # Overfitting Detection
                overfit = result.get('overfitting_analysis', {})
                if overfit:
                    if overfit.get('is_suspicious'):
                        st.markdown("### ⚠️ Overfitting Warning")
                        st.markdown(
                            f'<div class="warn-box">{overfit["reason"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="success-box">✅ {overfit.get("reason", "Scores look healthy.")}</div>',
                            unsafe_allow_html=True
                        )
                    
                    if overfit.get('train_score') is not None:
                        oc1, oc2, oc3 = st.columns(3)
                        oc1.metric("Train Score", f"{overfit['train_score']:.4f}")
                        oc2.metric("CV Score", f"{overfit['cv_score']:.4f}")
                        oc3.metric("Gap (overfit indicator)", f"{overfit.get('gap', 0):.4f}")
            else:
                st.info("No model scores available.")

        # ── Profile tab ──
        with tab_profile:
            profile = result.get('profile_report', {})
            if profile:
                st.markdown("### Data Profile Summary")
                
                # Basic info
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Rows", profile.get('n_rows', 'N/A'))
                pc2.metric("Columns", profile.get('n_cols', 'N/A'))
                pc3.metric("Quality Score", f"{profile.get('quality_score', 0)}/100")
                pc4.metric("Task Type", result.get('task_type', 'N/A').title())
                
                st.markdown(f"**Target Column:** `{result.get('target_column', 'N/A')}`")
                
                # Column Types
                col_types = profile.get('column_types', {})
                if col_types:
                    st.markdown("#### Column Types")
                    ct_df = pd.DataFrame([
                        {'Column': col, 'Type': ctype}
                        for col, ctype in col_types.items()
                    ])
                    st.dataframe(ct_df, use_container_width=True, hide_index=True)
                
                # describe() Anomaly Detection
                anomalies = profile.get('describe_anomalies', [])
                if anomalies:
                    st.markdown("### 🔍 Anomalies Detected (from describe() analysis)")
                    st.markdown(
                        '<div class="warn-box">'
                        'These are unusual patterns found by analyzing descriptive statistics — '
                        'just like a data scientist does with <code>df.describe()</code>.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    anom_df = pd.DataFrame(anomalies)
                    st.dataframe(anom_df, use_container_width=True, hide_index=True)
                
                # Uniformity Issues
                uniformity = profile.get('uniformity_issues', {})
                if uniformity:
                    st.markdown("### 🔄 Non-Uniform Category Values Detected")
                    for col, issues in uniformity.items():
                        with st.expander(f"Column: `{col}` — {len(issues)} inconsistencies"):
                            for canonical, variants in issues.items():
                                st.write(f"  `{canonical}` has variants: {variants}")
                    st.markdown(
                        '<div class="info-box">✅ These have been automatically standardized by the Cleaner.</div>',
                        unsafe_allow_html=True
                    )
                
                # Warnings
                warnings = profile.get('warnings', [])
                if warnings:
                    st.markdown("#### Warnings")
                    for w in warnings:
                        st.write(w)
            else:
                st.info("No profile data available.")

        # ── Cleaning tab ──
        with tab_cleaning:
            cleaning = result.get('cleaning_report', {})
            if cleaning:
                st.markdown("### Cleaning Report")
                
                cc1, cc2, cc3 = st.columns(3)
                dup = cleaning.get('duplicate_removal', {})
                cc1.metric("Duplicates Removed", dup.get('rows_removed', 0))
                cc2.metric("Rows Remaining", dup.get('rows_remaining', 'N/A'))
                cc3.metric("Columns Imputed", len(cleaning.get('missing_value_handling', {})))
                
                # Missing value handling details
                mv = cleaning.get('missing_value_handling', {})
                if mv:
                    st.markdown("#### Missing Value Handling")
                    mv_df = pd.DataFrame([
                        {'Column': col, 'Strategy': info.get('strategy', 'N/A'),
                         'Value': str(info.get('value', 'N/A'))}
                        for col, info in mv.items()
                    ])
                    st.dataframe(mv_df, use_container_width=True, hide_index=True)
                
                # Outlier handling
                outliers = cleaning.get('outlier_handling', {})
                if outliers:
                    st.markdown("#### Outlier Treatment (IQR Method)")
                    out_df = pd.DataFrame([
                        {'Column': col, 'N Outliers': info.get('n_outliers', 0),
                         'Outlier %': info.get('outlier_pct', 0),
                         'Action': info.get('action', 'N/A'),
                         'Lower Bound': info.get('lower_bound', 'N/A'),
                         'Upper Bound': info.get('upper_bound', 'N/A')}
                        for col, info in outliers.items()
                    ])
                    st.dataframe(out_df, use_container_width=True, hide_index=True)
                
                # Category standardization
                cat_std = cleaning.get('category_standardization', {})
                if cat_std:
                    st.markdown("#### Category Standardization")
                    for col, mapping in cat_std.items():
                        with st.expander(f"Column: `{col}`"):
                            for old, new in mapping.items():
                                st.write(f"  `{old}` → `{new}`")
            else:
                st.info("No cleaning report available.")

        # ── Features tab ──
        with tab_features:
            feat_report = result.get('feature_report', {})
            if feat_report:
                st.markdown("### Feature Engineering Report")
                
                # VIF Analysis
                vif_info = feat_report.get('vif_analysis', {})
                if vif_info:
                    st.markdown("#### VIF Multicollinearity Analysis")
                    st.markdown(
                        '<div class="info-box">'
                        '<b>Variance Inflation Factor (VIF)</b> measures multicollinearity. '
                        'VIF &gt; 10 means the feature is highly correlated with others and should be removed. '
                        f'Threshold used: {vif_info.get("threshold", 10)}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    removed_vif = vif_info.get('removed_features', [])
                    if removed_vif:
                        st.markdown(f"**Removed {len(removed_vif)} features due to high VIF:**")
                        vif_rm_df = pd.DataFrame(removed_vif)
                        st.dataframe(vif_rm_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("All features have acceptable VIF — no multicollinearity issues.")
                    
                    final_vif = vif_info.get('final_vif_scores', {})
                    if final_vif:
                        st.markdown("#### Final VIF Scores (remaining features)")
                        vif_df = pd.DataFrame([
                            {'Feature': f, 'VIF': v}
                            for f, v in sorted(final_vif.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(vif_df, use_container_width=True, hide_index=True)
                
                # Encoding
                encoding = feat_report.get('encoding', {})
                if encoding:
                    st.markdown("#### Encoding Applied")
                    enc_df = pd.DataFrame([
                        {'Column': col, 'Method': info.get('method', 'N/A'),
                         'Unique Values': info.get('n_unique', 'N/A')}
                        for col, info in encoding.items()
                    ])
                    st.dataframe(enc_df, use_container_width=True, hide_index=True)
                
                # Scaling
                scaling = feat_report.get('scaling', {})
                if scaling:
                    st.markdown("#### Scaling")
                    st.write(f"**Method:** {scaling.get('method', 'N/A')}")
                    st.write(f"**Reason:** {scaling.get('reason', 'N/A')}")
                    st.write(f"**Columns Scaled:** {scaling.get('n_columns', 0)}")
                
                # Dropped columns
                dropped = feat_report.get('dropped_columns', [])
                if dropped:
                    st.markdown("#### Dropped Columns")
                    st.write(f"ID columns removed: {dropped}")
            else:
                st.info("No feature engineering report available.")

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

        # ── Error Analysis tab ──
        with tab_errors:
            error_analysis = result.get('error_analysis', {})
            if error_analysis and error_analysis.get('summary'):
                st.markdown("### ⚠️ Error Analysis")
                st.markdown(
                    '<div class="info-box">'
                    'Error analysis shows <b>where and why</b> the model makes mistakes — '
                    'just like a real data scientist would investigate after training.'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.write(f"**Summary:** {error_analysis['summary']}")
                
                if result.get('task_type') == 'classification':
                    # Class-level error rates
                    class_errors = error_analysis.get('class_errors', [])
                    if class_errors:
                        st.markdown("#### Per-Class Error Rates")
                        ce_df = pd.DataFrame(class_errors)
                        st.dataframe(ce_df, use_container_width=True, hide_index=True)
                    
                    ec1, ec2 = st.columns(2)
                    ec1.metric("Total Misclassified", error_analysis.get('total_errors', 'N/A'))
                    ec2.metric("Error Rate", f"{error_analysis.get('error_rate', 0):.1f}%")
                else:
                    # Regression error metrics
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("MAE", f"{error_analysis.get('mae', 0):.4f}")
                    ec2.metric("RMSE", f"{error_analysis.get('rmse', 0):.4f}")
                    ec3.metric("Median Error", f"{error_analysis.get('median_error', 0):.4f}")
                    
                    # Error patterns
                    patterns = error_analysis.get('error_patterns', [])
                    if patterns:
                        st.markdown("#### Error by Target Value Range")
                        pat_df = pd.DataFrame(patterns)
                        st.dataframe(pat_df, use_container_width=True, hide_index=True)
                    
                    # Worst predictions
                    worst = error_analysis.get('worst_samples', [])
                    if worst:
                        st.markdown("#### Top 10 Worst Predictions")
                        worst_df = pd.DataFrame(worst)
                        st.dataframe(worst_df, use_container_width=True, hide_index=True)
            else:
                st.info("No error analysis available.")

        # ── Segment Analysis tab ──
        with tab_segments:
            seg_analysis = result.get('segment_analysis', {})
            if seg_analysis and seg_analysis.get('summary'):
                st.markdown("### 📐 Model Segmentation Analysis")
                st.markdown(
                    '<div class="info-box">'
                    'Segment analysis checks if the model performs <b>equally well across all data groups</b>. '
                    'For example, does the model work for all age groups, or does it fail for age &lt; 25?'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.write(f"**Summary:** {seg_analysis['summary']}")
                
                # Problem segments highlighted
                problems = seg_analysis.get('problem_segments', [])
                if problems:
                    st.markdown("#### ⚠️ Underperforming Segments")
                    st.markdown(
                        '<div class="warn-box">'
                        'These segments have significantly worse performance than overall. '
                        'Consider training separate models or adding more data for these groups.'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    prob_df = pd.DataFrame(problems)
                    st.dataframe(prob_df[['column', 'segment', 'n_samples', 'score', 'overall_score', 'gap']],
                                use_container_width=True, hide_index=True)
                else:
                    st.markdown(
                        '<div class="success-box">✅ Model performs consistently across all segments.</div>',
                        unsafe_allow_html=True
                    )
                
                # All segments
                all_segments = seg_analysis.get('segments', [])
                if all_segments:
                    with st.expander(f"View All {len(all_segments)} Segments"):
                        seg_df = pd.DataFrame(all_segments)
                        st.dataframe(seg_df, use_container_width=True, hide_index=True)
            else:
                st.info("No segment analysis available.")

        # ── Predict Your Own Values tab ──
        with tab_predict:
            st.markdown("### 🎯 Predict With Your Own Values")
            st.markdown(
                '<div class="info-box">'
                'Enter your own feature values below and the trained model will make a prediction. '
                'This lets you test "what-if" scenarios.'
                '</div>',
                unsafe_allow_html=True
            )
            
            feature_names = result.get('feature_names', [])
            feature_report_data = result.get('feature_report', {})
            
            if feature_names and result.get('ensemble_model'):
                st.markdown("#### Enter Feature Values")
                
                # Create input form
                user_inputs = {}
                n_cols_per_row = 3
                
                for i in range(0, len(feature_names), n_cols_per_row):
                    cols = st.columns(n_cols_per_row)
                    for j, col in enumerate(cols):
                        feat_idx = i + j
                        if feat_idx < len(feature_names):
                            feat_name = feature_names[feat_idx]
                            with col:
                                user_inputs[feat_name] = st.number_input(
                                    feat_name,
                                    value=0.0,
                                    format="%.4f",
                                    key=f"predict_input_{feat_name}"
                                )
                
                if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
                    try:
                        import numpy as np_pred
                        input_df = pd.DataFrame([user_inputs])

                        # Ensure columns are in the right order
                        input_df = input_df[feature_names]

                        # Apply the same numeric scaler used during training
                        scalers = result.get('scalers', {})
                        numeric_scaler = scalers.get('numeric')
                        cols_scaled = (result.get('feature_report', {})
                                       .get('scaling', {})
                                       .get('columns_scaled', []))
                        if numeric_scaler is not None and cols_scaled:
                            available_scaled = [c for c in cols_scaled if c in input_df.columns]
                            if available_scaled:
                                input_df[available_scaled] = numeric_scaler.transform(
                                    input_df[available_scaled]
                                )

                        model = result['ensemble_model']
                        prediction = model.predict(input_df)[0]
                        
                        # Decode prediction for classification
                        encoders = result.get('encoders', {})
                        task_type = result.get('task_type', '')
                        display_prediction = prediction
                        
                        if task_type == 'classification' and 'target' in encoders:
                            try:
                                display_prediction = encoders['target'].inverse_transform([int(prediction)])[0]
                            except Exception:
                                display_prediction = prediction
                        
                        st.markdown("---")
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#059669,#10B981);'
                            f'border-radius:12px;padding:24px;text-align:center;color:white;">'
                            f'<div style="font-size:1.2em;opacity:0.9;">Predicted Value</div>'
                            f'<div style="font-size:2.5em;font-weight:800;">{display_prediction}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Show prediction confidence for classification
                        if task_type == 'classification' and hasattr(model, 'predict_proba'):
                            try:
                                proba = model.predict_proba(input_df)[0]
                                if 'target' in encoders:
                                    classes = encoders['target'].classes_
                                else:
                                    classes = [f"Class {i}" for i in range(len(proba))]
                                
                                st.markdown("#### Prediction Confidence")
                                prob_df = pd.DataFrame({
                                    'Class': classes,
                                    'Probability': [f"{p:.2%}" for p in proba]
                                })
                                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                            except Exception:
                                pass
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.info("Train a model first (click **ML Prediction** above).")

        # Errors
        if result.get('errors'):
            with st.expander("⚠️ Pipeline Errors & Warnings"):
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

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Plotly JSON Figure Editor")

st.title("Plotly JSON Figure Editor")

st.markdown(
    """
Upload Plotly figure JSON files exported from your other app and interactively
tweak layout and trace properties (title, colors, axes, legend, etc.).
All controls are on the main page for easier editing.
"""
)

# ──────────────────────────
# LOAD FIGURES (MAIN PAGE, NOT SIDEBAR)
# ──────────────────────────

with st.expander("📂 Load / select figure", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload one or more Plotly JSON files",
        type=["json"],
        accept_multiple_files=True,
    )

    raw_json_text = st.text_area(
        "Or paste Plotly JSON here",
        value="",
        height=120,
        placeholder="Paste JSON exported from fig.to_json()",
    )

    figs = {}

    def load_fig_from_json_str(name: str, text: str):
        try:
            fig = pio.from_json(text)
            figs[name] = fig
        except Exception as e:
            st.error(f"Failed to load {name}: {e}")

    # From uploaded files
    if uploaded_files:
        for f in uploaded_files:
            content = f.read().decode("utf-8")
            load_fig_from_json_str(f.name, content)

    # From textarea
    if raw_json_text.strip():
        load_fig_from_json_str("pasted_json", raw_json_text.strip())

    if not figs:
        st.info("Upload a JSON file or paste JSON above to get started.")
        st.stop()

    fig_name = st.selectbox(
        "Select figure to edit", options=list(figs.keys()), index=0
    )

# Copy to avoid mutating the original object that pio.from_json returns
fig: go.Figure = go.Figure(figs[fig_name])

# ──────────────────────────
# PRECOMPUTE SAFE DEFAULTS FROM FIG
# (only for things that are very safe to read)
# ──────────────────────────

current_title = ""
if fig.layout and fig.layout.title and fig.layout.title.text:
    current_title = str(fig.layout.title.text)

current_x_title = ""
if fig.layout.xaxis and fig.layout.xaxis.title and fig.layout.xaxis.title.text:
    current_x_title = str(fig.layout.xaxis.title.text)

current_y_title = ""
if fig.layout.yaxis and fig.layout.yaxis.title and fig.layout.yaxis.title.text:
    current_y_title = str(fig.layout.yaxis.title.text)

# ──────────────────────────
# MAIN CONTROL TABS (CENTER PAGE)
# ──────────────────────────

tab_layout, tab_axes, tab_trace = st.tabs(["🎨 Layout", "📐 Axes", "🧬 Trace styling"])

# ---------- LAYOUT TAB ----------
with tab_layout:
    st.subheader("Layout & Styling")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Title**")
        title_text = st.text_input("Title text", value=current_title)
        title_font_size = st.slider("Title font size", 8, 48, 18)
        title_font_family = st.text_input("Title font family", value="Arial, sans-serif")

    with col2:
        st.markdown("**Title position**")
        title_x = st.slider("Title X position", 0.0, 1.0, 0.5, step=0.01)
        title_y = st.slider("Title Y position", 0.0, 1.0, 0.95, step=0.01)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Figure size**")
        fig_width = st.number_input("Width (px)", min_value=300, max_value=3000, value=900)
        fig_height = st.number_input("Height (px)", min_value=200, max_value=3000, value=600)

        st.markdown("**Template**")
        default_templates = [
            "plotly",
            "plotly_white",
            "plotly_dark",
            "ggplot2",
            "seaborn",
            "simple_white",
            "presentation",
            "xgridoff",
            "ygridoff",
            "gridon",
        ]
        template = st.selectbox(
            "Plotly template",
            options=default_templates,
            index=0,  # "plotly"
        )

    with col4:
        st.markdown("**Background colors**")
        paper_bgcolor = st.color_picker("Paper background", value="#ffffff")
        plot_bgcolor = st.color_picker("Plot background", value="#ffffff")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Legend**")
        show_legend = st.checkbox("Show legend", value=True)
        legend_x = st.slider("Legend X", -0.2, 1.2, 1.02, step=0.01)
        legend_y = st.slider("Legend Y", -0.2, 1.2, 1.0, step=0.01)

    with col6:
        st.markdown("**Interaction & Margins**")
        hovermode = st.selectbox(
            "Hover mode",
            options=["closest", "x", "y", "x unified", "y unified", "off"],
            index=0,
        )
        dragmode = st.selectbox(
            "Drag mode",
            options=["zoom", "pan", "select", "lasso"],
            index=0,
        )

        margin_l = st.number_input("Left margin", value=60)
        margin_r = st.number_input("Right margin", value=40)
        margin_t = st.number_input("Top margin", value=80)
        margin_b = st.number_input("Bottom margin", value=60)

# ---------- AXES TAB ----------
with tab_axes:
    st.subheader("Axes configuration")

    axis_type_options = ["linear", "log", "date", "category"]

    colx, coly = st.columns(2)

    with colx:
        st.markdown("**X Axis**")
        xaxis_title = st.text_input("X-axis title", value=current_x_title)
        xaxis_type = st.selectbox("X-axis scale", options=axis_type_options, index=0)
        xaxis_showgrid = st.checkbox("Show x grid", value=True)
        xaxis_zeroline = st.checkbox("Show x zero line", value=False)
        xaxis_tickangle = st.slider("X tick angle", -90, 90, 0)
        xaxis_autorange = st.checkbox("X auto range", value=True)
        xaxis_range_min = st.text_input("X min (if manual range)", value="")
        xaxis_range_max = st.text_input("X max (if manual range)", value="")

    with coly:
        st.markdown("**Y Axis**")
        yaxis_title = st.text_input("Y-axis title", value=current_y_title)
        yaxis_type = st.selectbox("Y-axis scale", options=axis_type_options, index=0)
        yaxis_showgrid = st.checkbox("Show y grid", value=True)
        yaxis_zeroline = st.checkbox("Show y zero line", value=False)
        yaxis_tickangle = st.slider("Y tick angle", -90, 90, 0)
        yaxis_autorange = st.checkbox("Y auto range", value=True)
        yaxis_range_min = st.text_input("Y min (if manual range)", value="")
        yaxis_range_max = st.text_input("Y max (if manual range)", value="")

# ---------- TRACE TAB ----------
selected_trace_index = None
trace_visible = True
trace_color = "#1f77b4"
marker_size = 8.0
marker_symbol = "circle"
line_width = 2.0
line_dash = "solid"
trace_opacity = 1.0

with tab_trace:
    st.subheader("Per-trace styling")

    if len(fig.data) == 0:
        st.info("No traces found in this figure.")
    else:
        trace_names = []
        for i, tr in enumerate(fig.data):
            name = tr.name if tr.name is not None else f"trace {i}"
            trace_names.append(f"{i}: {name}")

        selected_trace_label = st.selectbox(
            "Select trace to edit",
            options=trace_names,
        )
        selected_trace_index = int(selected_trace_label.split(":")[0])
        tr = fig.data[selected_trace_index]

        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Visibility & color**")
            trace_visible = st.checkbox(
                "Visible", value=(tr.visible is None or tr.visible is True)
            )
            trace_color = st.color_picker("Color", value="#1f77b4")
            trace_opacity = st.slider(
                "Opacity",
                0.0,
                1.0,
                float(tr.opacity if tr.opacity is not None else 1.0),
            )

        with colB:
            st.markdown("**Markers & lines**")
            marker_size = st.slider("Marker size", 1.0, 40.0, 8.0)
            marker_symbol = st.selectbox(
                "Marker symbol",
                options=[
                    "circle", "square", "diamond", "cross", "x",
                    "triangle-up", "triangle-down", "triangle-left", "triangle-right",
                    "pentagon", "hexagon", "star", "hexagram"
                ],
                index=0,
            )
            line_width = st.slider("Line width", 0.0, 10.0, 2.0)
            line_dash = st.selectbox(
                "Line dash style",
                options=["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"],
                index=0,
            )

# ──────────────────────────
# APPLY CHANGES TO FIG
# ──────────────────────────

# Layout
fig.update_layout(
    title=dict(
        text=title_text,
        font=dict(size=title_font_size, family=title_font_family),
        x=title_x,
        y=title_y,
        xanchor="center",
        yanchor="top",
    ),
    template=template,
    width=fig_width,
    height=fig_height,
    paper_bgcolor=paper_bgcolor,
    plot_bgcolor=plot_bgcolor,
    showlegend=show_legend,
    legend=dict(x=legend_x, y=legend_y),
    hovermode=None if hovermode == "off" else hovermode,
    dragmode=dragmode,
    margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
)

# Axes
xaxis_kwargs = dict(
    title=xaxis_title,
    type=xaxis_type,
    showgrid=xaxis_showgrid,
    zeroline=xaxis_zeroline,
    tickangle=xaxis_tickangle,
)
if not xaxis_autorange and xaxis_range_min.strip() and xaxis_range_max.strip():
    try:
        x_min = float(xaxis_range_min)
        x_max = float(xaxis_range_max)
        xaxis_kwargs["range"] = [x_min, x_max]
    except ValueError:
        pass
else:
    xaxis_kwargs["autorange"] = True

fig.update_xaxes(**xaxis_kwargs)

yaxis_kwargs = dict(
    title=yaxis_title,
    type=yaxis_type,
    showgrid=yaxis_showgrid,
    zeroline=yaxis_zeroline,
    tickangle=yaxis_tickangle,
)
if not yaxis_autorange and yaxis_range_min.strip() and yaxis_range_max.strip():
    try:
        y_min = float(yaxis_range_min)
        y_max = float(yaxis_range_max)
        yaxis_kwargs["range"] = [y_min, y_max]
    except ValueError:
        pass
else:
    yaxis_kwargs["autorange"] = True

fig.update_yaxes(**yaxis_kwargs)

# Trace
if len(fig.data) > 0 and selected_trace_index is not None:
    tr = fig.data[selected_trace_index]
    tr.visible = trace_visible

    if not hasattr(tr, "marker") or tr.marker is None:
        tr.marker = {}
    if not hasattr(tr, "line") or tr.line is None:
        tr.line = {}

    tr.marker.color = trace_color
    tr.marker.size = marker_size
    tr.marker.symbol = marker_symbol
    tr.line.width = line_width
    tr.line.dash = line_dash
    tr.opacity = trace_opacity

# ──────────────────────────
# DISPLAY FIGURE & EXPORT
# ──────────────────────────

st.markdown("---")
st.subheader("Preview")

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Export edited figure")

json_str = fig.to_json()
preview = json_str[:2000] + ("...\n" if len(json_str) > 2000 else "")
st.code(preview, language="json")

st.download_button(
    label="💾 Download edited figure JSON",
    data=json_str,
    file_name=f"edited_{fig_name}.json",
    mime="application/json",
)
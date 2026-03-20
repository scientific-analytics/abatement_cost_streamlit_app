import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

# ============================================================================
# DEFAULT DATA - Embedded clinker plant data (~139 plants)
# Intensity converted from kgCO2/t to tCO2/t for consistency
# ============================================================================
DEFAULT_PLANTS_DATA = """production_t_per_yr,production_intensity_tco2_per_t
784192.47,0.464
199361.61,0.780
419708.65,0.661
928605.38,0.901
414462.29,0.591
996808.04,0.681
577099.39,0.715
131158.95,0.958
839417.30,0.627
262317.91,0.733
682026.55,0.443
996808.04,0.571
419708.65,0.365
577099.39,0.684
839417.30,0.663
19297.02,0.914
170506.64,0.556
313676.99,0.809
262317.91,0.914
524635.81,0.712
524635.81,0.442
472125.01,1.020
419666.68,1.152
262291.67,0.707
288549.70,0.369
313676.99,0.757
1097869.46,0.897
721457.08,0.551
776015.06,0.543
1758967.47,0.425
814810.88,1.053
516256.33,0.925
313676.99,0.766
117628.87,0.789
483993.79,0.439
1206662.36,0.682
1265813.67,0.519
697931.30,0.652
563024.71,0.732
1252928.23,0.819
316458.72,0.944
322283.11,0.903
409649.01,0.559
682026.55,0.847
524635.81,0.482
682026.55,0.331
786953.71,0.819
472172.23,0.860
692519.27,0.893
689205.78,1.013
498404.02,0.597
786953.71,0.507
262317.91,0.679
314781.49,0.664
131158.95,0.661
16567.45,0.870
55224.82,0.850
1091242.48,0.364
367245.07,0.907
288549.70,0.581
55224.82,0.870
152144.38,1.219
524635.81,0.959
225593.40,0.851
262317.91,0.393
577099.39,0.661
524635.81,0.369
367245.07,0.384
209854.32,0.643
734490.13,0.701
524635.81,0.380
472172.23,0.338
99943.12,0.661
419708.65,0.325
786953.71,0.366
786953.71,0.261
275470.25,0.469
524635.81,1.067
419708.65,0.979
367245.07,1.045
209854.32,0.946
1573907.43,0.253
262317.91,0.814
629562.97,0.410
865649.09,0.687
980240.59,0.621
1573907.43,0.587
382984.14,0.720
1311589.53,0.313
331514.61,0.733
856730.28,0.879
593771.76,0.579
678326.49,0.886
548934.73,0.634
440694.08,0.498
229528.17,1.064
552248.22,0.664
238571.23,0.766
786953.71,0.633
503539.93,0.395
209854.32,0.780
1101735.20,0.339
262317.91,0.733
839417.30,0.391
367245.07,0.438
431305.86,0.921
513646.07,0.342
392096.24,0.327
490120.30,0.332
891880.88,0.663
891880.88,0.529
262317.91,0.445
392427.59,0.674
1731298.17,0.572
2098543.24,0.773
392096.24,0.661
784192.47,0.697
968477.71,0.169
993817.61,1.026
470755.71,0.689
262317.91,0.764
419708.65,0.850
472172.23,0.419
367245.07,0.782
472172.23,0.456
588144.36,0.573
392096.24,0.557
352886.61,0.824
785296.97,0.862
921426.16,1.013
427550.57,0.764
786875.02,0.921
592838.47,0.951
996808.04,0.744
1626371.01,0.328
196048.12,1.053
1049271.62,0.500
1521443.85,0.737
524635.81,0.680
"""

# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================

def get_final_intensity(starting_intensity, tech):
    """Compute final intensity after applying a technology."""
    improved = starting_intensity * (1 - tech["improvement_pct"])
    target_min, target_max = tech["target_range"]
    return max(min(improved, target_max), target_min)


def calc_threshold(current_intensity, final_intensity, investment, project_lifetime, discount_rate, carbon_price_growth):
    """Calculate carbon price threshold for a technology transition."""
    emissions_avoided = current_intensity - final_intensity
    if emissions_avoided <= 0:
        return float("inf")
    t = np.arange(1, project_lifetime + 1)
    return investment / sum(
        emissions_avoided * ((1 + carbon_price_growth) ** year) / ((1 + discount_rate) ** year)
        for year in t
    )


def compute_npv(current_intensity, tech, carbon_price_t0, project_lifetime, discount_rate, carbon_price_growth):
    """Compute NPV of switching to a technology given initial carbon price."""
    final_int = get_final_intensity(current_intensity, tech)
    emissions_avoided = current_intensity - final_int
    if emissions_avoided <= 0:
        return -tech["investment"]

    t = np.arange(1, project_lifetime + 1)
    carbon_prices = carbon_price_t0 * (1 + carbon_price_growth) ** t
    savings = emissions_avoided * carbon_prices
    discounted_savings = savings / (1 + discount_rate) ** t
    return discounted_savings.sum() - tech["investment"]


def find_optimal_technology_incremental(current_intensity, carbon_price_t0, available_techs, project_lifetime, discount_rate, carbon_price_growth):
    """Find the technology that maximizes NPV from current intensity."""
    best_tech = None
    best_npv = 0

    for tech in available_techs:
        npv = compute_npv(current_intensity, tech, carbon_price_t0, project_lifetime, discount_rate, carbon_price_growth)
        if npv > best_npv:
            best_npv = npv
            best_tech = tech

    return best_tech


def generate_carbon_trajectory(start_price, end_price, years, shape):
    """Generate carbon price trajectory based on shape."""
    t = np.linspace(0, 1, years)
    if shape == "Linear":
        trajectory = start_price + (end_price - start_price) * t
    elif shape == "Concave":
        trajectory = start_price + (end_price - start_price) * np.sqrt(t)
    else:  # Convex
        trajectory = start_price + (end_price - start_price) * (t ** 2)
    return trajectory


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Abatement Cost Analysis", layout="wide")

st.title("How can we decarbonize carbon-heavy industries?")

st.markdown("""
This tool analyzes how carbon pricing influences technology adoption for emission-intensive sectors.
It allows you to change parameters related to the way firms take their investment decisions, define up to three different mitigation technologies, and look at the resulting dynamics for a given facility, as well as for a set of facilities that might represent a given sector/region.

The tool is organised in three sections. **Parameters** is where you can define/modify parameters related to the characteristics of three mitigation technologies, and the decision-making process of firms.
Then, the **Single Facility Analysis** replicates a typical decision process at the facility level. Given its initial carbon intensity, at what carbon price is it effective to invest in a cleaner technology?
Finally, the **Sector Analysis** extends the results of the facility analysis to a set of facilities representative of a sector/region (you can upload your own list of facility characteristics).

The model is by definition very simplified, and it's main purpose is to be pedagogical.
""")

st.markdown('<p style="color: gray; font-style: italic;">Last update: March 2026. For any questions or feedback, please contact vincent.bouchet [at] scientificportfolio.com</p>', unsafe_allow_html=True)

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
st.header("Parameters")

st.markdown("""
The model allows you to define up to three decarbonization technologies. Each technology is characterized by an **improvement percentage** (the percentage reduction in emission intensity from the starting point), a **target range** (the achievable intensity range after applying the technology), and an **investment** cost (capital expenditure required per unit of production capacity in year 0).
This allows you to represent different types of technologies: energy efficiency improvements (whose target intensity depends on the starting point) or alternative technologies (whose target intensity is more fixed—in this case, focus on setting the target range).

""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p style="font-weight: bold; color: #17A589;">Technology 1</p>', unsafe_allow_html=True)
    tech1_name = st.text_input("Name", "Alternative fuels", key="tech1_name")
    t1c1, t1c2 = st.columns(2)
    with t1c1:
        tech1_improvement = st.number_input("Improvement %", 0, 100, 10, key="tech1_imp")
        tech1_target_min = st.number_input("Target Min (tCO2/t)", 0.0, 1.5, 0.700, 0.01, key="tech1_min")
    with t1c2:
        tech1_investment = st.number_input("Investment (EUR/t)", 0, 5000, 10, key="tech1_inv")
        tech1_target_max = st.number_input("Target Max (tCO2/t)", 0.0, 1.5, 0.850, 0.01, key="tech1_max")

with col2:
    st.markdown('<p style="font-weight: bold; color: #52BE80;">Technology 2</p>', unsafe_allow_html=True)
    tech2_name = st.text_input("Name", "Energy efficiency", key="tech2_name")
    t2c1, t2c2 = st.columns(2)
    with t2c1:
        tech2_improvement = st.number_input("Improvement %", 0, 100, 40, key="tech2_imp")
        tech2_target_min = st.number_input("Target Min (tCO2/t)", 0.0, 1.5, 0.400, 0.01, key="tech2_min")
    with t2c2:
        tech2_investment = st.number_input("Investment (EUR/t)", 0, 5000, 100, key="tech2_inv")
        tech2_target_max = st.number_input("Target Max (tCO2/t)", 0.0, 1.5, 0.600, 0.01, key="tech2_max")

with col3:
    st.markdown('<p style="font-weight: bold; color: #1E8449;">Technology 3</p>', unsafe_allow_html=True)
    tech3_name = st.text_input("Name", "CCUS", key="tech3_name")
    t3c1, t3c2 = st.columns(2)
    with t3c1:
        tech3_improvement = st.number_input("Improvement %", 0, 100, 80, key="tech3_imp")
        tech3_target_min = st.number_input("Target Min (tCO2/t)", 0.0, 1.5, 0.050, 0.01, key="tech3_min")
    with t3c2:
        tech3_investment = st.number_input("Investment (EUR/t)", 0, 5000, 1000, key="tech3_inv")
        tech3_target_max = st.number_input("Target Max (tCO2/t)", 0.0, 1.5, 0.250, 0.01, key="tech3_max")

# Build technology list
technologies = [
    {"name": tech1_name, "improvement_pct": tech1_improvement/100, "target_range": (tech1_target_min, tech1_target_max), "investment": tech1_investment},
    {"name": tech2_name, "improvement_pct": tech2_improvement/100, "target_range": (tech2_target_min, tech2_target_max), "investment": tech2_investment},
    {"name": tech3_name, "improvement_pct": tech3_improvement/100, "target_range": (tech3_target_min, tech3_target_max), "investment": tech3_investment},
]

tech_colors = {
    tech1_name: "#17A589",  # Teal/cyan for Technology 1
    tech2_name: "#52BE80",  # Medium green for Technology 2
    tech3_name: "#1E8449",  # Dark green for Technology 3
    "Base": "#6E2C00",      # Dark brown for Base/Initial
}

# Additional color constants
COLOR_DARK_BLUE = "#1A5276"
COLOR_LIGHT_BLUE = "#85C1E9"
COLOR_DARK_BROWN = "#6E2C00"

st.markdown("""
To decide whether to adopt a technology, firms compare the upfront investment cost with the future savings from avoided emissions. While the upfront investment occurs on day 0, the savings will be realized over the **project lifetime**. These future savings must be discounted to account for the time value of money and the risk associated with the project, typically using a **discount rate** representing the firm's weighted average cost of capital. Finally, the savings also depend on the **expected carbon price growth** rate, which represents the firm's expectation regarding future mitigation policy.
""")

col1, col2, col3 = st.columns(3)
with col1:
    project_lifetime = st.number_input("Project Lifetime (years)", 5, 40, 20)
    st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;">A higher project lifetime makes future savings more valuable, making investments more attractive.</p>', unsafe_allow_html=True)
with col2:
    discount_rate = st.number_input("Discount Rate", 0.01, 0.20, 0.10, 0.01, format="%.2f")
    st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;">A higher discount rate makes future savings less valuable, making investments less attractive.</p>', unsafe_allow_html=True)
with col3:
    carbon_price_growth = st.number_input("Expected Carbon Price Growth", 0.00, 0.15, 0.05, 0.01, format="%.2f")
    st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;">A higher expected growth rate makes future savings more valuable, making investments more attractive.</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SECTION 1: SIMPLE PLANT MODEL
# ============================================================================
st.header("1. Single Facility Analysis")

st.markdown("""
Let's first examine how these parameters affect a single plant's technology adoption decision.
Here, you need to define the starting intensity of the plant, as well as the technology you want to evaluate.
You will obtain a **carbon price threshold**, which represents the minimum initial carbon price at which a firm would choose to invest in the technology.
At this price, the discounted future savings from avoided emissions exactly offset the upfront investment cost. In other words, the net present value (NPV) of the investment equals zero.
""")

col1, col2 = st.columns(2)
with col1:
    initial_intensity = st.number_input("Starting Intensity (tCO2/t)", 0.20, 1.20, 0.900, 0.01)
with col2:
    selected_tech_name = st.selectbox("Select Technology", [t["name"] for t in technologies])

selected_tech = next(t for t in technologies if t["name"] == selected_tech_name)
final_intensity = get_final_intensity(initial_intensity, selected_tech)
emissions_avoided = initial_intensity - final_intensity

# Calculate threshold
t = np.arange(1, project_lifetime + 1)
years = np.arange(0, project_lifetime + 1)

if emissions_avoided > 0:
    carbon_price_threshold = selected_tech["investment"] / sum(
        emissions_avoided * ((1 + carbon_price_growth) ** year) / ((1 + discount_rate) ** year)
        for year in t
    )
    st.markdown(f"**Carbon Price Threshold for {selected_tech_name}:** {carbon_price_threshold:.1f} EUR/tCO2")
else:
    carbon_price_threshold = float("inf")
    st.warning(f"Technology {selected_tech_name} does not provide emission reduction at this starting intensity.")

# DCF Plot
if emissions_avoided > 0 and carbon_price_threshold != float("inf"):
    carbon_price = carbon_price_threshold * (1 + carbon_price_growth) ** t
    non_discounted = emissions_avoided * carbon_price
    discounted = non_discounted / (1 + discount_rate) ** t
    non_discounted_cf = np.concatenate(([-selected_tech["investment"]], non_discounted))
    discounted_cf = np.concatenate(([-selected_tech["investment"]], discounted))
    npv_over_time = np.cumsum(discounted_cf)
    carbon_price_full = np.concatenate(([carbon_price_threshold], carbon_price))

    # Create two columns: graph on left, explanation on right
    col_dcf_graph, col_dcf_text = st.columns([3, 1])

    with col_dcf_graph:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           row_heights=[0.4, 0.6])

        # Top subplot: Carbon price
        fig.add_trace(go.Scatter(x=years, y=carbon_price_full, mode='lines+markers',
                                 marker=dict(symbol='square', size=8),
                                 line=dict(color=COLOR_DARK_BLUE, width=2),
                                 name='Carbon Price',
                                 hovertemplate='Year %{x}<br>Carbon Price: %{y:.1f} EUR/tCO2<extra></extra>'),
                     row=1, col=1)

        # Bottom subplot: Cash flows
        fig.add_trace(go.Bar(x=years, y=non_discounted_cf, name='Non-discounted',
                            marker_color=COLOR_LIGHT_BLUE, opacity=0.7,
                            hovertemplate='Year %{x}<br>Non-discounted: %{y:.1f} EUR<extra></extra>'),
                     row=2, col=1)
        fig.add_trace(go.Bar(x=years, y=discounted_cf, name='Discounted',
                            marker_color=COLOR_DARK_BLUE,
                            hovertemplate='Year %{x}<br>Discounted: %{y:.1f} EUR<extra></extra>'),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=npv_over_time, mode='lines+markers',
                                 marker=dict(symbol='square', size=8),
                                 line=dict(color=COLOR_DARK_BLUE, width=2),
                                 name='NPV',
                                 hovertemplate='Year %{x}<br>NPV: %{y:.1f} EUR<extra></extra>'),
                     row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            height=400,
            barmode='overlay',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=30, b=50),
            plot_bgcolor='white'
        )
        fig.update_yaxes(title_text="Carbon price (EUR/tCO2)", row=1, col=1, rangemode='tozero',
                        gridcolor='rgba(128,128,128,0.2)', zeroline=False)
        fig.update_yaxes(title_text="Cash flows (EUR)", row=2, col=1,
                        gridcolor='rgba(128,128,128,0.2)', zeroline=False)
        fig.update_xaxes(title_text="Year", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with col_dcf_text:
        st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;"><b>How to read this graph:</b> The top panel shows the expected carbon price trajectory starting from the threshold price. The bottom panel shows the annual cash flows: the initial investment (negative) at year 0, followed by annual savings from avoided emissions. Gray bars represent non-discounted cash flows, black bars represent discounted cash flows, and the black line shows the cumulative NPV over time. At the threshold price, the NPV reaches exactly zero at the end of the project lifetime.</p>', unsafe_allow_html=True)

# Threshold Matrix

st.markdown("""
The decision-making becomes a bit more tricky when multiple technologies are available.
In this case, for a given technology, the corresponding carbon price threshold will depend on whether this is the first technology adopted by the plant or if it has already undergone adoption.
The table below presents these tradeoffs. 
As we can see, for a given technology, the threshold increases if it's adopted as a secondary or third technology. This is explained by the fact that the starting intensity used to evaluate the interest of the new technology is lower than in the base case.
""")

base_intensity = initial_intensity
all_techs = [{"name": "Base", "final_intensity": base_intensity, "investment": 0}]
for tech in technologies:
    final = get_final_intensity(base_intensity, tech)
    all_techs.append({"name": tech["name"], "final_intensity": final, "investment": tech["investment"]})

matrix_data = []
for from_tech in all_techs:
    row = {}
    for to_tech in all_techs:
        if from_tech["name"] == to_tech["name"]:
            row[to_tech["name"]] = "-"
        else:
            from_intensity = from_tech["final_intensity"]
            to_tech_def = next((t for t in technologies if t["name"] == to_tech["name"]), None)
            if to_tech_def:
                to_final = get_final_intensity(from_intensity, to_tech_def)
                threshold = calc_threshold(from_intensity, to_final, to_tech_def["investment"],
                                          project_lifetime, discount_rate, carbon_price_growth)
            else:
                threshold = float("inf")

            if threshold == float("inf"):
                row[to_tech["name"]] = "-"
            else:
                row[to_tech["name"]] = f"{threshold:.1f}"
    matrix_data.append(row)

matrix_df = pd.DataFrame(matrix_data, index=[t["name"] for t in all_techs])
matrix_df.index.name = "From \\ To"

# Create two columns: table on left, explanation on right
col_matrix_table, col_matrix_text = st.columns([3, 1])

with col_matrix_table:
    # Create styled HTML table with colored headers
    tech_names_list = [t["name"] for t in all_techs]

    # Build HTML table
    html_table = '<table style="width:100%; border-collapse: collapse; font-size: 14px;">'

    # Header row
    html_table += '<tr><th style="border: 1px solid #ddd; padding: 8px; background-color: #f5f5f5;">From \\ To</th>'
    for col_name in tech_names_list:
        col_color = tech_colors.get(col_name, "gray")
        html_table += f'<th style="border: 1px solid #ddd; padding: 8px; background-color: #f5f5f5; color: {col_color}; font-weight: bold;">{col_name}</th>'
    html_table += '</tr>'

    # Data rows
    for row_name in tech_names_list:
        row_color = tech_colors.get(row_name, "gray")
        html_table += f'<tr><td style="border: 1px solid #ddd; padding: 8px; color: {row_color}; font-weight: bold;">{row_name}</td>'
        for col_name in tech_names_list:
            value = matrix_df.loc[row_name, col_name]
            html_table += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{value}</td>'
        html_table += '</tr>'

    html_table += '</table>'
    st.markdown(html_table, unsafe_allow_html=True)

with col_matrix_text:
    st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;"><b>How to read this table:</b> Rows represent the current state, columns represent the target technology. Values are carbon price thresholds in EUR/tCO2. A dash (-) indicates the transition is not beneficial (either same technology or the target intensity would be higher than the current one).</p>', unsafe_allow_html=True)

# Carbon price threshold by starting intensity plot
st.markdown("""
The graphic below generalizes the previous table results for all starting intensities.
""")

intensities_t = np.linspace(0.2, 1.2, 200)

# Direct adoption lines - store data for interpretation
direct_thresholds = {}
for tech in technologies:
    thresholds = []
    valid_intensities = []

    for intensity_t in intensities_t:
        final_int = get_final_intensity(intensity_t, tech)
        if final_int < intensity_t:
            threshold = calc_threshold(intensity_t, final_int, tech["investment"],
                                       project_lifetime, discount_rate, carbon_price_growth)
            thresholds.append(threshold)
            valid_intensities.append(intensity_t)

    if valid_intensities:
        direct_thresholds[tech["name"]] = (valid_intensities, thresholds)

# Incremental adoption - compute and find max threshold for second adoption
adoption_data = {}
line_styles_incremental = ["--", ":", "-."]
max_second_adoption_threshold = 0

for start_intensity_t in intensities_t:
    current_intensity_t = start_intensity_t
    remaining_techs = technologies.copy()
    adoption_order = 0

    while remaining_techs:
        eligible = []
        for tech in remaining_techs:
            final_int = get_final_intensity(current_intensity_t, tech)
            if final_int < current_intensity_t:
                eligible.append((tech, final_int))

        if not eligible:
            break

        thresholds_list = []
        for tech, final_int in eligible:
            threshold = calc_threshold(current_intensity_t, final_int, tech["investment"],
                                       project_lifetime, discount_rate, carbon_price_growth)
            thresholds_list.append((threshold, tech, final_int))

        min_threshold, best_tech, best_final = min(thresholds_list, key=lambda x: x[0])

        key = (best_tech["name"], adoption_order)
        if key not in adoption_data:
            adoption_data[key] = []
        adoption_data[key].append((start_intensity_t, min_threshold))

        # Track max threshold for second adoption (order == 1)
        if adoption_order == 1 and min_threshold != float("inf"):
            max_second_adoption_threshold = max(max_second_adoption_threshold, min_threshold)

        current_intensity_t = best_final
        remaining_techs = [t for t in remaining_techs if t["name"] != best_tech["name"]]
        adoption_order += 1

# Calculate default y_max (just above max second adoption threshold, rounded up)
default_y_max = int(np.ceil(max_second_adoption_threshold / 50) * 50) + 50 if max_second_adoption_threshold > 0 else 300

# Generate interpretation text first (needed before displaying)
interpretation_parts = []

# Calculate thresholds at selected intensity for each technology
tech_at_intensity = []
for tech in technologies:
    final_int = get_final_intensity(initial_intensity, tech)
    if final_int < initial_intensity:
        threshold = calc_threshold(initial_intensity, final_int, tech["investment"],
                                   project_lifetime, discount_rate, carbon_price_growth)
        tech_at_intensity.append((tech["name"], threshold, final_int))

tech_at_intensity.sort(key=lambda x: x[1])

if tech_at_intensity:
    first_tech = tech_at_intensity[0]
    first_tech_color = tech_colors.get(first_tech[0], "gray")
    interpretation_parts.append(
        f"If the facility initial intensity is <b>{initial_intensity:.3f} tCO2/t</b>, the first technology to become economically viable is "
        f"<b style=\"color:{first_tech_color}\">{first_tech[0]}</b> at a carbon price of <b>{first_tech[1]:.1f} EUR/tCO2</b> (first intersection between the black vertical bar and a solid curve)."
    )

    # Check for subsequent technologies after first adoption
    new_intensity = first_tech[2]
    remaining = [t for t in technologies if t["name"] != first_tech[0]]

    next_techs = []
    for tech in remaining:
        final_int = get_final_intensity(new_intensity, tech)
        if final_int < new_intensity:
            threshold = calc_threshold(new_intensity, final_int, tech["investment"],
                                       project_lifetime, discount_rate, carbon_price_growth)
            # Also get the direct adoption threshold for comparison
            direct_final = get_final_intensity(initial_intensity, tech)
            direct_threshold = calc_threshold(initial_intensity, direct_final, tech["investment"],
                                              project_lifetime, discount_rate, carbon_price_growth)
            next_techs.append((tech["name"], threshold, direct_threshold))

    if next_techs:
        next_techs.sort(key=lambda x: x[1])
        next_tech = next_techs[0]
        next_tech_color = tech_colors.get(next_tech[0], "gray")
        interpretation_parts.append(
            f"After adopting <b style=\"color:{first_tech_color}\">{first_tech[0]}</b>, the facility intensity drops to <b>{new_intensity:.3f} tCO2/t</b>. "
            f"This lower intensity impacts the carbon price threshold of the remaining technologies. "
            f"For example, <b style=\"color:{next_tech_color}\">{next_tech[0]}</b> now becomes viable at <b>{next_tech[1]:.1f} EUR/tCO2</b> (dotted line), "
            f"which is higher than the direct adoption threshold ({next_tech[2]:.1f} EUR/tCO2) because the emissions reduction is now smaller."
        )

# Create two columns: graph on left, controls and explanation on right
col_graph, col_controls = st.columns([3, 1])

with col_controls:
    y_max = st.number_input("Y-axis limit", 50, 1000, default_y_max, 50, key="ylim_threshold")
    st.markdown(f'<p style="color: gray; font-style: italic; font-size: 0.85em;"><b>How to read this graph:</b> {" ".join(interpretation_parts)}</p>', unsafe_allow_html=True)

with col_graph:
    fig = go.Figure()

    # Plotly dash styles mapping
    dash_styles = {0: 'dash', 1: 'dot', 2: 'dashdot'}

    # Plot direct adoption lines
    for tech_name, (valid_intensities, thresholds) in direct_thresholds.items():
        fig.add_trace(go.Scatter(
            x=valid_intensities, y=thresholds,
            mode='lines',
            name=f"{tech_name} (direct)",
            line=dict(color=tech_colors[tech_name], width=2.5),
            hovertemplate=f'{tech_name}<br>Intensity: %{{x:.3f}} tCO2/t<br>Threshold: %{{y:.1f}} EUR/tCO2<extra></extra>'
        ))

    # Plot incremental adoption lines
    for (tech_name, order), points in sorted(adoption_data.items(), key=lambda x: (x[0][1], x[0][0])):
        points = sorted(points, key=lambda x: x[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        dash_style = dash_styles.get(order, 'dash')
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            name=f"{tech_name} (step {order+1})",
            line=dict(color=tech_colors[tech_name], width=2, dash=dash_style),
            hovertemplate=f'{tech_name} (step {order+1})<br>Intensity: %{{x:.3f}} tCO2/t<br>Threshold: %{{y:.1f}} EUR/tCO2<extra></extra>',
            showlegend=False
        ))

    # Vertical line for selected intensity
    fig.add_vline(x=initial_intensity, line_color=COLOR_DARK_BROWN, line_width=2,
                  annotation_text=f"Selected ({initial_intensity:.3f})", annotation_position="top")

    fig.update_layout(
        height=500,
        xaxis_title="Starting emission intensity (tCO2/t)",
        yaxis_title="Carbon price threshold (EUR/tCO2)",
        xaxis=dict(range=[0.2, 1.2], gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(range=[0, y_max], gridcolor='rgba(128,128,128,0.2)'),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=30, b=50),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# SECTION 2: EMPIRICAL ANALYSIS
# ============================================================================
st.header("2. Sector Analysis")

st.markdown("""
Now that we have understood technology adoption at the facility level, we can simulate the aggregate dynamics at the sector level. You can use the default dataset or upload your own data.
The default dataset represents ~139 clinker production plants located in Europe belonging to large listed firms. It was generated using publicly available data on plant-level production and emissions from the [Global Cement and Concrete Tracker](https://globalenergymonitor.org/projects/global-cement-and-concrete-tracker/) (Global Energy Monitor) and [Climate TRACE](https://climatetrace.org/), as of October 2025.

If you want to upload your own data, the CSV file must contain at least two columns: `production_t_per_yr` (annual production in tonnes) and `production_intensity_tco2_per_t` (emission intensity in tCO2 per tonne of production).
""")

uploaded_file = st.file_uploader("Upload CSV (optional)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(StringIO(DEFAULT_PLANTS_DATA))

# Check required columns
required_cols = ["production_t_per_yr", "production_intensity_tco2_per_t"]
if not all(col in df.columns for col in required_cols):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

# Filter to only use required columns
df = df[required_cols].dropna()
plant_intensities = df["production_intensity_tco2_per_t"].values  # Already in tCO2/t
plant_productions = df["production_t_per_yr"].values

st.write(f"**{len(df)}** plants | Intensity range: **{plant_intensities.min():.3f} - {plant_intensities.max():.3f} tCO2/t** | Total production: **{plant_productions.sum()/1e6:.1f} Mt/yr**")

# Distribution plot (no colors, just intensity bars)
st.markdown("""
As we can see below, the starting intensity of plants in the sector might be heterogeneous, and will therefore lead to different technology adoption decisions among facilities.
""")

# Create two columns: graph on left, explanation on right
col_dist_graph, col_dist_text = st.columns([3, 1])

with col_dist_graph:
    df_sorted = df.sort_values("production_intensity_tco2_per_t", ascending=False).reset_index(drop=True)
    df_sorted['plant_index'] = range(len(df_sorted))
    df_sorted['production_mt'] = df_sorted['production_t_per_yr'] / 1e6

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['plant_index'],
        y=df_sorted['production_intensity_tco2_per_t'],
        marker_color=COLOR_DARK_BROWN,
        hovertemplate='Plant #%{x}<br>Intensity: %{y:.3f} tCO2/t<br>Production: %{customdata:.2f} Mt/yr<extra></extra>',
        customdata=df_sorted['production_mt']
    ))

    fig.update_layout(
        height=250,
        xaxis_title="Plants (ordered by intensity)",
        yaxis_title="Intensity (tCO2/t)",
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        margin=dict(l=50, r=20, t=20, b=50),
        plot_bgcolor='white',
        bargap=0
    )

    st.plotly_chart(fig, use_container_width=True)

with col_dist_text:
    st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;"><b>How to read this graph:</b> Each bar represents a plant, ordered from highest to lowest emission intensity. The height of the bar indicates the plant\'s carbon intensity (tCO2 per tonne of production).</p>', unsafe_allow_html=True)

# Carbon Trajectory Settings
st.markdown("""
To simulate the sector dynamics, you can define a carbon price trajectory over 25 years. At each year, plants evaluate whether to adopt a new technology following the same decision process as in the previous section. Once a technology is adopted, the plant's intensity decreases, but alternative technologies may remain available for future adoption.

Note that in this model, the decision at each year is based on the observed carbon price, not the actual future trajectory. Companies still use the expected carbon price growth rate defined in the Parameters section to evaluate their investment decisions.
""")

col1, col2, col3 = st.columns(3)
with col1:
    start_carbon_price = st.number_input("Carbon Price Year 0 (EUR/tCO2)", 0, 500, 0)
with col2:
    end_carbon_price = st.number_input("Carbon Price Year 25 (EUR/tCO2)", 0, 1000, 300)
with col3:
    trajectory_shape = st.selectbox("Trajectory Shape", ["Linear", "Concave", "Convex"])

trajectory_years = 25
carbon_trajectory = generate_carbon_trajectory(start_carbon_price, end_carbon_price, trajectory_years, trajectory_shape)

# Simulate incremental adoption over time
tech_names = ["Base"] + [tech["name"] for tech in technologies]

# Store counts, production, and emissions by year
tech_counts_over_time = {name: [] for name in tech_names}
tech_production_over_time = {name: [] for name in tech_names}
total_emissions_over_time = []

# Initialize plant states
plant_states = [
    {"intensity": intensity, "production": prod, "current_tech": "Base", "available_techs": technologies.copy()}
    for intensity, prod in zip(plant_intensities, plant_productions)
]

for year_idx, cp in enumerate(carbon_trajectory):
    # For each plant, check if they want to adopt a new technology
    for plant in plant_states:
        while True:
            best_tech = find_optimal_technology_incremental(
                plant["intensity"], cp, plant["available_techs"],
                project_lifetime, discount_rate, carbon_price_growth
            )
            if best_tech is None:
                break

            new_intensity = get_final_intensity(plant["intensity"], best_tech)
            plant["intensity"] = new_intensity
            plant["current_tech"] = best_tech["name"]
            plant["available_techs"] = [t for t in plant["available_techs"] if t["name"] != best_tech["name"]]

    # Count technologies, production, and emissions
    counts = {name: 0 for name in tech_names}
    production = {name: 0.0 for name in tech_names}
    total_emissions = 0.0
    for plant in plant_states:
        counts[plant["current_tech"]] += 1
        production[plant["current_tech"]] += plant["production"]
        # Emissions = production * intensity (intensity is in tCO2/t, production in t/yr)
        total_emissions += plant["production"] * plant["intensity"]

    for name in tech_names:
        tech_counts_over_time[name].append(counts[name])
        tech_production_over_time[name].append(production[name] / 1e6)  # Convert to Mt

    total_emissions_over_time.append(total_emissions / 1e6)  # Convert to MtCO2

# Toggle for count vs production
display_mode = st.toggle("Show Production (Mt) instead of Plant Count", value=False)

data_to_plot = tech_production_over_time if display_mode else tech_counts_over_time
y_label = "Production (Mt/yr)" if display_mode else "Number of plants"

years_range = np.arange(trajectory_years)

# Create two columns: graph on left, explanation on right
col_adoption_graph, col_adoption_text = st.columns([3, 1])

with col_adoption_text:
    # How to read section
    st.markdown('<p style="color: gray; font-style: italic; font-size: 0.85em;"><b>How to read this graph:</b> The top panel shows the carbon price trajectory that you defined (solid line) and the resulting total sector emissions (dotted line). The bottom panel shows technology adoption: the stacked bars indicate how many plants (or production volume) use each technology at each year. Hover over the chart to see exact values.</p>', unsafe_allow_html=True)

# Set unit for hover templates
unit = "Mt/yr" if display_mode else "plants"

with col_adoption_graph:
    # Create two subplots with secondary y-axis on top
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                       row_heights=[0.4, 0.6],
                       specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

    # Subplot 1: Carbon price (left axis)
    fig.add_trace(go.Scatter(
        x=years_range, y=carbon_trajectory,
        mode='lines+markers',
        marker=dict(symbol='square', size=6),
        line=dict(color=COLOR_DARK_BLUE, width=2.5),
        name='Carbon Price',
        hovertemplate='Year %{x}<br>Carbon Price: %{y:.0f} EUR/tCO2<extra></extra>'
    ), row=1, col=1, secondary_y=False)

    # Subplot 1: Emissions (right axis)
    fig.add_trace(go.Scatter(
        x=years_range, y=total_emissions_over_time,
        mode='lines+markers',
        marker=dict(symbol='square', size=6),
        line=dict(color=COLOR_DARK_BROWN, width=2.5, dash='dot'),
        name='Emissions',
        hovertemplate='Year %{x}<br>Emissions: %{y:.1f} MtCO2/yr<extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # Subplot 2: Technology adoption (stacked bar)
    for name in tech_names:
        fig.add_trace(go.Bar(
            x=years_range,
            y=data_to_plot[name],
            name=name,
            marker_color=tech_colors.get(name, "gray"),
            hovertemplate=f'{name}<br>Year %{{x}}<br>Value: %{{y:.1f}} {unit}<extra></extra>'
        ), row=2, col=1)

    fig.update_layout(
        height=650,
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white'
    )

    # Update axes
    fig.update_yaxes(title_text="Carbon Price (EUR/tCO2)", row=1, col=1, secondary_y=False,
                    title_font=dict(color=COLOR_DARK_BLUE), tickfont=dict(color=COLOR_DARK_BLUE),
                    gridcolor='rgba(128,128,128,0.2)', rangemode='tozero')
    fig.update_yaxes(title_text="Emissions (MtCO2/yr)", row=1, col=1, secondary_y=True,
                    title_font=dict(color=COLOR_DARK_BROWN), tickfont=dict(color=COLOR_DARK_BROWN),
                    gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text=y_label, row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(title_text="Year", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

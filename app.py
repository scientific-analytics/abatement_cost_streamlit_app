import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

The tool is organised in three sections. **Parameters** is where you can define/modify parameters related to the decision-making process of firmsand the characteristics of three mitigation technologies.
Then, the **Single Facility Analysis** replicates a typical decision process at the facility level. Given its initial carbon intensity, at what carbon price is it effective to invest in a cleaner technology?
Finally, the **Sector Analysis** extends the results of the facility analysis to a set of facilities representative of a sector/region (you can upload your own list of facility characteristics).

The model is by definition very simplified, and it's main purpose is to be pedagogical.
""")

st.markdown('<p style="color: gray; font-style: italic;">Last update: March 2026. For any questions or feedback, please contact vincent.bouchet [at] scientificportfolio.com</p>', unsafe_allow_html=True)

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
st.header("Parameters")

st.subheader("Global Parameters")
st.markdown("""
These parameters define the economic context for investment decisions:
- **Project Lifetime**: How long the technology investment will generate benefits (depreciation horizon)
- **Discount Rate**: The rate used to discount future cash flows (reflects cost of capital and risk)
- **Expected Carbon Price Growth**: Annual growth rate firms expect for carbon prices when making decisions
""")

col1, col2, col3 = st.columns(3)
with col1:
    project_lifetime = st.number_input("Project Lifetime (years)", 5, 40, 20)
with col2:
    discount_rate = st.number_input("Discount Rate", 0.01, 0.20, 0.10, 0.01, format="%.2f")
with col3:
    carbon_price_growth = st.number_input("Expected Carbon Price Growth", 0.00, 0.15, 0.05, 0.01, format="%.2f")

st.subheader("Technology Definitions")
st.markdown("""
Define three decarbonization technologies. Each technology is characterized by:
- **Improvement %**: Percentage reduction in emission intensity from the starting point
- **Target Range**: The achievable intensity range (min-max) after applying the technology
- **Investment**: Capital expenditure required per unit of production capacity (EUR per tonne)
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Technology 1**")
    tech1_name = st.text_input("Name", "Alternative fuels", key="tech1_name")
    t1c1, t1c2 = st.columns(2)
    with t1c1:
        tech1_improvement = st.number_input("Improvement %", 0, 100, 10, key="tech1_imp")
        tech1_target_min = st.number_input("Target Min (tCO2/t)", 0.0, 1.5, 0.700, 0.01, key="tech1_min")
    with t1c2:
        tech1_investment = st.number_input("Investment (EUR/t)", 0, 5000, 10, key="tech1_inv")
        tech1_target_max = st.number_input("Target Max (tCO2/t)", 0.0, 1.5, 0.850, 0.01, key="tech1_max")

with col2:
    st.markdown("**Technology 2**")
    tech2_name = st.text_input("Name", "Energy efficiency", key="tech2_name")
    t2c1, t2c2 = st.columns(2)
    with t2c1:
        tech2_improvement = st.number_input("Improvement %", 0, 100, 40, key="tech2_imp")
        tech2_target_min = st.number_input("Target Min (tCO2/t)", 0.0, 1.5, 0.400, 0.01, key="tech2_min")
    with t2c2:
        tech2_investment = st.number_input("Investment (EUR/t)", 0, 5000, 100, key="tech2_inv")
        tech2_target_max = st.number_input("Target Max (tCO2/t)", 0.0, 1.5, 0.600, 0.01, key="tech2_max")

with col3:
    st.markdown("**Technology 3**")
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
    tech1_name: "#E67E22",
    tech2_name: "#82E0AA",
    tech3_name: "#27AE60",
    "Base": "#8B4513",
}

st.markdown("---")

# ============================================================================
# SECTION 1: SIMPLE PLANT MODEL
# ============================================================================
st.header("1. Simple Plant Model")

st.markdown("""
Let's first examine how a single plant makes its technology adoption decision.
The **carbon price threshold** is the minimum initial carbon price at which a firm would choose to invest in a technology.
At this price, the net present value (NPV) of the investment equals zero: the discounted future savings from avoided emissions exactly offset the upfront investment cost.
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
    st.success(f"**Carbon Price Threshold for {selected_tech_name}:** {carbon_price_threshold:.1f} EUR/tCO2")
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax1.plot(years, carbon_price_full, color="black", marker="s", linewidth=2)
    ax1.set_ylabel("Carbon price (EUR/tCO2)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2.bar(years, non_discounted_cf, alpha=0.3, color="gray", label="Non-discounted")
    ax2.bar(years, discounted_cf, color="black", label="Discounted")
    ax2.plot(years, npv_over_time, marker="s", linewidth=2, color="black", label="NPV")
    ax2.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cash flows (EUR)")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Threshold Matrix
st.subheader("Transition Threshold Matrix")
st.markdown("""
As there are usually multiple technologies available (with different investment costs and resulting improvements), this section also presents how these technologies might be adopted sequentially (i.e., if you adopt a first technology, you might need to wait longer before moving to another than if you had directly adopted technology B).This matrix shows the carbon price threshold (in EUR/tCO2) required to make each technology transition economically viable.
Rows represent the current state, columns represent the target technology. A dash (-) indicates the transition is not beneficial
(either same technology or the target intensity would be higher than the current one).
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
st.dataframe(matrix_df, use_container_width=True)

# Carbon price threshold by starting intensity plot
st.subheader("Carbon Price Threshold by Starting Intensity")

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

# Y-axis limit control
y_max = st.number_input("Y-axis maximum (EUR/tCO2)", 50, 1000, default_y_max, 50, key="ylim_threshold")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot direct adoption lines
for tech_name, (valid_intensities, thresholds) in direct_thresholds.items():
    ax.plot(valid_intensities, thresholds, label=f"{tech_name} (direct)",
            color=tech_colors[tech_name], linewidth=2.5, linestyle="-")

# Plot incremental adoption lines
for (tech_name, order), points in sorted(adoption_data.items(), key=lambda x: (x[0][1], x[0][0])):
    points = sorted(points, key=lambda x: x[0])
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    linestyle = line_styles_incremental[order % len(line_styles_incremental)]
    ax.plot(x_vals, y_vals, color=tech_colors[tech_name], linestyle=linestyle, linewidth=2)

# Vertical line for selected intensity
ax.axvline(x=initial_intensity, color="black", linestyle="-", linewidth=2, label=f"Selected ({initial_intensity:.3f} tCO2/t)")

ax.set_xlabel("Starting emission intensity (tCO2/t)")
ax.set_ylabel("Carbon price threshold (EUR/tCO2)")
ax.set_xlim(0.2, 1.2)
ax.set_ylim(0, y_max)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Generate interpretation text
st.markdown("**How to read this graph:**")

# Find thresholds at selected intensity
interpretation_parts = []
adoption_sequence = []

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
    interpretation_parts.append(
        f"Starting at **{initial_intensity:.3f} tCO2/t**, the first technology to become economically viable is "
        f"**{first_tech[0]}** at a carbon price of **{first_tech[1]:.1f} EUR/tCO2** (solid line intersection with the vertical bar)."
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
            next_techs.append((tech["name"], threshold))

    if next_techs:
        next_techs.sort(key=lambda x: x[1])
        next_tech = next_techs[0]
        interpretation_parts.append(
            f"After adopting {first_tech[0]}, your intensity drops to **{new_intensity:.3f} tCO2/t**. "
            f"From this lower intensity, **{next_tech[0]}** becomes viable at **{next_tech[1]:.1f} EUR/tCO2** "
            f"(dashed line). This is higher than the direct adoption threshold because the emissions reduction is smaller."
        )

    # Show all direct thresholds
    all_direct = ", ".join([f"{t[0]}: {t[1]:.1f}" for t in tech_at_intensity])
    interpretation_parts.append(f"Direct adoption thresholds at {initial_intensity:.3f} tCO2/t: {all_direct} EUR/tCO2.")

st.markdown(" ".join(interpretation_parts))

st.markdown("---")

# ============================================================================
# SECTION 2: EMPIRICAL ANALYSIS
# ============================================================================
st.header("2. Empirical Analysis")

st.markdown("""
Now we apply the framework to a portfolio of plants. Upload your own data or use the default dataset of ~139 clinker production plants.
The CSV must contain at least two columns: `production_t_per_yr` (annual production) and `production_intensity_tco2_per_t` (emission intensity in tCO2 per tonne).
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

st.write(f"Loaded **{len(df)}** plants | Intensity range: **{plant_intensities.min():.3f} - {plant_intensities.max():.3f} tCO2/t** | Total production: **{plant_productions.sum()/1e6:.1f} Mt/yr**")

# Distribution plot (no colors, just intensity bars)
st.subheader("Distribution of Initial Intensities")

df_sorted = df.sort_values("production_intensity_tco2_per_t", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(range(len(df_sorted)), df_sorted["production_intensity_tco2_per_t"], width=1, color="gray")
ax.set_xlabel("Plants (ordered by intensity)")
ax.set_ylabel("Intensity (tCO2/t)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Carbon Trajectory Settings
st.subheader("Technology Adoption Over Time")

st.markdown("""
Define a carbon price trajectory over 25 years. At each year, plants evaluate whether to adopt a new technology
based on the current carbon price and their expected growth rate. Once a technology is adopted, the plant's
intensity decreases and it can no longer adopt that same technology.
""")

col1, col2, col3 = st.columns(3)
with col1:
    start_carbon_price = st.number_input("Starting Carbon Price (EUR/tCO2)", 0, 500, 50)
with col2:
    end_carbon_price = st.number_input("Ending Carbon Price (EUR/tCO2)", 0, 1000, 300)
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

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Subplot 1: Carbon price and emissions
ax1.plot(years_range, carbon_trajectory, color="black", linewidth=2.5, linestyle="-", marker="s", markersize=4, label="Carbon Price")
ax1.set_ylabel("Carbon Price (EUR/tCO2)")
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(years_range, total_emissions_over_time, color="black", linewidth=2.5, linestyle=":", marker="s", markersize=4, label="Emissions")
ax1_twin.set_ylabel("Emissions (MtCO2/yr)")
ax1_twin.legend(loc="upper right")

# Subplot 2: Technology adoption (stacked bar)
bottom = np.zeros(trajectory_years)
for name in tech_names:
    ax2.bar(years_range, data_to_plot[name], bottom=bottom, width=0.8,
            label=name, color=tech_colors.get(name, "gray"))
    bottom += np.array(data_to_plot[name])

ax2.set_xlabel("Year")
ax2.set_ylabel(y_label)
ax2.set_xlim(-0.5, trajectory_years - 0.5)
ax2.legend(title="Technology", loc="upper left")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Year selector for summary - compact layout
year_options = list(range(trajectory_years))
col_label, col_select, col_spacer = st.columns([1, 1, 4])
with col_label:
    st.markdown("**Select Year:**")
with col_select:
    selected_year = st.selectbox("Year", year_options, index=trajectory_years - 1, label_visibility="collapsed")

# Build summary table
if display_mode:
    final_values = {name: tech_production_over_time[name][selected_year] for name in tech_names}
    unit = "Mt/yr"
else:
    final_values = {name: tech_counts_over_time[name][selected_year] for name in tech_names}
    unit = "plants"

# Create summary as a dataframe table
summary_data = {
    "Carbon Price": [f"{carbon_trajectory[selected_year]:.0f} EUR/tCO2"],
    "Emissions": [f"{total_emissions_over_time[selected_year]:.1f} MtCO2/yr"],
}
for name in tech_names:
    value = final_values[name]
    if display_mode:
        summary_data[name] = [f"{value:.1f} {unit}"]
    else:
        summary_data[name] = [f"{int(value)} {unit}"]

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

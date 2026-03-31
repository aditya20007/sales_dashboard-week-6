"""
=============================================================
  Interactive Sales Dashboard  —  dashboard.py
  Week 6: Data Visualization Mastery with Seaborn & Plotly
  Author : Senior Engineer (5 yrs)
=============================================================
Run:
    python dashboard.py          # generates all PNGs + HTML dashboard
    python dashboard.py --show   # additionally opens the HTML in browser
"""

import os
import sys
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe on any server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 0.  CONSTANTS & PATHS
# ──────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "sales_data.csv")
VIZ_DIR     = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Brand palette (consistent across every chart)
PALETTE     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
PRODUCT_CLR = dict(zip(["Laptop", "Phone", "Tablet", "Headphones", "Monitor"], PALETTE))
REGION_CLR  = {"East": "#4C72B0", "West": "#DD8452", "North": "#55A868", "South": "#C44E52"}

# Seaborn global theme
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# ──────────────────────────────────────────────────────────
# 1.  DATA LOADING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Month"]       = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Month_Label"] = df["Date"].dt.strftime("%b %Y")
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["Revenue"]     = df["Total_Sales"]          # alias for clarity
    return df

# ──────────────────────────────────────────────────────────
# 2.  CHART 1 — SEABORN BOX PLOT: Price Distribution
# ──────────────────────────────────────────────────────────
def chart_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("Product")["Price"].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df, x="Product", y="Price",
        order=order, palette=PALETTE,
        width=0.55, linewidth=1.4,
        flierprops=dict(marker="o", markerfacecolor="#888", markersize=5, alpha=0.6),
        ax=ax,
    )
    # Overlay stripplot for individual points
    sns.stripplot(
        data=df, x="Product", y="Price",
        order=order, color="#222", size=3.5, alpha=0.35, jitter=True, ax=ax,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_title("Price Distribution by Product", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Product Category", fontsize=12)
    ax.set_ylabel("Unit Price (₹)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)

    # Annotate medians
    medians = df.groupby("Product")["Price"].median()
    for i, prod in enumerate(order):
        ax.text(i, medians[prod] + 600, f"₹{medians[prod]:,.0f}",
                ha="center", va="bottom", fontsize=9, color="#333", fontweight="bold")

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "01_boxplot_price_distribution.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out}")
    return out

# ──────────────────────────────────────────────────────────
# 3.  CHART 2 — SEABORN VIOLIN PLOT: Sales by Region
# ──────────────────────────────────────────────────────────
def chart_violin(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("Region")["Revenue"].median().sort_values(ascending=False).index

    sns.violinplot(
        data=df, x="Region", y="Revenue",
        order=order, palette=list(REGION_CLR.values()),
        inner="quartile", linewidth=1.3, ax=ax,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e3:,.0f}K"))
    ax.set_title("Revenue Distribution by Region", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Sales Region", fontsize=12)
    ax.set_ylabel("Total Sales Revenue (₹)", fontsize=12)

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "02_violin_revenue_by_region.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out}")
    return out

# ──────────────────────────────────────────────────────────
# 4.  CHART 3 — SEABORN HEATMAP: Correlation Matrix
# ──────────────────────────────────────────────────────────
def chart_heatmap_corr(df: pd.DataFrame):
    num_cols = ["Quantity", "Price", "Total_Sales"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle only

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#ddd",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix — Numerical Variables",
                 fontsize=15, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "03_heatmap_correlation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out}")
    return out

# ──────────────────────────────────────────────────────────
# 5.  CHART 4 — SEABORN HEATMAP: Product × Region Revenue
# ──────────────────────────────────────────────────────────
def chart_heatmap_pivot(df: pd.DataFrame):
    pivot = df.pivot_table(values="Revenue", index="Product",
                           columns="Region", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot, annot=True, fmt=",.0f",
        cmap="YlOrRd", linewidths=0.4, linecolor="#eee",
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title("Total Revenue: Product × Region (₹)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Product", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "04_heatmap_product_region.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓  Saved: {out}")
    return out

# ──────────────────────────────────────────────────────────
# 6.  CHART 5 — SEABORN 2×2 SUBPLOT DASHBOARD
# ──────────────────────────────────────────────────────────
def chart_multiplot(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sales Overview Dashboard", fontsize=18, fontweight="bold", y=1.01)

    # A) Bar: Revenue by Product
    rev_prod = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    bars = axes[0, 0].bar(rev_prod.index, rev_prod.values / 1e6,
                           color=PALETTE, edgecolor="white", linewidth=0.8)
    axes[0, 0].set_title("Revenue by Product (₹M)", fontweight="bold")
    axes[0, 0].set_ylabel("Revenue (₹ Millions)")
    for bar, val in zip(bars, rev_prod.values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"₹{val/1e6:.1f}M", ha="center", fontsize=9, fontweight="bold")

    # B) Horizontal Bar: Revenue by Region
    rev_reg = df.groupby("Region")["Revenue"].sum().sort_values()
    colors_reg = [REGION_CLR[r] for r in rev_reg.index]
    axes[0, 1].barh(rev_reg.index, rev_reg.values / 1e6,
                    color=colors_reg, edgecolor="white")
    axes[0, 1].set_title("Revenue by Region (₹M)", fontweight="bold")
    axes[0, 1].set_xlabel("Revenue (₹ Millions)")
    for i, val in enumerate(rev_reg.values):
        axes[0, 1].text(val / 1e6 + 0.01, i, f"₹{val/1e6:.1f}M",
                        va="center", fontsize=9, fontweight="bold")

    # C) Line: Monthly Revenue trend
    monthly = df.groupby("Month")["Revenue"].sum().reset_index()
    axes[1, 0].plot(monthly["Month"], monthly["Revenue"] / 1e6,
                    marker="o", color=PALETTE[0], linewidth=2.2, markersize=7)
    axes[1, 0].fill_between(monthly["Month"], monthly["Revenue"] / 1e6,
                             alpha=0.15, color=PALETTE[0])
    axes[1, 0].set_title("Monthly Revenue Trend", fontweight="bold")
    axes[1, 0].set_ylabel("Revenue (₹ Millions)")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%b %Y") if hasattr(matplotlib, "dates") else plt.NullFormatter()
    )

    # D) Pie: Quantity share by Product
    qty_prod = df.groupby("Product")["Quantity"].sum()
    wedges, texts, autotexts = axes[1, 1].pie(
        qty_prod.values, labels=qty_prod.index,
        autopct="%1.1f%%", colors=PALETTE,
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold")
    axes[1, 1].set_title("Units Sold by Product", fontweight="bold")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "05_multiplot_overview.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved: {out}")
    return out

# ──────────────────────────────────────────────────────────
# 7.  CHART 6 — PLOTLY INTERACTIVE: Sales Trend Line
# ──────────────────────────────────────────────────────────
def chart_plotly_trend(df: pd.DataFrame):
    monthly_prod = (
        df.groupby(["Month", "Product"])["Revenue"]
        .sum()
        .reset_index()
    )
    monthly_prod["Month_str"] = monthly_prod["Month"].dt.strftime("%b %Y")

    fig = px.line(
        monthly_prod, x="Month_str", y="Revenue",
        color="Product", markers=True,
        color_discrete_map=PRODUCT_CLR,
        title="<b>Monthly Revenue Trend by Product</b>",
        labels={"Revenue": "Revenue (₹)", "Month_str": "Month", "Product": "Product"},
        template="plotly_white",
    )
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=13),
        title_font_size=18,
        yaxis_tickformat="₹,.0f",
        height=450,
    )
    out = os.path.join(VIZ_DIR, "06_plotly_monthly_trend.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  ✓  Saved: {out}")
    return fig

# ──────────────────────────────────────────────────────────
# 8.  CHART 7 — PLOTLY INTERACTIVE: Bubble Chart
# ──────────────────────────────────────────────────────────
def chart_plotly_bubble(df: pd.DataFrame):
    agg = (
        df.groupby(["Product", "Region"])
        .agg(Total_Revenue=("Revenue", "sum"),
             Avg_Price=("Price", "mean"),
             Total_Qty=("Quantity", "sum"))
        .reset_index()
    )

    fig = px.scatter(
        agg, x="Avg_Price", y="Total_Revenue",
        size="Total_Qty", color="Product",
        facet_col="Region",
        color_discrete_map=PRODUCT_CLR,
        title="<b>Revenue vs Avg Price (bubble = quantity sold)</b>",
        labels={"Avg_Price": "Avg Unit Price (₹)",
                "Total_Revenue": "Total Revenue (₹)",
                "Total_Qty": "Units Sold"},
        template="plotly_white",
        size_max=45,
        hover_data={"Total_Qty": True, "Total_Revenue": ":,.0f", "Avg_Price": ":,.0f"},
    )
    fig.update_layout(
        height=420,
        font=dict(family="Arial", size=12),
        title_font_size=17,
        showlegend=True,
    )
    out = os.path.join(VIZ_DIR, "07_plotly_bubble_chart.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  ✓  Saved: {out}")
    return fig

# ──────────────────────────────────────────────────────────
# 9.  CHART 8 — PLOTLY INTERACTIVE: Sunburst (Customer Seg)
# ──────────────────────────────────────────────────────────
def chart_plotly_sunburst(df: pd.DataFrame):
    fig = px.sunburst(
        df, path=["Region", "Product"], values="Revenue",
        color="Region",
        color_discrete_map=REGION_CLR,
        title="<b>Customer Segmentation: Region → Product Revenue</b>",
        template="plotly_white",
    )
    fig.update_traces(
        textinfo="label+percent entry",
        hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>%{percentParent:.1%} of parent<extra></extra>",
    )
    fig.update_layout(height=480, font=dict(family="Arial", size=13), title_font_size=17)
    out = os.path.join(VIZ_DIR, "08_plotly_sunburst.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  ✓  Saved: {out}")
    return fig

# ──────────────────────────────────────────────────────────
# 10. FULL PLOTLY DASHBOARD (multi-subplot HTML)
# ──────────────────────────────────────────────────────────
def build_plotly_dashboard(df: pd.DataFrame):
    # Pre-compute aggregates
    rev_prod    = df.groupby("Product")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
    rev_reg     = df.groupby("Region")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
    monthly     = df.groupby("Month")["Revenue"].sum().reset_index()
    monthly_prod= df.groupby(["Month", "Product"])["Revenue"].sum().reset_index()
    pivot       = df.pivot_table("Revenue", "Product", "Region", aggfunc="sum")
    qty_prod    = df.groupby("Product")["Quantity"].sum().reset_index()
    box_data    = [(prod, grp["Price"].tolist()) for prod, grp in df.groupby("Product")]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Revenue by Product",
            "Revenue by Region",
            "Monthly Revenue Trend",
            "Units Sold by Product",
            "Price Distribution (Box)",
            "Revenue Heatmap: Product × Region",
        ),
        specs=[
            [{"type": "bar"},  {"type": "bar"}],
            [{"type": "scatter", "colspan": 2}, None],
            [{"type": "box"},  {"type": "heatmap"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # Row 1 Col 1 — Bar product
    fig.add_trace(
        go.Bar(
            x=rev_prod["Product"], y=rev_prod["Revenue"],
            marker_color=[PRODUCT_CLR[p] for p in rev_prod["Product"]],
            text=[f"₹{v/1e6:.1f}M" for v in rev_prod["Revenue"]],
            textposition="outside",
            name="Product Revenue",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Row 1 Col 2 — Bar region
    fig.add_trace(
        go.Bar(
            x=rev_reg["Region"], y=rev_reg["Revenue"],
            marker_color=[REGION_CLR[r] for r in rev_reg["Region"]],
            text=[f"₹{v/1e6:.1f}M" for v in rev_reg["Revenue"]],
            textposition="outside",
            name="Region Revenue",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Row 2 — Multi-line monthly trend (full width)
    for prod, grp in monthly_prod.groupby("Product"):
        grp = grp.sort_values("Month")
        fig.add_trace(
            go.Scatter(
                x=grp["Month"], y=grp["Revenue"],
                mode="lines+markers",
                name=prod,
                line=dict(color=PRODUCT_CLR[prod], width=2.2),
                marker=dict(size=7),
                hovertemplate=f"<b>{prod}</b><br>%{{x|%b %Y}}<br>₹%{{y:,.0f}}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Row 3 Col 1 — Box plot price
    for prod, prices in box_data:
        fig.add_trace(
            go.Box(
                y=prices, name=prod,
                marker_color=PRODUCT_CLR[prod],
                boxmean="sd",
                showlegend=False,
            ),
            row=3, col=1,
        )

    # Row 3 Col 2 — Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="YlOrRd",
            text=[[f"₹{v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(len=0.35, y=0.12),
        ),
        row=3, col=2,
    )

    # Layout polish
    fig.update_layout(
        title=dict(
            text="<b>🛒 Interactive Sales Performance Dashboard</b>",
            font=dict(size=22, family="Arial"),
            x=0.5,
        ),
        height=1100,
        template="plotly_white",
        paper_bgcolor="#F9FAFB",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
        ),
    )
    fig.update_yaxes(tickformat="₹,.0f", row=1, col=1)
    fig.update_yaxes(tickformat="₹,.0f", row=1, col=2)
    fig.update_yaxes(tickformat="₹,.0f", row=2, col=1)
    fig.update_yaxes(tickformat="₹,.0f", row=3, col=1)

    out = os.path.join(os.path.dirname(__file__), "interactive_dashboard.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"  ✓  Saved: {out}")
    return out

# ──────────────────────────────────────────────────────────
# 11. ENTRY POINT
# ──────────────────────────────────────────────────────────
def main(show: bool = False):
    print("\n🚀  Sales Dashboard Generator  —  loading data …")
    df = load_data()
    print(f"   Loaded {len(df)} rows × {len(df.columns)} cols  |  "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}\n")

    print("📊  Generating Seaborn charts …")
    chart_boxplot(df)
    chart_violin(df)
    chart_heatmap_corr(df)
    chart_heatmap_pivot(df)
    chart_multiplot(df)

    print("\n📈  Generating Plotly interactive charts …")
    chart_plotly_trend(df)
    chart_plotly_bubble(df)
    chart_plotly_sunburst(df)

    print("\n🏗️  Building full interactive HTML dashboard …")
    dash_path = build_plotly_dashboard(df)

    print("\n✅  All done!  Summary:")
    print(f"   Static charts  →  {VIZ_DIR}/")
    print(f"   Dashboard HTML →  {dash_path}")

    if show:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(dash_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the sales dashboard.")
    parser.add_argument("--show", action="store_true", help="Open dashboard in browser after generation.")
    args = parser.parse_args()
    main(show=args.show)
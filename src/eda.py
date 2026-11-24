import pandas as pd
import plotly.express as px
from src.preprocessing import find_column


def chart_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#292524",
        margin=dict(l=20, r=20, t=20, b=20),
        hoverlabel=dict(bgcolor="#292524", font_color="#FFFFFF"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E7E5E4", zeroline=False)
    return fig


def price_distribution(df, target):
    fig = px.histogram(df, x=target, nbins=40)
    return chart_theme(fig)


def brand_average_price(df, target):
    col = find_column(df, ("brand", "make"))
    if not col:
        return None
    summary = df.groupby(col, dropna=True)[target].mean().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(summary, x=col, y=target)
    fig.update_layout(xaxis_tickangle=-45)
    return chart_theme(fig)


def categorical_distribution(df, keyword):
    col = find_column(df, (keyword,))
    if not col:
        return None
    counts = df[col].fillna("Unknown").value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.pie(counts, names=col, values="Count", hole=0.45)
    return chart_theme(fig)


def year_price_trend(df, target):
    col = find_column(df, ("vehicleage", "year"))
    if not col:
        return None
    trend = df.groupby(col, dropna=True)[target].median().reset_index()
    fig = px.line(trend, x=col, y=target, markers=True)
    return chart_theme(fig)


def mileage_price(df, target):
    km_col = find_column(df, ("km_driven", "kilometer", "odometer"))
    if not km_col:
        return None
    fig = px.scatter(df, x=km_col, y=target, color=find_column(df, ("fuel",)) or None,
                     opacity=0.55)
    return chart_theme(fig)


def correlation_heatmap(df):
    numeric = df.select_dtypes(include="number").copy()
    if len(numeric.columns) < 2:
        return None
    fig = px.imshow(numeric.corr(), text_auto=".2f", aspect="auto")
    return chart_theme(fig)

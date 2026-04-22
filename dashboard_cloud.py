"""
Toronto 311 Service Requests — Cloud Dashboard
────────────────────────────────────────────────
Fetches data live from the City of Toronto Open Data API.
No local data files needed — deploy anywhere.

Run locally:
    pip install dash dash-bootstrap-components plotly pandas numpy statsmodels requests
    python dashboard_cloud.py

Deploy to Railway / Render:
    - Push this file + fetch_open_data.py + ward_boundaries.geojson + requirements_cloud.txt to GitHub
    - Connect repo to Railway or Render, set start command to:
        python dashboard_cloud.py
"""

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from statsmodels.tsa.statespace.sarimax import SARIMAX

from fetch_open_data import load_311_data

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (works both local and on hosting platforms)
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys
BASE_DIR     = Path(_sys.executable).parent if getattr(_sys, "frozen", False) else Path(__file__).parent
GEOJSON_FILE = BASE_DIR / "ward_boundaries.geojson"

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLOURS
# ─────────────────────────────────────────────────────────────────────────────
DIV_COLOURS = {
    "Forestry"                        : "#1a7340",
    "Parks and Recreation"            : "#f4a261",
    "Municipal Licensing & Standards" : "#264653",
    "Solid Waste Management Services" : "#e9c46a",
    "Toronto Water"                   : "#2196f3",
    "Transportation Services"         : "#9c27b0",
    "311 / Other"                     : "#90a4ae",
}
STATUS_COLOURS = {
    "Resolved"   : "#27ae60",
    "In Progress": "#f39c12",
    "New / Open" : "#3498db",
    "Cancelled"  : "#e74c3c",
    "Unknown"    : "#95a5a6",
}
ACCENT   = "#003F88"
BG       = "#F2F4F8"
CARD_BG  = "#FFFFFF"
COVID_S  = pd.Timestamp("2020-03-01")
COVID_E  = pd.Timestamp("2022-06-30")
MONTH_AB = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  — from Toronto Open Data API
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Loading 311 data from Toronto Open Data API …")
print("    (first run downloads ~90 MB and caches locally — subsequent runs take ~5 sec)")
df_api = load_311_data()

# Rename API columns to match dashboard internals
df_api = df_api.rename(columns={
    "service_request_type": "service_type",
    "fsa":                  "fsa",
})

print(f"✅  {len(df_api):,} rows loaded")

# ── Feature engineering ───────────────────────────────────────────────────────
df = df_api.copy()
df["year_month"] = df["creation_date"].dt.to_period("M")

# ward_num: already extracted by _normalise() in fetch_open_data.py
# but stored as float — convert for consistency
if "ward" in df.columns and "ward_num" not in df.columns:
    df["ward_num"] = df["ward"].astype(str).str.extract(r"\((\d+)\)")[0]
elif "ward" in df.columns:
    # ward_num comes from fetch_open_data._normalise as a float column
    df["ward_num"] = df["ward"].astype(float).astype("Int64").astype(str)
    df["ward_num"] = df["ward_num"].replace("<NA>", np.nan)

# FSA: validate Toronto postcodes (start with M)
if "fsa" in df.columns:
    df["fsa_clean"] = df["fsa"].where(
        df["fsa"].astype(str).str.match(r"^M[A-Z0-9]{2}$", na=False), np.nan)
else:
    df["fsa_clean"] = np.nan

DIVISION_MAP = {
    "Urban Forestry"                  : "Forestry",
    "Parks, Forestry & Recreation"    : "Forestry",
    "Environment, Climate & Forestry" : "Forestry",
    "Parks and Recreation"            : "Parks and Recreation",
    "Municipal Licensing & Standards" : "Municipal Licensing & Standards",
    "Solid Waste Management Services" : "Solid Waste Management Services",
    "Toronto Water"                   : "Toronto Water",
    "Transportation Services"         : "Transportation Services",
    "311"                             : "311 / Other",
    "Unknown"                         : "311 / Other",
}
df["division_clean"] = df["division"].map(DIVISION_MAP).fillna(df["division"])

STATUS_MAP = {
    "Closed":"Resolved","Completed":"Resolved",
    "Initiated":"New / Open","New":"New / Open",
    "In-progress ":"In Progress","In-Progress":"In Progress","In Progress":"In Progress",
    "Cancelled":"Cancelled","Unknown":"Unknown",
}
df["status_clean"] = df["status"].astype(str).str.strip().map(STATUS_MAP).fillna("Unknown")

# Ward labels: "W03 – Etobicoke–Lakeshore"
if "ward_name" in df.columns:
    _wn = df[["ward_num","ward_name"]].drop_duplicates("ward_num").dropna(subset=["ward_num"])
    WARD_LABELS = {str(row.ward_num): f"W{int(float(row.ward_num)):02d} – {row.ward_name}"
                   for row in _wn.itertuples()}
else:
    WARD_LABELS = {}

df_main = df[(df["year"] >= 2010) & (df["year"] <= 2026)].copy()
df_main["ward_label"] = df_main["ward_num"].map(WARD_LABELS).fillna(
    df_main["ward_num"].apply(lambda w: f"Ward {w}" if pd.notna(w) else "Unknown"))

# Temporal features for drill-down
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",
              9:"Fall",10:"Fall",11:"Fall"}
DOW_MAP    = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",
              4:"Friday",5:"Saturday",6:"Sunday"}
df_main["season"]   = df_main["month"].map(SEASON_MAP)
df_main["dow_name"] = df_main["creation_date"].dt.dayofweek.map(DOW_MAP)
df_main["hour"]     = df_main["creation_date"].dt.hour

# Dropdown lists
DIVS_ALL     = sorted(df_main["division_clean"].dropna().unique())
STATUSES_ALL = sorted(df_main["status_clean"].dropna().unique())
WARDS_ALL    = sorted(
    [(str(num), WARD_LABELS.get(str(num), f"Ward {num}"))
     for num in df_main["ward_num"].dropna().unique()],
    key=lambda x: int(x[0]) if str(x[0]).replace(".","").isdigit() else 99
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD WARD GeoJSON  (pre-built file in repo — no geopandas needed)
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Loading ward boundaries …")
with open(GEOJSON_FILE, "r") as f:
    WARD_GEOJSON = json.load(f)
print("✅  Ward boundaries loaded")

# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX FORECAST  (computed once at startup)
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Fitting SARIMAX model …")
df_f = df[(df["division_clean"] == "Forestry") & (df["year"] <= 2025)].copy()
ts_raw = df_f.groupby("year_month").size().reset_index(name="count")
ts_raw["date"] = ts_raw["year_month"].dt.to_timestamp()
MONTHLY_TS = ts_raw.set_index("date")["count"].asfreq("MS")
ts_filled  = MONTHLY_TS.interpolate(method="linear").bfill()

covid_x = pd.Series(
    ((MONTHLY_TS.index >= COVID_S) & (MONTHLY_TS.index <= COVID_E)).astype(int),
    index=MONTHLY_TS.index, name="covid")

train   = ts_filled[:"2022-12"]
test    = MONTHLY_TS["2023-01":]
exog_tr = covid_x[:"2022-12"].to_frame()
exog_te = covid_x["2023-01":].to_frame()

sarima = SARIMAX(train, exog=exog_tr, order=(1,1,1),
                 seasonal_order=(1,1,1,12),
                 enforce_stationarity=False,
                 enforce_invertibility=False).fit(disp=False)

N_AHEAD  = len(test) + 12
exog_fut = pd.DataFrame({"covid":[0]*N_AHEAD},
           index=pd.date_range("2023-01-01", periods=N_AHEAD, freq="MS"))
fc       = sarima.get_forecast(steps=N_AHEAD, exog=exog_fut)
FC_MEAN  = fc.predicted_mean
FC_CI    = fc.conf_int()
FITTED   = sarima.fittedvalues
tp       = sarima.apply(test, exog=exog_te).fittedvalues
FC_MAPE  = float(np.mean(np.abs((tp-test)/test))*100)
FC_MAE   = float(np.mean(np.abs(tp-test)))
FC_RMSE  = float(np.sqrt(np.mean((tp-test)**2)))
print(f"✅  Model ready  MAPE={FC_MAPE:.1f}%  MAE={FC_MAE:.0f}  RMSE={FC_RMSE:.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE STYLE SNIPPETS
# ─────────────────────────────────────────────────────────────────────────────
CARD_STYLE = {
    "borderRadius":"14px",
    "boxShadow":"0 2px 12px rgba(0,0,0,0.07)",
    "border":"none",
    "background":CARD_BG,
}
LBL = {
    "fontWeight":"700","fontSize":"11px","color":"#6B7280",
    "textTransform":"uppercase","letterSpacing":"0.7px","marginBottom":"6px",
}
CHART_CFG = {"displayModeBar":False}

def plotly_base():
    return dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARD BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, color):
    return dbc.Card(style={
        **CARD_STYLE,
        "borderLeft": f"5px solid {color}",
        "padding": "18px 22px",
        "height": "100%",
    }, children=[
        html.P(title, style={**LBL, "marginBottom":"4px"}),
        html.H3(value, style={
            "fontWeight":"800","margin":"0 0 4px","color":"#111827","fontSize":"28px",
            "lineHeight":"1.1"
        }),
        html.P(subtitle, style={"color":"#9CA3AF","fontSize":"12px","margin":0}),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    title="Toronto 311 Dashboard",
    meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}],
)
server = app.server   # expose Flask server for Gunicorn (needed by Railway/Render)

app.layout = html.Div(
    style={"fontFamily":"Inter, sans-serif","background":BG,"minHeight":"100vh"},
    children=[

    html.Div(style={
        "background":f"linear-gradient(120deg, {ACCENT} 0%, #1565C0 100%)",
        "padding":"22px 36px 18px","color":"white",
    }, children=[
        html.Div(style={"display":"flex","alignItems":"center","justifyContent":"space-between"}, children=[
            html.Div([
                html.H4("Toronto 311 Service Requests",
                        style={"margin":0,"fontWeight":"800","fontSize":"24px","letterSpacing":"-0.3px"}),
                html.P("Parks & Recreation Division  ·  2010–2026  ·  Live from Toronto Open Data",
                       style={"margin":"4px 0 0","opacity":"0.78","fontSize":"13px"}),
            ]),
            html.Div(style={"display":"flex","gap":"10px"}, children=[
                html.Span("Live Data", style={
                    "background":"rgba(255,255,255,0.15)","borderRadius":"20px",
                    "padding":"5px 14px","fontSize":"12px","fontWeight":"600",
                }),
                html.Span("5 Tabs", style={
                    "background":"rgba(255,255,255,0.15)","borderRadius":"20px",
                    "padding":"5px 14px","fontSize":"12px","fontWeight":"600",
                }),
            ]),
        ])
    ]),

    dbc.Container(fluid=True, style={"padding":"20px 28px"}, children=[

        dbc.Card(style={**CARD_STYLE,"padding":"20px 28px","marginBottom":"18px"}, children=[
            dbc.Row(align="end", children=[

                dbc.Col([
                    html.Div("Year Range", style=LBL),
                    dcc.RangeSlider(
                        id="sl-year", min=2010, max=2026, step=1, value=[2010,2026],
                        marks={y: {"label":str(y),"style":{"fontSize":"11px"}}
                               for y in range(2010, 2027, 2)},
                        tooltip={"placement":"bottom","always_visible":False},
                    )
                ], md=4, style={"paddingRight":"24px"}),

                dbc.Col([
                    html.Div("Division", style=LBL),
                    dcc.Dropdown(
                        id="dd-div",
                        options=[{"label":d,"value":d} for d in DIVS_ALL],
                        value=[], multi=True, placeholder="All divisions",
                        style={"fontSize":"13px"},
                    ),
                ], md=3),

                dbc.Col([
                    html.Div("Ward", style=LBL),
                    dcc.Dropdown(
                        id="dd-ward",
                        options=[{"label":lbl,"value":num} for num,lbl in WARDS_ALL],
                        value=[], multi=True, placeholder="All wards",
                        style={"fontSize":"13px"},
                    ),
                ], md=3),

                dbc.Col([
                    html.Div("Status", style=LBL),
                    dcc.Dropdown(
                        id="dd-status",
                        options=[{"label":s,"value":s} for s in STATUSES_ALL],
                        value=[], multi=True, placeholder="All statuses",
                        style={"fontSize":"13px"},
                    ),
                ], md=2),
            ])
        ]),

        dbc.Row(style={"marginBottom":"18px","rowGap":"12px"}, children=[
            dbc.Col(id="kpi-1", md=3),
            dbc.Col(id="kpi-2", md=3),
            dbc.Col(id="kpi-3", md=3),
            dbc.Col(id="kpi-4", md=3),
        ]),

        dbc.Card(style=CARD_STYLE, children=[
            dcc.Tabs(
                id="tabs", value="t-overview",
                colors={"border":"#e5e7eb","primary":ACCENT,"background":"#f9fafb"},
                style={"borderBottom":"1px solid #e5e7eb"},
                children=[
                    dcc.Tab(label="Overview",   value="t-overview",
                            style={"padding":"12px 20px","fontWeight":"500","fontSize":"13px"},
                            selected_style={"padding":"12px 20px","fontWeight":"700",
                                            "fontSize":"13px","borderTop":f"3px solid {ACCENT}","color":ACCENT}),
                    dcc.Tab(label="Divisions",  value="t-divisions",
                            style={"padding":"12px 20px","fontWeight":"500","fontSize":"13px"},
                            selected_style={"padding":"12px 20px","fontWeight":"700",
                                            "fontSize":"13px","borderTop":f"3px solid {ACCENT}","color":ACCENT}),
                    dcc.Tab(label="Geography",  value="t-geo",
                            style={"padding":"12px 20px","fontWeight":"500","fontSize":"13px"},
                            selected_style={"padding":"12px 20px","fontWeight":"700",
                                            "fontSize":"13px","borderTop":f"3px solid {ACCENT}","color":ACCENT}),
                    dcc.Tab(label="Forecast",   value="t-forecast",
                            style={"padding":"12px 20px","fontWeight":"500","fontSize":"13px"},
                            selected_style={"padding":"12px 20px","fontWeight":"700",
                                            "fontSize":"13px","borderTop":f"3px solid {ACCENT}","color":ACCENT}),
                    dcc.Tab(label="Temporal",   value="t-temporal",
                            style={"padding":"12px 20px","fontWeight":"500","fontSize":"13px"},
                            selected_style={"padding":"12px 20px","fontWeight":"700",
                                            "fontSize":"13px","borderTop":f"3px solid {ACCENT}","color":ACCENT}),
                ],
            ),
            dcc.Store(id="drill-store", data={"filters":[]}),
            dcc.Loading(
                type="circle", color=ACCENT,
                children=html.Div(id="tab-body", style={"padding":"24px"}),
            ),
        ]),

    ]),
])

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-1",    "children"),
    Output("kpi-2",    "children"),
    Output("kpi-3",    "children"),
    Output("kpi-4",    "children"),
    Output("tab-body", "children"),
    Input("sl-year",   "value"),
    Input("dd-div",    "value"),
    Input("dd-ward",   "value"),
    Input("dd-status", "value"),
    Input("tabs",      "value"),
)
def update(years, divs, wards, statuses, tab):
    try:
        tab   = tab or "t-overview"
        y0, y1 = years or [2010, 2026]

        m = (df_main["year"] >= y0) & (df_main["year"] <= y1)
        if divs:     m &= df_main["division_clean"].isin(divs)
        if wards:    m &= df_main["ward_num"].isin(wards)
        if statuses: m &= df_main["status_clean"].isin(statuses)
        dff = df_main[m].copy()

        total = len(dff)

        if total:
            vc          = dff["division_clean"].value_counts()
            top_div     = vc.index[0]
            top_div_pct = vc.iloc[0] / total * 100
            wvc         = dff["ward_num"].value_counts()
            top_w_num   = wvc.index[0] if len(wvc) else "—"
            top_w_lbl   = WARD_LABELS.get(str(top_w_num), f"Ward {top_w_num}")
            resolved    = int((dff["status_clean"] == "Resolved").sum())
            res_rate    = resolved / total * 100
            avg_mo      = dff.groupby("year_month").size().mean()
        else:
            top_div = "—"; top_div_pct = 0.0
            top_w_lbl = "—"; resolved = 0; res_rate = 0.0; avg_mo = 0.0

        k1 = kpi_card("Total Requests",  f"{total:,}",
                      f"{y0}–{y1}  ·  filtered view", ACCENT)
        k2 = kpi_card("Top Division",    str(top_div)[:24],
                      f"{top_div_pct:.1f}% of all requests",
                      DIV_COLOURS.get(top_div, "#888"))
        k3 = kpi_card("Top Ward",        str(top_w_lbl)[:24],
                      "Highest cumulative volume", "#e67e22")
        k4 = kpi_card("Resolution Rate", f"{res_rate:.1f}%",
                      f"{resolved:,} resolved  ·  avg {avg_mo:,.0f}/month", "#27ae60")

        if   tab == "t-overview":  body = _overview(dff)
        elif tab == "t-divisions": body = _divisions(dff)
        elif tab == "t-geo":       body = _geography(dff)
        elif tab == "t-temporal":  body = _temporal_shell()
        else:                      body = _forecast()

        return k1, k2, k3, k4, body

    except Exception as e:
        import traceback
        err = dbc.Alert(
            [html.Strong("Error: "), str(e), html.Br(),
             html.Pre(traceback.format_exc(), style={"fontSize":"11px","marginTop":"8px"})],
            color="danger", style={"margin":"20px"}
        )
        blank = html.Div()
        return blank, blank, blank, blank, err


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def _overview(dff):
    yr_d = dff.groupby(["year","division_clean"]).size().reset_index(name="count")
    fig1 = go.Figure()
    for div in yr_d["division_clean"].unique():
        d = yr_d[yr_d["division_clean"]==div].sort_values("year")
        fig1.add_trace(go.Scatter(
            x=d["year"], y=d["count"], mode="lines+markers", name=div,
            line=dict(color=DIV_COLOURS.get(div,"#888"), width=2.5),
            marker=dict(size=6, color=DIV_COLOURS.get(div,"#888")),
            hovertemplate=f"<b>{div}</b><br>%{{x}}: %{{y:,}} requests<extra></extra>",
        ))
    fig1.add_vrect(x0=2020, x1=2022.5, fillcolor="rgba(150,150,150,0.12)",
                   layer="below", line_width=0,
                   annotation_text="COVID-19", annotation_position="top left",
                   annotation_font_size=10, annotation_font_color="#aaa")
    fig1.update_layout(**plotly_base(),
        title="Annual Request Volume by Division",
        xaxis=dict(title="Year", showgrid=False, dtick=2),
        yaxis=dict(title="Requests", showgrid=True, gridcolor="#F3F4F6", separatethousands=True),
        hovermode="x unified", height=360,
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
    )

    mo_d = dff.groupby(["year","month"]).size().reset_index(name="count")
    hm   = mo_d.pivot(index="year", columns="month", values="count").fillna(0)
    hm.columns = [MONTH_AB[c-1] for c in hm.columns]
    fig2 = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns.tolist(), y=[str(y) for y in hm.index.tolist()],
        colorscale="YlOrRd", colorbar=dict(title="Reqs", thickness=12),
        hovertemplate="<b>%{y}  %{x}</b><br>%{z:,.0f} requests<extra></extra>",
    ))
    fig2.update_layout(**plotly_base(),
        title="Monthly Volume Heatmap (Year × Month)",
        xaxis_title="Month", yaxis_title="",
        height=360, margin=dict(t=55,b=35,l=50,r=10),
    )

    dff2 = dff.copy()
    dff2["season"] = dff2["month"].map(SEASON_MAP)
    seas = dff2["season"].value_counts().reset_index()
    seas.columns = ["season","count"]
    SCOLS = {"Summer":"#e74c3c","Spring":"#2ecc71","Fall":"#e67e22","Winter":"#3498db"}
    fig3 = go.Figure(go.Pie(
        labels=seas["season"], values=seas["count"], hole=0.58,
        marker_colors=[SCOLS.get(s,"#888") for s in seas["season"]],
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig3.update_layout(**plotly_base(),
        title="Requests by Season",
        height=360, margin=dict(t=55,b=20,l=0,r=0),
        legend=dict(orientation="h", y=-0.08),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig1, config=CHART_CFG), md=8),
            dbc.Col(dcc.Graph(figure=fig3, config=CHART_CFG), md=4),
        ], style={"marginBottom":"16px"}),
        dcc.Graph(figure=fig2, config=CHART_CFG),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DIVISIONS
# ─────────────────────────────────────────────────────────────────────────────
def _divisions(dff):
    yr_d = dff.groupby(["year","division_clean"]).size().reset_index(name="count")
    fig1 = go.Figure()
    for div in yr_d["division_clean"].unique():
        d = yr_d[yr_d["division_clean"]==div].sort_values("year")
        fig1.add_trace(go.Bar(
            x=d["year"], y=d["count"], name=div,
            marker_color=DIV_COLOURS.get(div,"#888"),
            hovertemplate=f"<b>{div}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    fig1.update_layout(**plotly_base(),
        barmode="stack", title="Requests by Year — Stacked by Division",
        xaxis=dict(title="Year", dtick=2, showgrid=False),
        yaxis=dict(title="Requests", showgrid=True, gridcolor="#F3F4F6", separatethousands=True),
        height=340, margin=dict(t=55,b=70,l=10,r=10),
        legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
    )

    sts = dff["status_clean"].value_counts().reset_index()
    sts.columns = ["status","count"]
    fig2 = go.Figure(go.Pie(
        labels=sts["status"], values=sts["count"], hole=0.55,
        marker_colors=[STATUS_COLOURS.get(s,"#888") for s in sts["status"]],
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig2.update_layout(**plotly_base(),
        title="Request Status Breakdown",
        height=340, margin=dict(t=55,b=20,l=0,r=0),
        legend=dict(orientation="h", y=-0.08),
    )

    svc = (dff.groupby("service_type").size().reset_index(name="count")
           .sort_values("count", ascending=False).head(20))
    svc["cumulative_pct"] = svc["count"].cumsum() / svc["count"].sum() * 100
    n80    = int((svc["cumulative_pct"] <= 80).sum()) + 1
    colors = [DIV_COLOURS.get("Parks and Recreation","#f4a261")]*n80 + ["#D1D5DB"]*(len(svc)-n80)
    fig3 = go.Figure(go.Bar(
        x=svc["service_type"], y=svc["count"],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>%{y:,} requests<extra></extra>",
        text=svc["count"].apply(lambda v: f"{v/1e3:.1f}K"),
        textposition="outside", textfont=dict(size=9),
    ))
    fig3.add_shape(type="line", xref="x", yref="paper",
                   x0=n80-0.5, x1=n80-0.5, y0=0, y1=1,
                   line=dict(color="#c0392b", width=1.8, dash="dash"))
    fig3.add_annotation(x=n80-0.5, yref="paper", y=0.97,
                        text=f"Top {n80} types = 80% of volume",
                        showarrow=False, xanchor="left",
                        font=dict(size=10, color="#c0392b"), bgcolor="white", opacity=0.85)
    fig3.update_layout(**plotly_base(),
        title="Top 20 Service Types — Highlighted = 80% of Volume",
        xaxis=dict(showgrid=False, tickangle=-38, title=""),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", separatethousands=True, title="Requests"),
        showlegend=False, height=360, margin=dict(t=55,b=140,l=10,r=10),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig1, config=CHART_CFG), md=8),
            dbc.Col(dcc.Graph(figure=fig2, config=CHART_CFG), md=4),
        ], style={"marginBottom":"16px"}),
        dcc.Graph(figure=fig3, config=CHART_CFG),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GEOGRAPHY
# ─────────────────────────────────────────────────────────────────────────────
def _geography(dff):
    dff_w = dff[dff["year"] >= 2018].copy()
    wrd = (dff_w[dff_w["ward_num"].notna()]
           .groupby(["ward_num","ward_label"]).size().reset_index(name="count"))
    wrd["ward_key"] = wrd["ward_num"].astype(str).str.split(".").str[0].str.zfill(2)

    fig_map = px.choropleth_mapbox(
        wrd, geojson=WARD_GEOJSON,
        locations="ward_key", featureidkey="properties.WARD_KEY",
        color="count", color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=9.8, center={"lat":43.72,"lon":-79.38}, opacity=0.75,
        hover_name="ward_label",
        hover_data={"count":":,","ward_key":False},
        labels={"count":"Total Requests"},
    )
    fig_map.update_layout(**plotly_base(),
        title="Total Requests by Ward (2018–present)",
        height=500, margin=dict(t=55,b=0,l=0,r=0),
        coloraxis_colorbar=dict(title="Requests", thickness=14, len=0.7),
    )

    top_w = wrd.sort_values("count", ascending=True).tail(15)
    fig_bar = go.Figure(go.Bar(
        x=top_w["count"], y=top_w["ward_label"], orientation="h",
        marker=dict(color=top_w["count"], colorscale="YlOrRd", showscale=False),
        text=top_w["count"].apply(lambda v: f"{v/1e3:.1f}K"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} requests<extra></extra>",
    ))
    fig_bar.update_layout(**plotly_base(),
        title="Top 15 Wards by Request Volume",
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6", separatethousands=True, title="Requests"),
        yaxis=dict(title=""),
        height=500, margin=dict(t=55,b=40,l=10,r=70),
    )

    fsa = (dff[dff["fsa_clean"].notna()]
           .groupby("fsa_clean").size().reset_index(name="count")
           .sort_values("count", ascending=False).head(15))
    fig_fsa = go.Figure(go.Bar(
        x=fsa["fsa_clean"], y=fsa["count"],
        marker=dict(color=fsa["count"], colorscale="Blues", showscale=False),
        hovertemplate="<b>FSA %{x}</b><br>%{y:,} requests<extra></extra>",
        text=fsa["count"].apply(lambda v: f"{v/1e3:.1f}K"),
        textposition="outside",
    ))
    fig_fsa.update_layout(**plotly_base(),
        title="Top 15 Postal Code Areas (FSA) by Volume",
        xaxis=dict(title="FSA", showgrid=False),
        yaxis=dict(title="Requests", showgrid=True, gridcolor="#F3F4F6", separatethousands=True),
        height=300, margin=dict(t=55,b=50,l=10,r=10),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_map, config={"displayModeBar":True}), md=7),
            dbc.Col(dcc.Graph(figure=fig_bar, config=CHART_CFG), md=5),
        ], style={"marginBottom":"16px"}),
        dcc.Graph(figure=fig_fsa, config=CHART_CFG),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
def _forecast():
    def _fmt(idx): return [d.strftime("%B %Y") for d in idx]

    fig = go.Figure()
    fig.add_vrect(x0=str(COVID_S), x1=str(COVID_E),
                  fillcolor="rgba(150,150,150,0.14)", layer="below", line_width=0,
                  annotation_text="COVID-19", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#999")
    fig.add_trace(go.Scatter(
        x=MONTHLY_TS.index, y=MONTHLY_TS.values, mode="lines",
        name="Observed", line=dict(color=DIV_COLOURS["Forestry"], width=2.2),
        customdata=list(zip(_fmt(MONTHLY_TS.index), MONTHLY_TS.values)),
        hovertemplate="<b>%{customdata[0]}</b><br>Observed: <b>%{customdata[1]:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=FITTED.index, y=FITTED.values, mode="lines",
        name="Fitted (training)", line=dict(color="#9CA3AF", width=1.2, dash="dash"),
        customdata=list(zip(_fmt(FITTED.index), FITTED.values)),
        hovertemplate="<b>%{customdata[0]}</b><br>Fitted: <b>%{customdata[1]:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(FC_CI.index)+list(FC_CI.index[::-1]),
        y=list(FC_CI.iloc[:,1])+list(FC_CI.iloc[:,0][::-1]),
        fill="toself", fillcolor="rgba(231,76,60,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=FC_MEAN.index, y=FC_MEAN.values, mode="lines",
        name="Forecast", line=dict(color="#e74c3c", width=2.8),
        customdata=list(zip(_fmt(FC_MEAN.index), FC_MEAN.values,
                            FC_CI.iloc[:,0].values, FC_CI.iloc[:,1].values)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Forecast: <b>%{customdata[1]:,.0f}</b><br>"
            "95% CI: %{customdata[2]:,.0f} – %{customdata[3]:,.0f}<extra></extra>"
        ),
    ))
    for xdate, color, label, xanch in [
        ("2023-01-01","#6B7280","Train / Test split","right"),
        ("2026-01-01","#e74c3c","Forecast horizon",  "left"),
    ]:
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=xdate, x1=xdate, y0=0, y1=1,
                      line=dict(color=color, width=1.4, dash="dot"))
        fig.add_annotation(x=xdate, yref="paper", y=0.97, text=label,
                           showarrow=False, font=dict(size=10, color=color),
                           xanchor=xanch, bgcolor="white", opacity=0.85)
    fig.update_layout(**plotly_base(),
        title=dict(
            text=(f"SARIMAX Forecast — Forestry Requests (2010–2026)<br>"
                  f"<sup>MAPE: {FC_MAPE:.1f}%  |  MAE: {FC_MAE:.0f} req/month"
                  f"  |  RMSE: {FC_RMSE:.0f} req/month</sup>"),
            font=dict(size=15),
        ),
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(title="Monthly Requests", showgrid=True, gridcolor="#F3F4F6", separatethousands=True),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.14, font=dict(size=11)),
        height=520, margin=dict(t=95,b=70,l=10,r=10),
    )
    return html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar":True,"scrollZoom":True}),
        dbc.Alert([html.Strong("Note: "),
                   "Forecast is for Forestry — the only division with 16 years of data "
                   "sufficient for seasonal modelling."],
                  color="info", style={"fontSize":"13px","marginTop":"14px","borderRadius":"10px"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — TEMPORAL DRILL-DOWN
# ─────────────────────────────────────────────────────────────────────────────
DRILL_LEVELS = ["Year", "Season", "Month", "Day of Week", "Hour of Day"]
DRILL_COLS   = ["year", "season", "month_name", "dow_name", "hour"]
SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]
DOW_ORDER    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
SCOLS_MAP    = {"Summer":"#e74c3c","Spring":"#2ecc71","Fall":"#e67e22","Winter":"#3498db"}

def _temporal_shell():
    return html.Div([
        dbc.Card(style={
            "background":"#F9FAFB","border":"1px solid #E5E7EB",
            "borderRadius":"10px","padding":"14px 20px","marginBottom":"16px"
        }, children=[
            dbc.Row(align="center", children=[
                dbc.Col(html.Div("Group by:", style={
                    "fontWeight":"700","fontSize":"12px","color":"#6B7280",
                    "textTransform":"uppercase","letterSpacing":"0.6px","whiteSpace":"nowrap"
                }), width="auto"),
                dbc.Col(dcc.RadioItems(
                    id="drill-level",
                    options=[
                        {"label": " Year",        "value": 0},
                        {"label": " Season",      "value": 1},
                        {"label": " Month",       "value": 2},
                        {"label": " Day of Week", "value": 3},
                        {"label": " Hour of Day", "value": 4},
                    ],
                    value=0, inline=True,
                    inputStyle={"marginRight":"5px","cursor":"pointer"},
                    labelStyle={"marginRight":"22px","fontSize":"13px","fontWeight":"500",
                                "cursor":"pointer","color":"#374151"},
                )),
                dbc.Col(dbc.Button("Clear Filter", id="drill-reset", n_clicks=0,
                                   size="sm", color="secondary", outline=True,
                                   style={"fontSize":"12px","float":"right"}), width="auto"),
            ]),
        ]),
        html.Div(id="drill-crumb", style={
            "fontSize":"12px","color":"#6B7280","marginBottom":"10px",
            "fontWeight":"500","minHeight":"18px"
        }),
        dcc.Graph(id="drill-chart", config={"displayModeBar":False}),
        html.P("Click any bar to filter, then change Group by to explore another dimension.",
               style={"fontSize":"12px","color":"#9CA3AF","marginTop":"8px","textAlign":"center"}),
    ])

def _build_drill_fig(dff, level, path):
    filtered = dff.copy()
    for i, val in enumerate(path):
        col = DRILL_COLS[i]
        filtered = filtered[filtered[col] == (int(val) if col == "year" else val)]

    col = DRILL_COLS[level]
    if col == "month_name":
        filtered["month_name"] = filtered["creation_date"].dt.strftime("%b")
    grouped = filtered.groupby(col).size().reset_index(name="count")

    if level == 1:
        grouped[col] = pd.Categorical(grouped[col], categories=SEASON_ORDER, ordered=True)
    elif level == 2:
        grouped[col] = pd.Categorical(grouped[col], categories=MONTH_AB, ordered=True)
    elif level == 3:
        grouped[col] = pd.Categorical(grouped[col], categories=DOW_ORDER, ordered=True)
    grouped = grouped.sort_values(col)
    grouped[col] = grouped[col].astype(str)

    if level == 0:   colors = [ACCENT]*len(grouped)
    elif level == 1: colors = [SCOLS_MAP.get(s,"#888") for s in grouped[col]]
    elif level == 3: colors = ["#e74c3c" if d in ["Saturday","Sunday"] else ACCENT for d in grouped[col]]
    else:
        n = max(len(grouped),1)
        colors = [f"hsl({int(200+140*i/n)},60%,48%)" for i in range(n)]

    crumb_parts = [f"{DRILL_LEVELS[i]}: <b>{path[i]}</b>" for i in range(len(path))]
    title_str   = " → ".join(crumb_parts) if crumb_parts else "All Years"
    next_label  = DRILL_LEVELS[level+1] if level < 4 else None
    subtitle    = f"Click a bar to drill into {next_label}" if next_label else "Deepest level"

    fig = go.Figure(go.Bar(
        x=grouped[col], y=grouped["count"],
        marker_color=colors,
        text=grouped["count"].apply(lambda v: f"{v/1e3:.1f}K" if v>=1000 else f"{int(v):,}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} requests<extra></extra>",
        customdata=grouped[col],
    ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        title=dict(text=f"{title_str}<br><sup style='color:#9CA3AF'>{subtitle}</sup>",
                   font=dict(size=14)),
        xaxis=dict(title=DRILL_LEVELS[level], showgrid=False, tickangle=-30 if level==2 else 0),
        yaxis=dict(title="Requests", showgrid=True, gridcolor="#F3F4F6", separatethousands=True),
        height=460, margin=dict(t=80,b=80,l=10,r=10),
    )
    return fig

@app.callback(
    Output("drill-chart", "figure"),
    Output("drill-crumb", "children"),
    Output("drill-store", "data"),
    Input("drill-chart",  "clickData"),
    Input("drill-reset",  "n_clicks"),
    Input("drill-level",  "value"),
    Input("sl-year",      "value"),
    Input("dd-div",       "value"),
    Input("dd-ward",      "value"),
    Input("dd-status",    "value"),
    dash.dependencies.State("drill-store", "data"),
    prevent_initial_call=False,
)
def update_drill(clickData, n_reset, radio_level, years, divs, wards, statuses, store):
    try:
        from dash import ctx
        trigger = ctx.triggered_id if ctx.triggered_id else "drill-level"
    except Exception:
        trigger = "drill-level"

    store = store or {"filters":[]}
    y0, y1 = years or [2010, 2026]
    m = (df_main["year"] >= y0) & (df_main["year"] <= y1)
    if divs:     m &= df_main["division_clean"].isin(divs)
    if wards:    m &= df_main["ward_num"].isin(wards)
    if statuses: m &= df_main["status_clean"].isin(statuses)
    dff = df_main[m].copy()
    dff["month_name"] = dff["creation_date"].dt.strftime("%b")

    if trigger == "drill-reset":
        store = {"filters":[]}
    elif trigger in ("sl-year","dd-div","dd-ward","dd-status"):
        store = {"filters":[]}
    elif trigger == "drill-chart" and clickData:
        display_col = DRILL_COLS[radio_level or 0]
        clicked_val = str(clickData["points"][0]["x"])
        existing    = [f for f in store.get("filters",[]) if f["col"] != display_col]
        store       = {"filters": existing + [{"col": display_col, "val": clicked_val}]}

    filters = store.get("filters",[])
    for f in filters:
        col, val = f["col"], f["val"]
        dff = dff[dff[col] == (int(val) if col == "year" else val)]

    level = radio_level if radio_level is not None else 0
    fig   = _build_drill_fig(dff, level, [])

    if filters:
        parts = [f"{DRILL_LEVELS[DRILL_COLS.index(f['col'])]}: {f['val']}"
                 for f in filters if f["col"] in DRILL_COLS]
        crumb = [html.Span("Active filters: ", style={"fontWeight":"600"}),
                 html.Span("  ·  ".join(parts), style={"color":ACCENT}),
                 html.Span("  — click 'Clear Filter' to reset", style={"color":"#9CA3AF"})]
    else:
        crumb = html.Span("No filters active — showing all data", style={"color":"#9CA3AF"})

    return fig, crumb, store


app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser, time

    def _open_browser():
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:8050")

    threading.Thread(target=_open_browser, daemon=True).start()
    print("\n  Toronto 311 Cloud Dashboard starting...")
    print("  Opening http://127.0.0.1:8050\n")
    # Use host="0.0.0.0" so it works on cloud platforms too
    port = int(__import__("os").environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)

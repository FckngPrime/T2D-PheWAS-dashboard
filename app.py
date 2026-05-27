import os

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (Dash, dcc, html, dash_table,
                  Input, Output, State, no_update, callback_context)

SCRIPT_DIR       = Path(__file__).parent
DATA_PATH        = SCRIPT_DIR / "final_results.parquet"
ICD10_NAMES_PATH = SCRIPT_DIR / "icd102019syst_codes.txt"
Q_SIGNIFICANT   = 0.05
EPS_QVAL        = 1e-300

# ICD-10 chapters: (short_id, display_label, first_code, last_code)
ICD10_CHAPTERS = [
    ("A-B",    "A00-B99 - Certain infectious and parasitic diseases",                                                              "A00", "B99"),
    ("C-D48",  "C00-D48 - Neoplasms",                                                                                              "C00", "D48"),
    ("D50-89", "D50-D89 - Diseases of the blood and blood-forming organs<br>and certain disorders involving the immune mechanism", "D50", "D89"),
    ("E",      "E00-E90 - Endocrine, nutritional and metabolic diseases",                                                          "E00", "E90"),
    ("F",      "F00-F99 - Mental and behavioural disorders",                                                                       "F00", "F99"),
    ("G",      "G00-G99 - Diseases of the nervous system",                                                                         "G00", "G99"),
    ("H0-59",  "H00-H59 - Diseases of the eye and adnexa",                                                                         "H00", "H59"),
    ("H60-95", "H60-H95 - Diseases of the ear and mastoid process",                                                                "H60", "H95"),
    ("I",      "I00-I99 - Diseases of the circulatory system",                                                                     "I00", "I99"),
    ("J",      "J00-J99 - Diseases of the respiratory system",                                                                     "J00", "J99"),
    ("K",      "K00-K93 - Diseases of the digestive system",                                                                       "K00", "K93"),
    ("L",      "L00-L99 - Diseases of the skin and subcutaneous tissue",                                                           "L00", "L99"),
    ("M",      "M00-M99 - Diseases of the musculoskeletal system<br>and connective tissue",                                        "M00", "M99"),
    ("N",      "N00-N99 - Diseases of the genitourinary system",                                                                   "N00", "N99"),
    ("O",      "O00-O99 - Pregnancy, childbirth and the puerperium",                                                               "O00", "O99"),
    ("P",      "P00-P96 - Certain conditions originating in<br>the perinatal period",                                              "P00", "P96"),
    ("Q",      "Q00-Q99 - Congenital malformations, deformations<br>and chromosomal abnormalities",                                "Q00", "Q99"),
    ("R",      "R00-R99 - Symptoms, signs and abnormal clinical<br>and laboratory findings, not elsewhere classified",             "R00", "R99"),
    ("S-T",    "S00-T98 - Injury, poisoning and certain other<br>consequences of external causes",                                 "S00", "T98"),
    ("V-Y",    "V01-Y98 - External causes of morbidity and mortality",                                                             "V01", "Y98"),
]

# Danish codes not in WHO's ICD-10 systematic file
DANISH_ICD10_OVERRIDES = {
    "C99": "Clinical and paraclinical findings in cancer disease (DK)",
    "E47": "Other underweight (DK)",
    "O49": "Gestational length as primary indication for labour induction (DK)",
    "R67": "Findings on assessment of general functional ability (DK)",
    "R97": "Brought in without signs of life (DK)",
    "T89": "Healthcare-associated infections (DK)",
}

# T2D-related codes:
#  E10-E14: diabetes diagnosis codes
#  E15, E16, R73, R81: glucose / pancreatic metabolism
#  E66, E78: metabolic risk factors strongly tied to T2D
#  O24: diabetes in pregnancy
#  H36, N08, G59, G63, I79: typical diabetic complications
CIRCULAR_CODES = ["E10", "E11", "E12", "E13", "E14", "E15", "E16",
                  "E66", "E78", "O24", "R73", "R81",
                  "H36", "N08", "G59", "G63", "I79"]

SEX_SYMBOLS = {"alle": "square", "maend": "triangle-up", "kvinder": "circle"}
SEX_LABELS  = {"alle": "■  Both sexes", "maend": "▲  Men", "kvinder": "●  Women"}

PALETTE        = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
CHAPTER_COLORS = {c[0]: PALETTE[i] for i, c in enumerate(ICD10_CHAPTERS)}

HR_THRESHOLD_DEFAULT = 1.5

def sig_label(hr_thresh):
    return f"FDR q<0.05 + HR>={hr_thresh:g} or <={1/hr_thresh:.2g}"

def icd10_to_chapter(code):
    if not isinstance(code, str) or len(code) < 3:
        return "?"
    head = code[:3].upper()
    for short, _, lo, hi in ICD10_CHAPTERS:
        if lo <= head <= hi:
            return short
    return "?"


def chapter_label(short):
    for s, name, *_ in ICD10_CHAPTERS:
        if s == short:
            return name
    return short


def chapter_label_plain(short):
    return chapter_label(short).replace("<br>", " ")


def load_icd10_names(path):
    if not Path(path).exists():
        print(f"Warning: {path} not found, diagnosis names omitted.")
        return {}
    names = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(";")
            if len(parts) >= 9 and parts[0] == "3":
                names[parts[6]] = parts[8]
    return names


def significance_mask(sub, hr_thresh=HR_THRESHOLD_DEFAULT):
    fdr = sub["sig_fdr_global"].fillna(False).astype(bool)
    hr  = sub["HR"]
    return fdr & ((hr >= hr_thresh) | (hr <= 1/hr_thresh))

# Indlæs Cox-resultater og smid Cox-fits væk som ikke konvergerede / havde
# eksploderede CI'er. Tilføj log2(HR), -log10(p), kapitel-kolonner mm.
df = pd.read_parquet(DATA_PATH)
df = df[df["converged"].astype(bool)]
df = df[(df["HR_lo95"] > 0) & np.isfinite(df["HR_hi95"])].copy()

df["log2_HR"]       = np.log2(df["HR"].clip(lower=1e-12))
df["neg_log10_p"]   = -np.log10(df["p_val"].clip(lower=EPS_QVAL))
df["chapter"]       = df["icd10"].astype(str).map(icd10_to_chapter)
df["chapter_label"] = df["chapter"].map(chapter_label)

# Knyt diagnose-navn på hver 3-karakter-kode (fx E11 -> "Type 2 diabetes mellitus").
# Danske overrides anvendes oven på WHO-filen, så Danish-specific koder også
# får et menneskeligt navn.
ICD10_NAMES = load_icd10_names(ICD10_NAMES_PATH)
ICD10_NAMES.update(DANISH_ICD10_OVERRIDES)
df["icd10_name"] = df["icd10"].map(ICD10_NAMES).fillna("(unknown)")

# Re-beregn BH-FDR over HELE tabellen (R-pipelinen kørte det pr. chunk)
_, q_global, _, _        = multipletests(df["p_val"].values, method="fdr_bh")
df["q_val_global"]       = q_global
df["sig_fdr_global"]     = q_global < Q_SIGNIFICANT
df["neg_log10_q_global"] = -np.log10(np.clip(q_global, EPS_QVAL, None))

N_TESTS              = len(df)

df = df.sort_values(["chapter", "icd10"]).reset_index(drop=True)

NEG_LOG10_FDR = -np.log10(Q_SIGNIFICANT)

# Stabile lister af alle værdier — bruges af UI'en og (vigtigt!) af
# make_combined som itererer over ALL_CHAPTERS for at holde antallet
# af traces konstant på tværs af filterskift (så uirevision virker).
ALL_SEX      = sorted(df["sex"].dropna().unique().tolist())
_age_vals  = df["age_group"].dropna().unique().tolist()
ALL_AGE    = (["alle"] if "alle" in _age_vals else []) + \
             sorted([a for a in _age_vals if a != "alle"])
AGE_LABELS = {a: ("All ages" if a == "alle" else a) for a in ALL_AGE}
ALL_TW       = sorted(df["time_window"].dropna().unique().tolist())
ALL_ICD      = sorted(df["icd10"].dropna().unique().tolist())
ALL_CHAPTERS = sorted(df["chapter"].dropna().unique().tolist())

print(f"{N_TESTS:,} tests | Default highlighting rule: {sig_label(HR_THRESHOLD_DEFAULT)}")
df.head()

SEX_HOVER = {"alle": "Both", "maend": "Men", "kvinder": "Women"}
AGE_HOVER = {"alle": "All ages"}

HOVER = (
    "<b>%{customdata[0]}</b> - %{customdata[10]}<br>"
    "Chapter: %{text}<br>"
    "Age group: %{customdata[1]} | Sex: %{customdata[3]} | "
    "Time window: %{customdata[2]} y<br>"
    "N in stratum: %{customdata[11]}<br>"
    "HR = %{customdata[4]:.2f} (95% CI %{customdata[5]:.2f}-%{customdata[6]:.2f})<br>"
    "q = %{customdata[7]:.2e}<br>"
    "Events - cases: %{customdata[8]} | controls: %{customdata[9]}"
    "<extra></extra>"
)


def _customdata(sub):
    fmt = lambda s: s.apply(lambda x: "<20" if pd.isna(x) else str(int(x)))
    fmt_n = lambda s: s.apply(lambda x: f"{int(x):,}" if pd.notna(x) else "?")
    age_disp = sub["age_group"].astype(str).map(lambda a: AGE_HOVER.get(a, a))
    sex_disp = sub["sex"].astype(str).map(lambda s: SEX_HOVER.get(s, s))
    return np.stack([
        sub["icd10"].astype(str),         # 0
        age_disp,                         # 1
        sub["time_window"].astype(str),   # 2
        sex_disp,                         # 3
        sub["HR"], sub["HR_lo95"], sub["HR_hi95"],   # 4, 5, 6
        sub["q_val_global"],              # 7
        fmt(sub["n_events_case"]),        # 8
        fmt(sub["n_events_crtl"]),        # 9
        sub["icd10_name"].astype(str),    # 10
        fmt_n(sub["n_total"]),            # 11
    ], axis=-1)


def make_combined(df_plot, highlight_keys=None, ui_rev="main",
                  hr_thresh=HR_THRESHOLD_DEFAULT):
    label = sig_label(hr_thresh)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=(f"Volcano - highlighting: {label}",
                        f"Manhattan - highlighting: {label}"))

    if df_plot.empty:
        fig.update_layout(height=900, uirevision=ui_rev,
            annotations=[dict(text="No data after filtering",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="#888"))])
        return fig

    df_plot = df_plot.copy()
    df_plot["_uid"] = list(zip(df_plot["icd10"], df_plot["age_group"],
                                df_plot["time_window"], df_plot["sex"]))
    df_plot["_highlight"] = df_plot["_uid"].isin(highlight_keys or set())

    df_man = df_plot.sort_values(["chapter", "icd10"]).reset_index(drop=True)
    df_man["_x_man"] = np.arange(len(df_man))

    sig_v   = significance_mask(df_plot, hr_thresh)
    sig_m   = significance_mask(df_man,  hr_thresh)
    centers = []

    chapters_present = sorted(df_plot["chapter"].dropna().unique().tolist())
    for chap in chapters_present:
        color = CHAPTER_COLORS.get(chap, "#888")
        sub_v = df_plot[df_plot["chapter"] == chap]
        sub_m = df_man[df_man["chapter"] == chap]
        centers.append((chap, sub_m["_x_man"].mean()))

        for sub, xcol, row, sig in [(sub_v, "log2_HR", 1, sig_v),
                                     (sub_m, "_x_man",  2, sig_m)]:
            mask     = sig.loc[sub.index].values
            symbols  = sub["sex"].map(SEX_SYMBOLS).fillna("circle").values
            is_short = sub["_highlight"].values

            sizes   = np.where(mask, 11, 6)
            opacity = np.where(mask, 0.95, 0.25)
            line_w  = np.where(is_short, 2.5, 0.0)

            fig.add_trace(go.Scattergl(
                x=sub[xcol], y=sub["neg_log10_q_global"], mode="markers",
                showlegend=False,
                text=[chapter_label(chap)] * len(sub),
                marker=dict(size=sizes, color=color, symbol=symbols,
                            line=dict(width=line_w, color="black"),
                            opacity=opacity),
                customdata=_customdata(sub), hovertemplate=HOVER,
            ), row=row, col=1)

    for r in (1, 2):
        fig.add_hline(y=NEG_LOG10_FDR, line_dash="dash", line_color="red",
            annotation_text=f"q = {Q_SIGNIFICANT}",
            annotation_position="top right", row=r, col=1)
    fig.add_vline(x=np.log2(hr_thresh), line_dash="dash", line_color="orange",
        annotation_text=f"HR >= {hr_thresh:g}",
        annotation_position="top right", row=1, col=1)
    fig.add_vline(x=-np.log2(hr_thresh), line_dash="dash", line_color="orange",
        annotation_text=f"HR <= {1/hr_thresh:.2g}",
        annotation_position="top left", row=1, col=1)

    fig.update_xaxes(title_text="log2(HR)  (>0: higher risk in T2D group)",
                     row=1, col=1)
    fig.update_yaxes(title_text="-log10(q global)", row=1, col=1)
    fig.update_xaxes(title_text="ICD-10 code (grouped by chapter)",
                     tickmode="array",
                     tickvals=[c for _, c in centers],
                     ticktext=[c for c, _ in centers],
                     showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="-log10(q global)", row=2, col=1)

    fig.update_layout(
        height=900, uirevision=ui_rev,
        showlegend=False,
        hovermode="closest", clickmode="event",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


def make_forest(shortlist):
    if not shortlist:
        fig = go.Figure()
        fig.update_layout(
            height=200,
            annotations=[dict(
                text="Click points on the plot above to add them here "
                     "as a forest plot",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="#888"))],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
        )
        return fig

    items = sorted(shortlist, key=lambda r: r["HR"], reverse=True)
    sex_short = {"alle": "Both", "maend": "Men", "kvinder": "Women"}

    labels = [
        f"{r['icd10']} | {sex_short.get(r['sex'], r['sex'])} | "
        f"{r['age_group']} | {r['time_window']}y"
        for r in items
    ]
    hrs     = [r["HR"]      for r in items]
    los     = [r["HR_lo95"] for r in items]
    his     = [r["HR_hi95"] for r in items]
    chaps   = [icd10_to_chapter(r["icd10"]) for r in items]
    colors  = [CHAPTER_COLORS.get(c, "#888") for c in chaps]
    symbols = [SEX_SYMBOLS.get(r["sex"], "circle") for r in items]

    age_disp = lambda a: "All ages" if a == "alle" else a
    hover = [
        f"<b>{r['icd10']} - {r['icd10_name']}</b><br>"
        f"Chapter: {chapter_label_plain(icd10_to_chapter(r['icd10']))}<br>"
        f"Age group: {age_disp(r['age_group'])} | "
        f"Sex: {sex_short.get(r['sex'], r['sex'])} | "
        f"Time window: {r['time_window']} y<br>"
        f"N in stratum: {r.get('n_total', '?')}<br>"
        f"HR = {r['HR']:.2f} (95% CI {r['HR_lo95']:.2f}-{r['HR_hi95']:.2f})<br>"
        f"q = {r['q_val_global']:.2e}<br>"
        f"Events - cases: {r['n_events_case']} | controls: {r['n_events_crtl']}"
        for r in items
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hrs, y=labels, mode="markers",
        marker=dict(size=12, color=colors, symbol=symbols,
                    line=dict(width=1, color="black")),
        error_x=dict(
            type="data", symmetric=False,
            array=[h - hr for h, hr in zip(his, hrs)],
            arrayminus=[hr - l for l, hr in zip(los, hrs)],
            thickness=1.5, color="#444", width=4,
        ),
        text=hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=his, y=labels, mode="text",
        text=[f"  {hr:.2f} ({l:.2f}-{h:.2f})"
              for hr, l, h in zip(hrs, los, his)],
        textposition="middle right",
        textfont=dict(size=11, color="#444"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_vline(x=1, line_dash="dash", line_color="#888")

    x_min = max(0.01, min(los) * 0.7)
    x_max = max(his) * 2.5

    n = len(items)
    height = max(250, 60 + 28 * n)

    lo_exp = int(np.floor(np.log2(x_min)))
    hi_exp = int(np.ceil(np.log2(x_max)))
    tick_vals = [2.0**e for e in range(lo_exp, hi_exp + 1)]
    tick_text = [(f"{v:g}" if v >= 1 else f"{v:.3g}") for v in tick_vals]

    fig.update_layout(
        height=height,
        xaxis=dict(
            title="Hazard ratio (log2-spaced)", type="log",
            range=[np.log10(x_min), np.log10(x_max)],
            tickvals=tick_vals, ticktext=tick_text,
        ),
        yaxis=dict(autorange="reversed", title=""),
        margin=dict(l=200, r=40, t=20, b=60),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig

def filter_df(df_full, sex, age, tw, chapter, search_codes, exclude_circ,
              n_case, n_ctrl):
    sub = df_full
    if sex:          sub = sub[sub["sex"].isin(sex)]
    if age:          sub = sub[sub["age_group"] == age]
    if tw is not None: sub = sub[sub["time_window"] == tw]
    if chapter:      sub = sub[sub["chapter"].isin(chapter)]
    if search_codes: sub = sub[sub["icd10"].isin(search_codes)]
    if exclude_circ and "yes" in exclude_circ:
        sub = sub[~sub["icd10"].isin(CIRCULAR_CODES)]
    # NaN i n_events_* = censureret (oprindeligt 1-19). fillna(0) -> filtreres fra ved positiv tærskel.
    if n_case > 0: sub = sub[sub["n_events_case"].fillna(0) >= n_case]
    if n_ctrl > 0: sub = sub[sub["n_events_crtl"].fillna(0) >= n_ctrl]
    return sub

DEFAULT_SEX       = ["alle"] if "alle" in ALL_SEX else ALL_SEX[:1]
DEFAULT_AGE       = ("alle"  if "alle"  in ALL_AGE
                     else "18-29" if "18-29" in ALL_AGE
                     else ALL_AGE[0])
DEFAULT_TW        = 5.0 if 5.0 in ALL_TW else ALL_TW[0]
DEFAULT_NCASE     = 100
DEFAULT_NCTRL     = 200
DEFAULT_HR_THRESH = HR_THRESHOLD_DEFAULT

BOX = {"border": "1px solid #d0d0d0", "borderRadius": "6px",
       "padding": "12px", "marginBottom": "12px",
       "backgroundColor": "#ffffff"}

NUM_INPUT_STYLE = {"width": "100%", "padding": "6px",
                   "fontSize": "13px", "boxSizing": "border-box"}

app = Dash(__name__)
app.title = "T2D PheWAS - dashboard"

sidebar = html.Div([
    html.H3("Filters"),
    html.Button("Reset all", id="btn-reset-all", n_clicks=0,
        style={"width": "100%", "marginBottom": "12px",
               "backgroundColor": "#fff4e6", "border": "1px solid #d9a679",
               "padding": "6px", "fontWeight": "bold", "cursor": "pointer"}),

    html.Div([
        html.Label("Show only specific codes", style={"fontWeight": "bold"}),
        dcc.Dropdown(id="f-search",
            options=[{"label": f"{c} - {ICD10_NAMES.get(c, '(unknown)')}",
                      "value": c} for c in ALL_ICD],
            multi=True, placeholder="Type code or diagnosis (empty = all)",
            optionHeight=35, maxHeight=525,
            style={"fontSize": "12px"}),
        html.Small("When empty, all codes from the selected chapters are shown.",
                   style={"color": "#888", "fontSize": "11px"}),
    ], style=BOX),

    html.Div([
        html.Label("ICD-10 chapters", style={"fontWeight": "bold"}),
        html.Div([
            html.Button("Select all", id="btn-chap-all", n_clicks=0,
                style={"flex": 1, "fontSize": "11px",
                       "padding": "2px", "marginRight": "4px"}),
            html.Button("Deselect all", id="btn-chap-none", n_clicks=0,
                style={"flex": 1, "fontSize": "11px", "padding": "2px"}),
        ], style={"display": "flex", "marginBottom": "8px",
                  "marginTop": "4px"}),
        dcc.Checklist(id="f-chapter",
            options=[{"label": " " + chapter_label_plain(c), "value": c}
                     for c in ALL_CHAPTERS],
            value=ALL_CHAPTERS,
            labelStyle={"display": "block", "fontSize": "12px",
                        "padding": "1px 0"}),
    ], style=BOX),

    html.Div([
        html.Label("Sex stratum"),
        dcc.Checklist(id="f-sex",
            options=[{"label": SEX_LABELS[v], "value": v} for v in ALL_SEX],
            value=DEFAULT_SEX, labelStyle={"display": "block"}),
        html.Br(),
        html.Label("Age group"),
        dcc.RadioItems(id="f-age",
            options=[{"label": AGE_LABELS[v], "value": v} for v in ALL_AGE],
            value=DEFAULT_AGE, labelStyle={"display": "block"}),
        html.Br(),
        html.Label("Time window"),
        dcc.RadioItems(id="f-tw",
            options=[{"label": f"{int(v)} y", "value": v} for v in ALL_TW],
            value=DEFAULT_TW,
            labelStyle={"display": "inline-block", "marginRight": "10px"}),
    ], style=BOX),

    html.Div([
        html.Label("Exclude T2D-related codes", style={"fontWeight": "bold"}),
        dcc.Checklist(id="f-exclude-circ",
            options=[{"label": "  " + ", ".join(CIRCULAR_CODES), "value": "yes"}],
            value=[]),
    ], style=BOX),

    html.Div([
        html.Label("HR threshold (highlighting)", style={"fontWeight": "bold"}),
        dcc.Input(id="f-hr-thresh", type="number",
                  min=1.0, step=0.1, value=DEFAULT_HR_THRESH,
                  debounce=True, style=NUM_INPUT_STYLE),
        html.Div(style={"height": "6px"}),
        html.Small("Points are highlighted when q<0.05 AND HR >= threshold or "
                   "HR <= 1/threshold.",
                   style={"color": "#888", "fontSize": "11px"}),
    ], style=BOX),

    html.Div([
        html.Label("Min events - cases (T2D)", style={"fontWeight": "bold"}),
        dcc.Input(id="f-nev-case", type="number",
                  min=0, step=1, value=DEFAULT_NCASE,
                  debounce=True, style=NUM_INPUT_STYLE),
        html.Div(style={"height": "10px"}),
        html.Label("Min events - controls", style={"fontWeight": "bold"}),
        dcc.Input(id="f-nev-ctrl", type="number",
                  min=0, step=1, value=DEFAULT_NCTRL,
                  debounce=True, style=NUM_INPUT_STYLE),
        html.Div(style={"height": "6px"}),
        html.Small("Events 1-9 snap to 0, 10-19 snap to 20 (privacy censoring).",
                   style={"color": "#888", "fontSize": "11px"}),
    ], style=BOX),

    html.Div([
        html.Button("Clear shortlist", id="btn-clear", n_clicks=0,
                    style={"width": "100%", "marginBottom": "8px"}),
        html.Button("Download shortlist (CSV)", id="btn-download", n_clicks=0,
                    style={"width": "100%"}),
        dcc.Download(id="dl-shortlist"),
    ], style=BOX),
],
style={"width": "320px", "padding": "16px", "fontSize": "13px"})


main = html.Div([
    html.H2("T2D PheWAS - interactive dashboard"),

    html.Div([
        html.Div([
            html.Span("ICD-10 chapters:",
                style={"fontWeight": "bold", "fontSize": "14px",
                       "marginRight": "12px", "whiteSpace": "nowrap"}),
        ] + [
            html.Span([
                html.Span("●",
                    style={"color": color, "fontSize": "22px",
                           "marginRight": "5px",
                           "verticalAlign": "middle"}),
                html.Span(chap_short,
                    style={"verticalAlign": "middle"}),
            ],
            title=chapter_label_plain(chap_short),
            style={"display": "inline-block", "marginRight": "14px",
                   "fontSize": "13px", "whiteSpace": "nowrap",
                   "cursor": "help"})
            for chap_short, color in CHAPTER_COLORS.items()
        ], style={"display": "flex", "flexWrap": "wrap",
                  "alignItems": "center", "rowGap": "4px"}),

        html.Div([
            html.Span("Sex stratum:",
                style={"fontWeight": "bold", "fontSize": "14px",
                       "marginRight": "12px"}),
            html.Span(SEX_LABELS["alle"],
                style={"marginRight": "16px", "fontSize": "13px"}),
            html.Span(SEX_LABELS["maend"],
                style={"marginRight": "16px", "fontSize": "13px"}),
            html.Span(SEX_LABELS["kvinder"],
                style={"fontSize": "13px"}),
        ], style={"marginTop": "8px"}),
    ], style={
        "position": "sticky", "top": "0", "zIndex": 100,
        "backgroundColor": "#f5f5f5",
        "padding": "12px 16px",
        "borderBottom": "1px solid #ccc",
        "borderRadius": "4px",
        "marginBottom": "12px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.06)",
    }),

    html.Div([
        html.Div(id="stats-line", style={"color": "#666", "fontSize": "13px"}),
        html.P([
            "Tip: use ", html.B("ICD-10 chapters"),
            " in the sidebar to filter chapters. ",
            html.Br(),
            "Click a point to add to the shortlist. Use ",
            html.B("box-select"), " or ", html.B("lasso-select"),
            " for multiple.",
        ], style={"color": "#444", "margin": 0, "marginTop": "6px"}),
    ], style=BOX),

    html.Div([
        dcc.Graph(id="g-combined", config={"displaylogo": False}),
    ], style={**BOX, "height": "920px"}),

    html.Div([
        html.H3("Shortlist", style={"marginTop": 0}),
        html.Div(id="shortlist-summary",
                 style={"marginBottom": "8px", "color": "#444"}),
        dash_table.DataTable(id="shortlist-table",
            columns=[
                {"name": "ICD-10",         "id": "icd10"},
                {"name": "Diagnosis",      "id": "icd10_name"},
                {"name": "Chapter",        "id": "chapter_label"},
                {"name": "Sex",            "id": "sex"},
                {"name": "Age",            "id": "age_group"},
                {"name": "Years",          "id": "time_window"},
                {"name": "HR",             "id": "HR",
                 "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "95% CI low",     "id": "HR_lo95",
                 "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "95% CI high",    "id": "HR_hi95",
                 "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "q",              "id": "q_val_global",
                 "type": "numeric", "format": {"specifier": ".2e"}},
                {"name": "Events - cases", "id": "n_events_case"},
                {"name": "Events - ctrl",  "id": "n_events_crtl"},
            ],
            data=[], row_deletable=True, page_size=5,
            sort_action="native", filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontSize": "12px",
                        "fontFamily": "system-ui, sans-serif"},
            style_header={"backgroundColor": "#eef", "fontWeight": "bold"}),
    ], style=BOX),

    html.Div([
        html.H3("Forest plot of shortlist", style={"marginTop": 0}),
        html.P(
            "HR with 95% confidence intervals, log2-spaced axis. "
            "Color = ICD-10 chapter. Shape = sex. Sorted by HR descending.",
            style={"color": "#666", "fontSize": "12px",
                   "marginTop": 0, "marginBottom": "6px"},
        ),
        html.Div([
            dcc.Graph(id="g-forest", config={"displaylogo": False}),
        ], style={"maxHeight": "850px", "overflowY": "auto",
                  "border": "1px solid #eee", "borderRadius": "4px"}),
    ], style=BOX),

    dcc.Store(id="store-shortlist", data=[]),
],
style={"flex": 1, "padding": "16px", "minWidth": 0})


app.layout = html.Div([sidebar, main],
    style={"display": "flex", "fontFamily": "system-ui, sans-serif",
           "backgroundColor": "#f0f0f0", "minHeight": "100vh"})

def _summary(rows):
    n = len(rows or [])
    return "0 selections." if n == 0 else (
        f"{n} selections. Click 'x' to remove, or 'Clear shortlist'.")


def _snap_event_value(v):
    if v is None or v == 0 or v >= 20: return no_update
    return 0 if v < 10 else 20


@app.callback(
    Output("g-combined", "figure"),
    Output("stats-line", "children"),
    Input("f-sex",          "value"),
    Input("f-age",          "value"),
    Input("f-tw",           "value"),
    Input("f-chapter",      "value"),
    Input("f-exclude-circ", "value"),
    Input("f-nev-case",     "value"),
    Input("f-nev-ctrl",     "value"),
    Input("f-hr-thresh",    "value"),
    Input("f-search",       "value"),
    Input("store-shortlist", "data"),
    Input("btn-reset-all",  "n_clicks"),
)
def update_plots(sex, age, tw, chapter, excl_circ,
                 n_min_case, n_min_ctrl, hr_thresh,
                 search_codes, shortlist, n_reset):
    if hr_thresh is None or hr_thresh < 1.0:
        hr_thresh = DEFAULT_HR_THRESH
    n_min_case = n_min_case if n_min_case is not None else 0
    n_min_ctrl = n_min_ctrl if n_min_ctrl is not None else 0

    sub       = filter_df(df, sex, age, tw, chapter, search_codes or [],
                          excl_circ, n_min_case, n_min_ctrl)
    highlight = {tuple(item["_uid"]) for item in (shortlist or [])}
    ui_rev    = f"main-{n_reset or 0}"
    fig = make_combined(sub, highlight, ui_rev=ui_rev, hr_thresh=hr_thresh)

    n     = len(sub)
    n_sig = int(significance_mask(sub, hr_thresh).sum()) if n else 0
    stats = (f"Showing {n:,} points | Highlighted ({sig_label(hr_thresh)}): "
             f"{n_sig:,} of {N_TESTS:,} valid tests")
    return fig, stats


@app.callback(Output("f-nev-case", "value", allow_duplicate=True),
              Input("f-nev-case", "value"), prevent_initial_call=True)
def snap_case(v): return _snap_event_value(v)


@app.callback(Output("f-nev-ctrl", "value", allow_duplicate=True),
              Input("f-nev-ctrl", "value"), prevent_initial_call=True)
def snap_ctrl(v): return _snap_event_value(v)


@app.callback(
    Output("f-chapter", "value", allow_duplicate=True),
    Input("btn-chap-all",  "n_clicks"),
    Input("btn-chap-none", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_chapters(_n_all, _n_none):
    triggered = (callback_context.triggered[0]["prop_id"]
                 if callback_context.triggered else "")
    return ALL_CHAPTERS if triggered.startswith("btn-chap-all") else []


@app.callback(
    Output("store-shortlist",   "data"),
    Output("shortlist-table",   "data"),
    Output("shortlist-summary", "children"),
    Input("g-combined",      "clickData"),
    Input("g-combined",      "selectedData"),
    Input("shortlist-table", "data_previous"),
    Input("btn-clear",       "n_clicks"),
    State("shortlist-table", "data"),
    State("store-shortlist", "data"),
)
def update_shortlist(click_data, select_data, prev_rows, n_clear,
                     current_rows, store):
    store     = list(store or [])
    triggered = (callback_context.triggered[0]["prop_id"]
                 if callback_context.triggered else "")

    if triggered.startswith("btn-clear"):
        return [], [], "0 selections."

    if (triggered.startswith("shortlist-table")
            and prev_rows is not None and current_rows is not None):
        kept = {(r["icd10"], r["age_group"], str(r["time_window"]), r["sex"])
                for r in current_rows}
        store = [item for item in store if tuple(item["_uid"]) in kept]
        return store, current_rows, _summary(current_rows)

    points = []
    if click_data  and "points" in click_data:  points.extend(click_data["points"])
    if select_data and "points" in select_data: points.extend(select_data["points"])

    if points:
        existing = {tuple(item["_uid"]) for item in store}
        for p in points:
            cd = p.get("customdata")
            if not cd or len(cd) < 11: continue
            uid = (str(cd[0]), str(cd[1]), str(cd[2]), str(cd[3]))
            if uid in existing: continue
            existing.add(uid)
            store.append({
                "_uid": list(uid),
                "icd10": str(cd[0]), "age_group": str(cd[1]),
                "time_window": str(cd[2]), "sex": str(cd[3]),
                "HR": float(cd[4]), "HR_lo95": float(cd[5]), "HR_hi95": float(cd[6]),
                "q_val_global": float(cd[7]),
                "n_events_case": str(cd[8]), "n_events_crtl": str(cd[9]),
                "icd10_name": str(cd[10]),
                "n_total": str(cd[11]) if len(cd) >= 12 else "?",
                "chapter_label": chapter_label_plain(icd10_to_chapter(str(cd[0]))),
            })

    table = [{k: v for k, v in item.items() if k != "_uid"} for item in store]
    return store, table, _summary(table)


@app.callback(Output("dl-shortlist", "data"),
              Input("btn-download", "n_clicks"),
              State("shortlist-table", "data"),
              prevent_initial_call=True)
def download_shortlist(_n, rows):
    if not rows: return no_update
    return dcc.send_data_frame(pd.DataFrame(rows).to_csv,
                                "phewas_shortlist.csv", index=False)


@app.callback(
    Output("f-sex",             "value"),
    Output("f-age",             "value"),
    Output("f-tw",              "value"),
    Output("f-chapter",         "value"),
    Output("f-exclude-circ",    "value"),
    Output("f-nev-case",        "value"),
    Output("f-nev-ctrl",        "value"),
    Output("f-hr-thresh",       "value"),
    Output("f-search",          "value"),
    Output("store-shortlist",   "data",     allow_duplicate=True),
    Output("shortlist-table",   "data",     allow_duplicate=True),
    Output("shortlist-summary", "children", allow_duplicate=True),
    Input("btn-reset-all", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(_n):
    return (DEFAULT_SEX, DEFAULT_AGE, DEFAULT_TW, ALL_CHAPTERS,
            [], DEFAULT_NCASE, DEFAULT_NCTRL, DEFAULT_HR_THRESH,
            [], [], [], "0 selections.")


@app.callback(
    Output("g-forest", "figure"),
    Input("store-shortlist", "data"),
)
def update_forest(shortlist):
    return make_forest(shortlist)

server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

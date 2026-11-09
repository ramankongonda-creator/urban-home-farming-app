"""
charts.py - draws the results from analytics.py
  mpl(res)    -> matplotlib Figure   (notebook + report PNGs)
  plotly(res) -> plotly Figure       (Streamlit app, interactive hover)
One palette and one set of rules for both, so a chart looks the same wherever it appears.
"""
import numpy as np
import pandas as pd

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS, NEG = "#2a78d6", "#e34948"
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]
INTENT = ["#e34948", "#f2a5a4", "#c3c2b7", "#86b6ef", "#2a78d6"]
UNKNOWN = "#b5b3ad"
CHANNEL_COLORS = {"Housing-society demo": SERIES[0], "Referral": SERIES[1], "Corporate wellness": SERIES[2],
                  "Website (organic)": SERIES[3], "Google search": SERIES[4], "Instagram ads": SERIES[5], "Unknown": UNKNOWN}


def _is_pct(res):
    return "%" in res.get("unit", "")


def _fmt(res, v):
    u = res.get("unit", "")
    if pd.isna(v):
        return ""
    if "%" in u:
        return f"{v:.0f}%" if abs(v) >= 10 or v == 0 else f"{v:.1f}%"
    if "Spearman" in u:
        return f"{v:+.2f}"
    if "percentage-point" in u:
        return f"{v:+.0f} pts"
    if "₹" in u:
        return f"₹{v:,.0f}"
    if "(x)" in u:
        return f"{v:.1f}x"
    if "score" in u:
        return f"{v:.1f}"
    return f"{v:,.0f}"


def _series_colors(res, names):
    if res.get("series_colors"):
        return dict(zip(names, res["series_colors"]))
    if all(n in CHANNEL_COLORS for n in names):
        return {n: CHANNEL_COLORS[n] for n in names}
    if res.get("ordinal"):
        ramp = ORDINAL if len(names) == 4 else ORDINAL[-len(names):]
        return dict(zip(names, ramp))
    return dict(zip(names, SERIES))


# ============================================================================ matplotlib
def mpl(res, width=8.6):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11.5, "axes.titleweight": "bold",
                         "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white"})
    k, d = res["kind"], res["data"]
    n_rows = len(d) if k in ("hbar", "hbar_signed", "hbar_ci", "funnel", "stacked_hbar") else 6
    h = {"heatmap": 7.4, "scatter": 4.6, "box": 4.4}.get(k, max(3.0, .42 * n_rows + 1.4))
    fig, ax = plt.subplots(figsize=(width, h), dpi=150)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    vfmt = lambda v: _fmt(res, v)

    def hgrid():
        ax.xaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)

    def vgrid():
        ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)

    if k in ("hbar", "hbar_signed", "hbar_ci", "funnel"):
        lab = d["label"].astype(str).tolist()
        val = (d["rate"] * 100 if k == "hbar_ci" else d["count"] if k == "funnel" else d["value"]).to_numpy(float)
        if k == "funnel":
            lab, val = lab[::-1], val[::-1]
        col = [POS if v >= 0 else NEG for v in val] if k == "hbar_signed" else [POS] * len(val)
        y = np.arange(len(val))
        ax.barh(y, val, height=.62, color=col, edgecolor="white", linewidth=1)
        if k == "hbar_ci":
            ax.errorbar(val, y, xerr=[val - d.lo * 100, d.hi * 100 - val], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        ax.set_yticks(y, lab)
        span = np.nanmax(np.abs(val)) if len(val) else 1
        for yi, v in zip(y, val):
            txt = vfmt(v) if k != "funnel" else f"{v:,.0f}  ({v / val.max() * 100:.0f}% of leads)" if val.max() else ""
            if k == "hbar_ci":
                v_end = d.hi.iloc[yi] * 100
            else:
                v_end = v
            ax.text(v_end + (span * .015 if v >= 0 else -span * .015), yi, txt, va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=8.5)
        if k == "hbar_signed":
            ax.axvline(0, color=AXIS, lw=1)
            lo_, hi_ = np.nanmin(val), np.nanmax(val); pad = (hi_ - lo_) * .25
            ax.set_xlim(min(lo_ - pad, -pad * .5), hi_ + pad)
        else:
            ax.set_xlim(0, span * 1.32 if k != "funnel" else span * 1.45)
        for v, l in res.get("refs", []):
            ax.axvline(v, color=INK2, lw=1, ls="--"); ax.text(v, len(val) - .35, " " + l, color=INK2, fontsize=8, va="bottom")
        hgrid()
        if _is_pct(res):
            ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        elif "₹" in res.get("unit", ""):
            ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
        ax.set_xlabel(res.get("unit", ""))

    elif k in ("vbar", "vbar_ci"):
        lab = d["label"].astype(str).tolist()
        val = (d["rate"] * 100 if k == "vbar_ci" else d.iloc[:, 1] if "value" not in d else d["value"]).to_numpy(float)
        col = INTENT if res.get("colors") == "intent" else [POS] * len(val)
        x = np.arange(len(val))
        ax.bar(x, val, width=.6, color=col, edgecolor="white", linewidth=1)
        top = val.copy()
        if k == "vbar_ci":
            ax.errorbar(x, val, yerr=[val - d.lo * 100, d.hi * 100 - val], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
            top = d.hi.to_numpy() * 100
        for xi, v, t in zip(x, val, top):
            label = vfmt(v) + (f"\nn={int(d['n'].iloc[xi]):,}" if "n" in d else "")
            ax.text(xi, t + np.nanmax(top) * .02, label, ha="center", va="bottom", color=INK, fontsize=8.5)
        ax.set_xticks(x, lab); ax.set_ylim(0, np.nanmax(top) * 1.25); vgrid(); ax.spines["left"].set_visible(False)
        if _is_pct(res):
            ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_ylabel(res.get("unit", ""))

    elif k in ("grouped_vbar", "grouped_hbar"):
        names = res.get("series_order") or list(dict.fromkeys(d.series))
        cats = list(dict.fromkeys(d.label))
        cmap = _series_colors(res, names)
        bw = .8 / len(names)
        pos = np.arange(len(cats))
        for i, nm in enumerate(names):
            sub = d[d.series == nm].set_index("label").reindex(cats)["value"].to_numpy(float)
            off = pos - .4 + bw * (i + .5)
            if k == "grouped_vbar":
                ax.bar(off, sub, width=bw * .92, color=cmap[nm], label=nm, edgecolor="white", linewidth=1)
                for xi, v in zip(off, sub):
                    ax.text(xi, v + np.nanmax(d.value) * .015, vfmt(v), ha="center", va="bottom", fontsize=8, color=INK)
            else:
                ax.barh(-off, sub, height=bw * .92, color=cmap[nm], label=nm, edgecolor="white", linewidth=1)
                for yi, v in zip(-off, sub):
                    ax.text(v + np.nanmax(d.value) * .015, yi, vfmt(v), va="center", fontsize=8.5, color=INK)
        if k == "grouped_vbar":
            ax.set_xticks(pos, cats); ax.set_ylim(0, np.nanmax(d.value) * 1.2); vgrid(); ax.spines["left"].set_visible(False)
            ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            ax.set_yticks(-pos, cats); ax.set_xlim(0, np.nanmax(d.value) * 1.2); hgrid()
            ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12 if k == "grouped_vbar" else -.1), ncol=len(names))

    elif k in ("stacked_hbar", "stacked_vbar"):
        names = list(d.columns)
        cmap = _series_colors(res, names)
        cats = [str(c) for c in d.index]
        acc = np.zeros(len(d))
        pos = np.arange(len(d))
        for nm in names:
            v = d[nm].to_numpy(float)
            if k == "stacked_hbar":
                ax.barh(pos, v, left=acc, height=.6, color=cmap[nm], label=nm, edgecolor="white", linewidth=1.5)
                for yi, (a, w) in enumerate(zip(acc, v)):
                    if w >= 8:
                        dark_bg = cmap[nm] in ("#256abf", "#0d366b", "#3987e5", POS, NEG)
                        ax.text(a + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8, color="white" if dark_bg else INK)
            else:
                ax.bar(pos, v, bottom=acc, width=.62, color=cmap[nm], label=nm, edgecolor="white", linewidth=1)
            acc += v
        if k == "stacked_hbar":
            ax.set_yticks(pos, cats); ax.invert_yaxis(); ax.set_xlim(0, 100); hgrid()
            ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            ax.set_xticks(pos, cats); vgrid(); ax.spines["left"].set_visible(False)
            for xi, t in zip(pos, acc):
                ax.text(xi, t * 1.01, f"{t:,.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
            ax.set_ylim(0, acc.max() * 1.12); ax.set_ylabel(res.get("unit", ""))
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=min(len(names), 4))

    elif k == "hist":
        ax.hist(d["value"], bins=res["bins"], color=POS, edgecolor="white", linewidth=1.2)
        for i, (v, l) in enumerate(res.get("refs", [])):
            ax.axvline(v, color=INK2, lw=1.1, ls="--")
            ax.text(v, ax.get_ylim()[1] * (.97 - .08 * i), (l + " ") if i == 0 else (" " + l), color=INK2, fontsize=8.5, va="top",
                    ha="right" if i == 0 else "left")
        vgrid(); ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"₹{v / 1000:.0f}k"))
        ax.set_xlabel(res.get("xlab", "")); ax.set_ylabel(res.get("unit", ""))

    elif k == "heatmap":
        import matplotlib.colors as mc
        cmap = mc.LinearSegmentedColormap.from_list("div", [NEG, "#f0efec", POS])
        m = d.to_numpy(float)
        ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(d)), d.columns, rotation=55, ha="right", fontsize=8); ax.set_yticks(range(len(d)), d.index, fontsize=8)
        for i in range(len(d)):
            for j in range(len(d)):
                if i != j:
                    ax.text(j, i, f"{m[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=6.3,
                            color="white" if abs(m[i, j]) > .45 else INK)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mc.Normalize(-1, 1))
        cb = fig.colorbar(sm, ax=ax, fraction=.035, pad=.02); cb.outline.set_visible(False); cb.set_label("Spearman r", color=INK2)

    elif k == "box":
        order = res.get("order") or list(dict.fromkeys(d.group))
        data = [d.loc[d.group == g, "value"].dropna().to_numpy() for g in order]
        bp = ax.boxplot(data, widths=.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=INK2), capprops=dict(color=INK2), boxprops=dict(edgecolor=INK2))
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor("#cde2fb" if order[i] != "Not disclosed" else "#e1e0d9")
        for i, arr in enumerate(data):
            if len(arr):
                ax.text(i + 1, np.median(arr), f" ₹{np.median(arr):,.0f}", va="bottom", ha="left", fontsize=8, color=INK)
        for v, l in res.get("refs", []):
            ax.axhline(v, color=NEG, lw=1, ls="--"); ax.text(.55, v, l, color=NEG, fontsize=8, va="bottom", ha="left")
        ax.set_xticks(range(1, len(order) + 1), [o.replace("Rs ", "₹").replace("-", "–") for o in order], fontsize=8.3)
        vgrid(); ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"₹{v:,.0f}")); ax.set_ylabel(res.get("unit", ""))

    elif k == "scatter":
        ax.scatter(d[res["x"]], d[res["y"]], s=10, color=POS, alpha=.28, linewidths=0)
        sl, ic = res["fit"]
        xs = np.linspace(d[res["x"]].min(), d[res["x"]].max(), 50)
        ax.plot(xs, ic + sl * xs, color=NEG, lw=2, label="Linear trend")
        ax.legend(frameon=False, loc="upper left")
        vgrid(); ax.xaxis.grid(True, color=GRID, lw=.6); ax.spines["left"].set_visible(False)
        f_ = mt.FuncFormatter(lambda v, _: f"₹{v:,.0f}")
        ax.xaxis.set_major_formatter(f_); ax.yaxis.set_major_formatter(f_)
        ax.set_xlabel(res.get("xlab", "")); ax.set_ylabel(res.get("ylab", ""))

    elif k == "line":
        names = list(dict.fromkeys(d.series))
        cmap = _series_colors(res, names)
        ends = [d[d.series == nm].sort_values(res["x"]).value.iloc[-1] for nm in names]
        span = (d.value.max() - d.value.min()) or 1
        label_ends = len(ends) < 2 or abs(ends[0] - ends[1]) > .06 * span
        for nm in names:
            sub = d[d.series == nm].sort_values(res["x"])
            ax.plot(sub[res["x"]], sub["value"], color=cmap[nm], lw=2, marker="o", ms=5, label=nm)
            last = sub.iloc[-1]
            if label_ends: ax.annotate(_fmt(res, last.value) if "index" not in res.get("unit", "") else f"{last.value:.0f}",
                        (last[res["x"]], last.value), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
        for v, l in res.get("refs_y", []):
            ax.axhline(v, color=AXIS, lw=1)
        vgrid(); ax.spines["left"].set_visible(False)
        if _is_pct(res):
            ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%")); ax.set_ylim(0, None)
        if res["x"] in ("price_shown", "x"):
            xs_ = sorted(d[res["x"]].unique()); ax.set_xticks(xs_, [f"₹{v:,.0f}" for v in xs_])
        ax.set_xlabel(res.get("xlab", "")); ax.set_ylabel(res.get("unit", ""))
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=len(names))

    ax.set_title(res["title"], loc="left", pad=12)
    fig.tight_layout()
    return fig


# ============================================================================ plotly
def plotly(res, height=None):
    import plotly.graph_objects as go
    k, d = res["kind"], res["data"]
    fig = go.Figure()
    unit = res.get("unit", "")
    pctf = _is_pct(res)
    hov = lambda v: _fmt(res, v)

    if k in ("hbar", "hbar_signed", "hbar_ci", "funnel"):
        lab = d["label"].astype(str).tolist()
        val = (d["rate"] * 100 if k == "hbar_ci" else d["count"] if k == "funnel" else d["value"]).astype(float).tolist()
        col = [POS if v >= 0 else NEG for v in val] if k == "hbar_signed" else POS
        text = [f"{v:,.0f} ({v / max(val) * 100:.0f}%)" for v in val] if k == "funnel" else [hov(v) for v in val]
        err = dict(type="data", symmetric=False, array=(d.hi * 100 - d.rate * 100).tolist(), arrayminus=(d.rate * 100 - d.lo * 100).tolist(),
                   color=INK2, thickness=1) if k == "hbar_ci" else None
        custom = d["n"].tolist() if "n" in d else None
        fig.add_bar(y=lab, x=val, orientation="h", marker_color=col, text=text, textposition="outside", cliponaxis=False, error_x=err,
                    customdata=custom, hovertemplate="%{y}: %{text}" + (" (n=%{customdata:,})" if custom else "") + "<extra></extra>")
        fig.update_yaxes(autorange="reversed" if k == "funnel" else True)
        for v, l in res.get("refs", []):
            fig.add_vline(x=v, line_dash="dash", line_color=INK2, annotation_text=l, annotation_position="top", annotation_font_color=INK2)
        if k == "hbar_signed":
            fig.add_vline(x=0, line_color=AXIS)
        fig.update_xaxes(ticksuffix="%" if pctf else "", tickprefix="₹" if "₹" in unit else "", title=unit)
        height = height or max(300, 38 * len(lab) + 120)

    elif k in ("vbar", "vbar_ci"):
        lab = d["label"].astype(str).tolist()
        val = (d["rate"] * 100 if k == "vbar_ci" else d["value"] if "value" in d else d.iloc[:, 1]).astype(float).tolist()
        col = INTENT if res.get("colors") == "intent" else POS
        err = dict(type="data", symmetric=False, array=(d.hi * 100 - d.rate * 100).tolist(), arrayminus=(d.rate * 100 - d.lo * 100).tolist(),
                   color=INK2, thickness=1) if k == "vbar_ci" else None
        custom = d["n"].tolist() if "n" in d else None
        fig.add_bar(x=lab, y=val, marker_color=col, text=[hov(v) for v in val], textposition="outside", cliponaxis=False, error_y=err,
                    customdata=custom, hovertemplate="%{x}: %{text}" + (" (n=%{customdata:,})" if custom else "") + "<extra></extra>")
        fig.update_yaxes(ticksuffix="%" if pctf else "", title=unit)

    elif k in ("grouped_vbar", "grouped_hbar"):
        names = res.get("series_order") or list(dict.fromkeys(d.series))
        cmap = _series_colors(res, names)
        for nm in names:
            sub = d[d.series == nm]
            if k == "grouped_vbar":
                fig.add_bar(x=sub.label.astype(str), y=sub.value, name=nm, marker_color=cmap[nm], text=[hov(v) for v in sub.value],
                            textposition="outside", cliponaxis=False, hovertemplate=f"{nm}<br>%{{x}}: %{{text}}<extra></extra>")
            else:
                fig.add_bar(y=sub.label.astype(str), x=sub.value, name=nm, orientation="h", marker_color=cmap[nm], text=[hov(v) for v in sub.value],
                            textposition="outside", cliponaxis=False, hovertemplate=f"{nm}<br>%{{y}}: %{{text}}<extra></extra>")
        fig.update_layout(barmode="group", bargap=.25, bargroupgap=.06)
        if k == "grouped_hbar":
            fig.update_yaxes(autorange="reversed"); fig.update_xaxes(ticksuffix="%")
        else:
            fig.update_yaxes(ticksuffix="%")

    elif k in ("stacked_hbar", "stacked_vbar"):
        names = list(d.columns)
        cmap = _series_colors(res, names)
        for nm in names:
            if k == "stacked_hbar":
                fig.add_bar(y=[str(i) for i in d.index], x=d[nm], name=str(nm), orientation="h", marker_color=cmap[nm],
                            text=[f"{v:.0f}%" if v >= 8 else "" for v in d[nm]], textposition="inside", insidetextanchor="middle",
                            hovertemplate=f"{nm}<br>%{{y}}: %{{x:.0f}}%<extra></extra>")
            else:
                fig.add_bar(x=[str(i) for i in d.index], y=d[nm], name=str(nm), marker_color=cmap[nm],
                            hovertemplate=f"{nm}<br>%{{x}}: %{{y:,}} leads<extra></extra>")
        fig.update_layout(barmode="stack", bargap=.3)
        if k == "stacked_hbar":
            fig.update_yaxes(autorange="reversed"); fig.update_xaxes(ticksuffix="%", range=[0, 100])
        else:
            fig.update_yaxes(title=unit)
        height = height or (max(260, 70 * len(d) + 140) if k == "stacked_hbar" else None)

    elif k == "hist":
        fig.add_histogram(x=d["value"], xbins=dict(start=res["bins"][0], end=res["bins"][-1], size=res["bins"][1] - res["bins"][0]),
                          marker_color=POS, marker_line_color="white", marker_line_width=1,
                          hovertemplate="₹%{x}: %{y:,} respondents<extra></extra>")
        for v, l in res.get("refs", []):
            fig.add_vline(x=v, line_dash="dash", line_color=INK2, annotation_text=l, annotation_position="top right", annotation_font_color=INK2)
        fig.update_xaxes(title=res.get("xlab", ""), tickprefix="₹"); fig.update_yaxes(title=unit)

    elif k == "heatmap":
        fig.add_heatmap(z=d.values, x=list(d.columns), y=list(d.index), zmin=-1, zmax=1,
                        colorscale=[[0, NEG], [.5, "#f0efec"], [1, POS]], text=d.round(2).values, texttemplate="%{text}",
                        textfont=dict(size=9), hovertemplate="%{y} × %{x}: r = %{z:.2f}<extra></extra>", colorbar=dict(title="r"))
        fig.update_yaxes(autorange="reversed", automargin=True); fig.update_xaxes(tickangle=-45, automargin=True)
        height = height or 760

    elif k == "box":
        order = res.get("order") or list(dict.fromkeys(d.group))
        for g in order:
            v = d.loc[d.group == g, "value"]
            fig.add_box(y=v, name=g.replace("Rs ", "₹"), marker_color=POS if g != "Not disclosed" else MUTED, boxpoints=False,
                        fillcolor="#cde2fb" if g != "Not disclosed" else "#e1e0d9", line_color=INK2)
        for v, l in res.get("refs", []):
            fig.add_hline(y=v, line_dash="dash", line_color=NEG, annotation_text=l, annotation_font_color=NEG)
        fig.update_layout(showlegend=False); fig.update_yaxes(tickprefix="₹", title=unit)

    elif k == "scatter":
        fig.add_scatter(x=d[res["x"]], y=d[res["y"]], mode="markers", marker=dict(color=POS, size=6, opacity=.3), name="Respondent",
                        hovertemplate="Spend ₹%{x:,.0f}<br>Max price ₹%{y:,.0f}<extra></extra>")
        sl, ic = res["fit"]; xs = np.linspace(d[res["x"]].min(), d[res["x"]].max(), 50)
        fig.add_scatter(x=xs, y=ic + sl * xs, mode="lines", line=dict(color=NEG, width=2.5), name="Linear trend", hoverinfo="skip")
        fig.update_xaxes(title=res.get("xlab", ""), tickprefix="₹"); fig.update_yaxes(title=res.get("ylab", ""), tickprefix="₹")

    elif k == "line":
        names = list(dict.fromkeys(d.series))
        cmap = _series_colors(res, names)
        for nm in names:
            sub = d[d.series == nm].sort_values(res["x"])
            fig.add_scatter(x=sub[res["x"]], y=sub.value, mode="lines+markers", name=nm, line=dict(color=cmap[nm], width=2.5),
                            marker=dict(size=8), hovertemplate=f"{nm}<br>%{{x}}: %{{y:.1f}}<extra></extra>")
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(ticksuffix="%" if pctf else "", title=unit, rangemode="tozero" if pctf else "normal")
        if res["x"] in ("price_shown", "x"):
            xs_ = sorted(d[res["x"]].unique())
            fig.update_xaxes(tickvals=xs_, ticktext=[f"₹{v:,.0f}" for v in xs_])
        fig.update_xaxes(title=res.get("xlab", ""))

    fig.update_layout(template="plotly_white", height=height or 400, margin=dict(l=10, r=30, t=30, b=10),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig

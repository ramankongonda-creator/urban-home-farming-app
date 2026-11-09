"""
Streamlit dashboard - Urban home-farming start-up: descriptive & diagnostic analytics
Run:  streamlit run app.py
Data: synthetic (see 01_generate_data.py). Calculations: analytics.py. Charts: charts.py.
"""
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

import analytics as A
import charts as C
import models as M

st.set_page_config(page_title="Home-farming start-up · EDA", page_icon="🌱", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1200px}
.explain{border-left:3px solid #2a78d6;padding:.2rem 0 .2rem 1rem;margin:.2rem 0 1.6rem 0}
.explain p{margin:.25rem 0}
.q{color:#52514e;font-style:italic;margin-top:-.6rem}
[data-testid="stMetricValue"]{font-size:1.6rem}
</style>""", unsafe_allow_html=True)


@st.cache_data
def run_models():
    s, p = A.load()
    return dict(cls=M.classification(s), lead=M.lead_scoring(p), clu=M.clustering(s), arm=M.association(s), reg=M.regression(s),
                pres=M.prescriptive(s, p))


def fig_layout(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=30, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08, x=0), font=dict(size=13))
    return fig


def note(shows, why, so_what):
    st.markdown(f"<div class='explain'><p><b>What it shows.</b> {shows}</p><p><b>Why.</b> {why}</p><p><b>So what.</b> {so_what}</p></div>",
                unsafe_allow_html=True)


@st.cache_data
def load():
    s, p = A.load()
    raw_s = pd.read_csv("data/raw/survey_raw.csv")
    raw_p = pd.read_csv("data/raw/pipeline_raw.csv")
    log = pd.read_csv("data/clean/cleaning_log.csv")
    return s, p, raw_s, raw_p, log


s_all, p_all, raw_s, raw_p, clog = load()

PAGES = ["Overview", "1 · Data preparation", "2 · Survey: descriptive", "3 · Survey: diagnostic",
         "4 · Pipeline: descriptive", "5 · Pipeline: diagnostic", "6 · End-to-end validation",
         "7 · Predictive: 4 algorithms", "8 · Prescriptive: break-even & plan"]
page = st.sidebar.radio("Section", PAGES)
st.sidebar.caption("All data is synthetic, generated to test the business idea. Numbers in the explanations are recalculated "
                   "from whatever subset the filters select.")

# ---------------------------------------------------------------- filters
s, p = s_all, p_all
if page.startswith(("2", "3")):
    st.sidebar.subheader("Survey filters")
    tiers = st.sidebar.multiselect("City tier", [1, 2, 3], [1, 2, 3], format_func=lambda t: {1: "Metro", 2: "Tier 2", 3: "Tier 3"}[t])
    inc = st.sidebar.multiselect("Income band", A.INCOME_ORDER, A.INCOME_ORDER)
    spc = st.sidebar.multiselect("Growing space", A.SPACE_ORDER, A.SPACE_ORDER)
    s = s_all[s_all.city_tier.isin(tiers) & s_all.income_label.isin(inc) & s_all.space_label.isin(spc)]
    st.sidebar.metric("Respondents selected", f"{len(s):,}", f"{len(s) / len(s_all):.0%} of total", delta_color="off")
if page.startswith(("4", "5")):
    st.sidebar.subheader("Pipeline filters")
    cities = st.sidebar.multiselect("City", ["Bengaluru", "Pune"], ["Bengaluru", "Pune"])
    chans = st.sidebar.multiselect("Channel", A.CHANNELS + ["Unknown"], A.CHANNELS + ["Unknown"])
    months = sorted(p_all.lead_month.unique())
    m0, m1 = st.sidebar.select_slider("Lead month", months, (months[0], months[-1]))
    p = p_all[p_all.city.isin(cities) & p_all.channel.isin(chans) & p_all.lead_month.between(m0, m1)]
    st.sidebar.metric("Leads selected", f"{len(p):,}", f"{len(p) / len(p_all):.0%} of total", delta_color="off")


def show(fn, *args, table=None):
    try:
        res = fn(*args)
    except Exception:
        st.warning(f"**{fn.__name__}**: not enough data in this selection to draw this chart. Widen the filters.")
        return None
    st.subheader(f"{res['id'].upper()}. {res['title']}")
    st.markdown(f"<p class='q'>{res['question']}</p>", unsafe_allow_html=True)
    st.plotly_chart(C.plotly(res), use_container_width=True, config={"displaylogo": False})
    t = res["text"]
    st.markdown(f"<div class='explain'><p><b>What it shows.</b> {t['shows']}</p><p><b>Why.</b> {t['why']}</p>"
                f"<p><b>So what.</b> {t['so_what']}</p></div>", unsafe_allow_html=True)
    if table is not None:
        with st.expander("Table view"):
            st.dataframe(table(res), use_container_width=True, hide_index=True)
    return res


def data_table(res):
    d = res["data"]
    return d.reset_index() if not isinstance(d.index, pd.RangeIndex) else d


# ================================================================= pages
if page == "Overview":
    st.title("🌱 Urban home-farming start-up — will it sell?")
    st.markdown("A vertical balcony growing kit with a monthly seed-and-nutrient refill subscription, for Indian city homes. "
                "This dashboard validates the idea **end to end**: a consumer survey tests demand *before* launch, and a 6-month "
                "pilot sales pipeline tests whether that demand turns into paying, staying customers.")
    c = st.columns(4)
    c[0].metric("Survey responses (clean)", f"{len(s_all):,}")
    c[1].metric("Would buy at shown price", A.pct(s_all.would_buy.mean()))
    c[2].metric("Median max price for kit", A.inr(s_all.wtp_system.median()))
    c[3].metric("Would subscribe to refills", A.pct(s_all.would_subscribe.mean()))
    c = st.columns(4)
    c[0].metric("Pilot leads", f"{len(p_all):,}")
    c[1].metric("Lead → purchase", A.pct(p_all.purchased.mean(), 1))
    c[2].metric("Blended CAC", A.inr(p_all.acquisition_cost_inr.sum() / p_all.purchased.sum()))
    c[3].metric("Subscriber monthly churn", A.pct(A.churn_rate(p_all[p_all.refill_subscribed == 1]), 1))
    st.markdown("### How to read this dashboard")
    st.markdown("""
| Section | Analytics | Answers |
|---|---|---|
| 1 · Data preparation | Cleaning & transformation | What was wrong with the raw data and how it was fixed |
| 2 · Survey: descriptive | What is happening | Who the customers are, what they want, what they will pay |
| 3 · Survey: diagnostic | Why it is happening | Correlations and drivers of intent, price and subscription |
| 4 · Pipeline: descriptive | What happened in the pilot | Funnel, lead volume, conversion and CAC by channel |
| 5 · Pipeline: diagnostic | Why leads buy and stay | Price test, response speed, reps, retention, NPS |
| 6 · End-to-end validation | Survey vs reality | Correction factors and the finalised sales pipeline |
| 7 · Predictive: 4 algorithms | Who will buy, segments, feature bundles, willingness to pay | Classification, clustering, association rules, regression |
| 8 · Prescriptive | What to do and when it pays | Break-even, payback and the action plan |

Every chart has three notes underneath: **What it shows**, **Why**, and **So what** (the decision it supports). Use the sidebar filters
on sections 2–5; the notes recalculate for the selection.""")
    st.markdown("### Headline verdict")
    st.success("**Viable as a focused, demo-led business, not as a mass-market online product.** Launch at ₹4,999 in gated "
               "housing societies, build the pipeline on society demos and referrals, call every lead within 2 hours, and make the "
               "first harvest happen within 35 days. Paid social and search do not pay back.")

elif page.startswith("7"):
    R = run_models()
    st.title("Predictive analytics: the four algorithms")
    st.dataframe(M.COLUMN_MAP, use_container_width=True, hide_index=True)
    t1, t2, t3, t4 = st.tabs(["Classification", "Clustering", "Association rules", "Regression"])
    with t1:
        c = R["cls"]; res = c["results"]
        st.subheader("Who will buy? — three classifiers compared")
        st.dataframe(res.drop(columns="cm").round(3), use_container_width=True, hide_index=True)
        cols = st.columns(3)
        for col, r in zip(cols, res.itertuples()):
            f = go.Figure(go.Heatmap(z=r.cm, x=["Pred: no", "Pred: buy"], y=["Actual: no", "Actual: buy"], text=r.cm, texttemplate="%{text}",
                                     colorscale=[[0, "#f0efec"], [1, C.POS]], showscale=False))
            f.update_yaxes(autorange="reversed"); col.markdown(f"**{r.model}** · acc {r.accuracy:.0%} · AUC {r.auc:.2f}")
            col.plotly_chart(fig_layout(f, 260), use_container_width=True)
        b = c["best"]
        note(f"Confusion matrices on a {c['n_test']}-person test set. Best: {b.model}, AUC {b.auc:.2f}, recall {b.recall:.0%}, precision {b.precision:.0%}.",
             f"Only {c['base_rate']:.0%} are buyers, so accuracy alone misleads; AUC and recall matter. The shallow decision tree is weakest because intent "
             "rises smoothly with each factor.", f"Targeting the model's list is {b.precision / c['base_rate']:.1f}x better than random.")
        o = c["odds"]
        f = go.Figure(go.Bar(y=o.factor, x=o.odds_ratio_per_sd, orientation="h", marker_color=[C.POS if v >= 1 else C.NEG for v in o.odds_ratio_per_sd],
                             text=[f"{v:.2f}x" for v in o.odds_ratio_per_sd], textposition="outside"))
        f.add_vline(x=1, line_color=C.AXIS); st.markdown("**Odds ratios (logistic regression, per 1 SD)**")
        st.plotly_chart(fig_layout(f, 480), use_container_width=True)
        l = R["lead"]; g = l["gains"]
        st.subheader("Lead scoring on the pilot CRM")
        st.dataframe(l["results"].drop(columns="cm").round(3), use_container_width=True, hide_index=True)
        f = go.Figure(go.Bar(x=[f"D{d}" for d in g.decile], y=g.conversion * 100, marker_color=[C.POS] * 3 + ["#86b6ef"] * 7,
                             text=[f"{v:.0%}" for v in g.conversion], textposition="outside"))
        f.add_hline(y=l["base_rate"] * 100, line_dash="dash", annotation_text=f"average {l['base_rate']:.0%}"); f.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_layout(f), use_container_width=True)
        note(f"Contacted leads ranked into ten groups by predicted score. The top decile converts at {g.conversion.iloc[0]:.0%}.",
             "Channel, response time, price, space, income and demo type together separate buyers from browsers.",
             f"The top 30% of leads hold {g.cum_share_of_buyers.iloc[2]:.0%} of buyers: give them a 2-hour call and an in-person demo; nurture the rest automatically.")
    with t2:
        k = R["clu"]; pr = k["profile"]; sc = k["scan"]
        a, b = st.columns(2)
        a.plotly_chart(fig_layout(go.Figure(go.Scatter(x=sc.k, y=sc.inertia, mode="lines+markers", line_color=C.POS)), 280).update_layout(title="Elbow"), use_container_width=True)
        b.plotly_chart(fig_layout(go.Figure(go.Scatter(x=sc.k, y=sc.silhouette, mode="lines+markers", line_color=C.SERIES[1])), 280).update_layout(title="Silhouette"), use_container_width=True)
        f = go.Figure()
        f.add_bar(y=pr.name, x=pr.would_buy * 100, name="Would buy kit", orientation="h", marker_color=C.POS, text=[f"{v:.0%}" for v in pr.would_buy], textposition="outside")
        f.add_bar(y=pr.name, x=pr.would_subscribe * 100, name="Would subscribe", orientation="h", marker_color=C.SERIES[1], text=[f"{v:.0%}" for v in pr.would_subscribe], textposition="outside")
        f.update_yaxes(autorange="reversed"); f.update_xaxes(ticksuffix="%"); f.update_layout(barmode="group")
        st.plotly_chart(fig_layout(f, 420), use_container_width=True)
        st.dataframe(pr.round(2), use_container_width=True, hide_index=True)
        top, low = pr.iloc[0], pr.iloc[-1]
        note(f"k-means with k = {k['k']} (highest silhouette, {k['silhouette']:.3f}) on attitudes, space, sunlight, income, spend, household and experience.",
             "A low silhouette is normal for attitude data: people blend between groups, so segments are tendencies, not hard boxes.",
             f"Launch on {top['name']} ({top.would_buy:.0%} buy, {top.share:.0%} of market); market the refill to busy tech-savvy professionals; "
             f"exclude {low['name']} ({low.would_buy:.0%}) from paid marketing.")
    with t3:
        ar = R["arm"]; la = ar["launch"]
        st.dataframe(ar["feature_rules"].round(3), use_container_width=True, hide_index=True)
        with st.expander("All rules (features + crops)"):
            st.dataframe(ar["rules"].round(3), use_container_width=True, hide_index=True)
        cmap = {"Launch": C.POS, "Phase 2 (month 6)": "#86b6ef", "Phase 3 / add-on": C.AXIS}
        f = go.Figure(go.Bar(y=la.feature, x=la.wanted_by_likely_buyers * 100, orientation="h", marker_color=[cmap[x] for x in la.phase],
                             text=[f"{v:.0%} · {ph}" for v, ph in zip(la.wanted_by_likely_buyers, la.phase)], textposition="outside"))
        f.update_yaxes(autorange="reversed"); f.update_xaxes(ticksuffix="%", range=[0, 80])
        st.plotly_chart(fig_layout(f, 420), use_container_width=True)
        note(f"Apriori found {ar['n_itemsets']} frequent itemsets and {len(ar['rules'])} rules (confidence ≥ 45%, lift > 1.05).",
             f"Lifts are modest (max {ar['rules'].lift.max():.2f}): features are wanted fairly independently, so this is the weakest technique here.",
             "Launch vertical + self-watering + organic seeds + hydroponic pods together; add the refill, soil option and sensor at month 6.")
    with t4:
        rg = R["reg"]; cf = rg["coef"].copy()
        vs = cf.factor == "Monthly veg spend"; cf.loc[vs, "rupees_per_unit"] *= 1000; cf.loc[vs, "factor"] = "Monthly veg spend (per ₹1,000)"
        cf = cf.sort_values("rupees_per_unit")
        st.dataframe(rg["perf"].round(3), use_container_width=True, hide_index=True)
        f = go.Figure(go.Bar(y=cf.factor, x=cf.rupees_per_unit, orientation="h", marker_color=[C.POS if v >= 0 else C.NEG for v in cf.rupees_per_unit],
                             text=[("−₹" if v < 0 else "₹") + f"{abs(v):,.0f}" for v in cf.rupees_per_unit], textposition="outside"))
        st.plotly_chart(fig_layout(f, 460), use_container_width=True)
        perf = rg["perf"]
        note(f"Linear regression explains {perf.r2_test.iloc[0]:.0%} of willingness to pay on test data (average error ₹{perf.mae.iloc[0]:,.0f}).",
             f"The random forest overfits (train R² {perf.r2_train.iloc[1]:.2f}, test {perf.r2_test.iloc[1]:.2f}), so the simpler model wins.",
             "Price tiers: ₹4,999 standard, ₹6,999 premium bundle for large-balcony/terrace homes earning ₹2 lakh+, ₹2,999 starter later.")

elif page.startswith("8"):
    R = run_models(); pres = R["pres"]
    st.title("Prescriptive analytics: break-even and the action plan")
    c = st.columns(2)
    for col, (name, v) in zip(c, pres.items()):
        col.subheader(name)
        col.metric("Monthly profit from", f"month {v['breakeven']}" if v["breakeven"] else "not within 48 months")
        col.metric("Cash payback", f"month {v['payback']}" if v["payback"] else "not within 48 months")
        col.metric("Peak funding needed" if v["payback"] else "Cash burned by month 48 (still falling)", A.inr(-v["min_cash"]))
        col.caption(f"CAC {A.inr(v['inputs']['cac'])} · churn {v['inputs']['churn']:.1%} · refill attach {v['inputs']['attach']:.0%}")
    f = go.Figure()
    for (name, v), colr in zip(pres.items(), [C.NEG, C.POS]):
        f.add_scatter(x=v["table"].month, y=v["table"].cumulative / 1e5, name=name, line=dict(color=colr, width=3))
    f.add_hline(y=0, line_color=C.AXIS); f.update_yaxes(title="cumulative cash (₹ lakh)"); f.update_xaxes(title="month")
    st.plotly_chart(fig_layout(f, 420), use_container_width=True)
    note("48-month cash model. Assumptions: ₹25 lakh upfront; fixed costs ₹3.5 lakh/month (year 1) and ₹6 lakh/month after; 80 buyers in month 1 "
         "growing 6% a month; kit margin ₹1,708 and refill margin ₹141 from the pilot.",
         "With pilot economics CAC equals kit margin and churn drains refill income, so it never recovers. The strategy changes three inputs taken "
         "from the pilot: CAC of society demos + referrals only, churn of fast-first-harvest subscribers, and +10 points refill attach.",
         "Sustainable only with the data-driven strategy.")
    st.subheader("The plan")
    st.markdown("""
| Decision | Recommendation | Evidence |
|---|---|---|
| Whom to target | Health-first families (kit + refill); hobby terrace gardeners (kit + expansion); busy tech-savvy professionals (refill); skip budget households | Clustering |
| Which leads to work | Score every lead; top 30% get a 2-hour call and in-person demo; bottom 40% automated nurturing | Lead scoring |
| What to launch | Vertical + self-watering + organic seeds + hydroponic pods; refill, soil option, sensor at month 6 | Association rules |
| What price | ₹4,999 standard; ₹6,999 premium; ₹2,999 starter later | Regression + A/B test |
| Which channels | Society demo days + ₹300 referral reward; pause Instagram and Google | LTV:CAC |
| How to retain | First harvest within 35 days; week-1 onboarding call | Pilot diagnostics |
""")

elif page.startswith("1"):
    st.title("Data preparation: cleaning and transformation")
    c = st.columns(4)
    c[0].metric("Survey rows", f"{len(s_all):,}", f"{len(s_all) - len(raw_s):,} removed", delta_color="off")
    c[1].metric("Survey columns", s_all.shape[1], f"+{s_all.shape[1] - raw_s.shape[1]} derived", delta_color="off")
    c[2].metric("Pipeline rows", f"{len(p_all):,}", f"{len(p_all) - len(raw_p):,} removed", delta_color="off")
    c[3].metric("Cleaning actions logged", len(clog))
    st.markdown("Every change made by `02_clean_transform.ipynb` is recorded below. Rules were fixed in advance; the generator's answer key "
                "confirms all 123 planted survey problems were caught with no false alarms.")
    st.dataframe(clog, use_container_width=True, hide_index=True)
    st.markdown("#### Raw vs clean — survey")
    a, b = st.columns(2)
    a.caption("Raw export (form tool)")
    a.dataframe(raw_s[["Q1 City", "Q3 Household size", "Q7 Monthly veg & herb spend (Rs)", "Q16a Max price for starter system (Rs)",
                       "Q12 Features wanted"]].head(12), use_container_width=True, hide_index=True)
    b.caption("Clean")
    b.dataframe(s_all[["city", "city_tier", "household_size", "veg_spend", "wtp_system", "n_features"]].head(12), use_container_width=True, hide_index=True)
    st.markdown("#### Raw vs clean — pipeline")
    a, b = st.columns(2)
    a.caption("Raw CRM export")
    a.dataframe(raw_p[["lead_created", "city", "channel", "purchased", "order_value_inr"]].head(12), use_container_width=True, hide_index=True)
    b.caption("Clean")
    b.dataframe(p_all[["lead_created", "city", "channel", "purchased", "order_value_inr", "response_bucket"]].head(12), use_container_width=True, hide_index=True)
    st.markdown("#### Download the clean data")
    a, b = st.columns(2)
    a.download_button("survey_clean.csv", s_all.to_csv(index=False), "survey_clean.csv", "text/csv")
    b.download_button("pipeline_clean.csv", p_all.to_csv(index=False), "pipeline_clean.csv", "text/csv")

elif page.startswith("2"):
    st.title("Survey — descriptive analytics")
    num = s[["household_size", "veg_spend", "wtp_system", "wtp_refill", "intent_score", "subscribe_score"]]
    with st.expander("Summary statistics", expanded=True):
        st.dataframe(num.describe().T.round(1), use_container_width=True)
    for f in [A.d1_profile, A.d2_intent, A.d3_wtp, A.d4_features, A.d5_worries, A.d6_kept_up]:
        show(f, s, table=None if f is A.d3_wtp else data_table)

elif page.startswith("3"):
    st.title("Survey — diagnostic analytics")
    st.info("Correlation shows which answers move together; it does not prove cause. The exception is price: each respondent saw a "
            "**randomly assigned** kit and refill price, so differences across prices are caused by price.")
    for f in [A.g1_corr, A.g2_drivers, A.g3_price, A.g4_wtp_income, A.g5_space, A.g6_wtp_spend, A.g7_subscribe, A.g8_feature_lift]:
        tbl = (lambda r: r["table"]) if f is A.g3_price else None if f in (A.g4_wtp_income, A.g6_wtp_spend) else data_table
        show(f, s, table=tbl)

elif page.startswith("4"):
    st.title("Pilot sales pipeline — descriptive analytics")
    c = st.columns(4)
    c[0].metric("Leads", f"{len(p):,}")
    c[1].metric("Buyers", f"{int(p.purchased.sum()):,}", A.pct(p.purchased.mean(), 1) + " of leads", delta_color="off")
    c[2].metric("Average order", A.inr(p.order_value_inr.mean()) if p.purchased.sum() else "–")
    c[3].metric("CAC", A.inr(p.acquisition_cost_inr.sum() / max(p.purchased.sum(), 1)))
    for f in [A.p1_funnel, A.p2_volume, A.p3_channel_conv, A.p4_cac, A.p5_basket]:
        show(f, p, table=data_table)

elif page.startswith("5"):
    st.title("Pilot sales pipeline — diagnostic analytics")
    for f in [A.q1_price, A.q2_response, A.q3_rep, A.q4_drivers, A.q5_retention, A.q6_nps]:
        show(f, p, table=data_table)

else:
    st.title("End-to-end validation: what customers said vs what they did")
    st.caption("Uses the full survey and full pilot (filters do not apply here).")
    show(A.e1_price_validation, s_all, p_all, table=data_table)
    show(A.e2_stated_vs_actual, s_all, p_all, table=lambda r: r["table"].round(3))
    show(A.e3_channel_ltv, p_all, table=lambda r: r["table"].round(2))
    st.subheader("The finalised sales pipeline")
    st.dataframe(A.pipeline_plan(s_all, p_all), use_container_width=True, hide_index=True)
    st.markdown("""**Recommendation.** Launch at **₹4,999** in large gated societies in Bengaluru and Pune. Generate leads through
society demo days and a referral reward; drop Instagram and Google until a cheaper creative or offer is proven. Route every lead
to a rep within 2 hours and prefer in-person demos. Sell the refill at ₹299 to time-poor households, ship fast-growing crops first,
and phone every buyer in week one so the first harvest lands inside 35 days.""")

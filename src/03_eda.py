# %% [md]
# # Step 3 – Descriptive and diagnostic analytics (EDA)
# **Urban home-farming start-up (India) · synthetic data**
#
# This notebook answers one question in four stages: **will this business sell, and how should its sales pipeline be built?**
#
# | Section | Data | Type of analytics | Question |
# |---|---|---|---|
# | 1 | Survey (2,940) | Descriptive – *what is happening* | Who are the customers, what do they want, what will they pay? |
# | 2 | Survey | Diagnostic – *why* | What drives intent, willingness to pay and subscription? |
# | 3 | Pilot CRM (4,200 leads) | Descriptive | How did leads flow through the funnel? |
# | 4 | Pilot CRM | Diagnostic | Why do some leads buy and stay while others do not? |
# | 5 | Both | End-to-end validation | Did what customers *said* match what they *did*? Which pipeline should be finalised? |
#
# Every chart is followed by three short paragraphs: **What it shows** (the facts), **Why** (the diagnosis) and
# **So what** (the business decision). The numbers inside the text are computed from the data, not typed in, so they
# always match the chart. The calculations live in `analytics.py`; the drawing code is in `charts.py`.
#
# **Methods used:** frequency and percentage tables, medians and quartiles, histograms and box plots, Spearman rank
# correlation (for ordered survey scales), Pearson correlation with a linear trend, 95% Wilson confidence intervals
# for rates, chi-square tests of independence, a randomised price test (A/B), and Kaplan-Meier survival for subscribers.

# %%
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown, Image
import analytics as A, charts as C

s, p = A.load()
FIG = 0
def show(res):
    """Draw a result, save it to figures/ for the report, and print its explanation."""
    global FIG; FIG += 1
    fig = C.mpl(res); path = f"figures/{FIG:02d}_{res['id']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    display(Markdown(f"### {res['id'].upper()}. {res['title']}\n*Question: {res['question']}*"))
    display(Image(path, width=760))
    t = res["text"]
    display(Markdown(f"**What it shows.** {t['shows']}\n\n**Why.** {t['why']}\n\n**So what.** {t['so_what']}"))
print(f"survey {s.shape}, pipeline {p.shape}")

# %% [md]
# ## 1. Survey – descriptive analytics
# ### 1.0 Summary statistics
# Central tendency and spread for the main numeric answers. Medians are quoted alongside means because rupee
# amounts are rarely symmetric.

# %%
num = s[["household_size", "veg_spend", "spend_per_person", "wtp_system", "wtp_refill", "att_health", "att_eco", "att_tech",
         "att_time_poor", "att_hobby", "intent_score", "subscribe_score"]]
desc = num.describe(percentiles=[.25, .5, .75]).T.rename(columns={"50%": "median"})
desc["missing"] = num.isna().sum()
desc[["count", "missing", "mean", "std", "min", "25%", "median", "75%", "max"]].round(1)

# %% [md]
# **Reading the table.** The typical household has 3–4 people and spends about ₹6,000 a month on vegetables and herbs.
# The median maximum price for a starter kit is ₹4,500 and for a monthly refill ₹300. Attitude scores average above
# the midpoint (3) for health and eco motives – the 'green' sentiment is real – while average purchase intent sits
# below the midpoint, which is normal for a new product category.

# %%
show(A.d1_profile(s))

# %%
show(A.d2_intent(s))

# %%
show(A.d3_wtp(s))

# %%
show(A.d4_features(s))

# %%
show(A.d5_worries(s))

# %%
show(A.d6_kept_up(s))

# %% [md]
# ## 2. Survey – diagnostic analytics (correlation-based)
# Descriptive charts say *what*; these charts look for *why* by relating answers to each other.
# Correlation is not causation – with one important exception: the kit price (Q15) and refill price (Q17) were
# **assigned at random**, so differences across prices are caused by price.

# %%
show(A.g1_corr(s))

# %%
show(A.g2_drivers(s))

# %%
show(A.g3_price(s))
display(A.g3_price(s)["table"].assign(would_buy=lambda x: (x.would_buy * 100).round(1), afford=lambda x: (x.afford * 100).round(1),
                                     rev100=lambda x: x.rev100.round(0)).rename(columns={"price_shown": "price (₹)", "would_buy": "would buy %",
                                     "afford": "can afford %", "rev100": "revenue per 100 people (₹)"}))

# %%
show(A.g4_wtp_income(s))

# %%
show(A.g5_space(s))

# %%
show(A.g6_wtp_spend(s))

# %%
show(A.g7_subscribe(s))

# %%
show(A.g8_feature_lift(s))

# %% [md]
# ## 3. Pilot sales pipeline – descriptive analytics
# After the survey, a hypothetical 6-month pilot (January–June 2027, Bengaluru and Pune) sold the kit through six
# channels. Every lead was logged in a CRM from first contact to refill subscription.

# %%
k = pd.Series({
    "Leads": f"{len(p):,}", "Buyers": f"{int(p.purchased.sum()):,}", "Lead-to-sale": A.pct(p.purchased.mean(), 1),
    "Average order value (incl. GST)": A.inr(p.order_value_inr.mean()), "Revenue (incl. GST)": A.inr(p.order_value_inr.sum()),
    "Marketing spend": A.inr(p.acquisition_cost_inr.sum()), "Blended CAC": A.inr(p.acquisition_cost_inr.sum() / p.purchased.sum()),
    "Median days lead → purchase": f"{p.days_to_close.median():.0f}", "Refill subscribers": f"{int(p.refill_subscribed.sum()):,}",
    "Monthly churn (subscribers)": A.pct(A.churn_rate(p[p.refill_subscribed == 1]), 1),
    "NPS (promoters − detractors)": f"{((p.nps_score >= 9).sum() - (p.nps_score <= 6).sum()) / p.nps_score.notna().sum() * 100:+.0f}"},
    name="Pilot KPI")
k.to_frame()

# %%
show(A.p1_funnel(p))

# %%
show(A.p2_volume(p))

# %%
show(A.p3_channel_conv(p))

# %%
show(A.p4_cac(p))

# %%
show(A.p5_basket(p))

# %% [md]
# ## 4. Pilot sales pipeline – diagnostic analytics

# %%
show(A.q1_price(p))

# %%
show(A.q2_response(p))

# %%
show(A.q3_rep(p))

# %%
show(A.q4_drivers(p))

# %%
show(A.q5_retention(p))

# %%
show(A.q6_nps(p))

# %% [md]
# ## 5. End-to-end validation – survey vs pilot
# The survey predicts; the pilot measures. Putting them side by side shows which survey findings can be trusted
# for planning and by how much they must be corrected.

# %%
show(A.e1_price_validation(s, p))

# %%
r = A.e2_stated_vs_actual(s, p); show(r)
display(r["table"].assign(survey=lambda x: (x.survey * 100).round(1), pilot=lambda x: (x.pilot * 100).round(1), ratio=lambda x: x.ratio.round(2))
        .rename(columns={"survey": "survey %", "pilot": "pilot %", "ratio": "pilot ÷ survey"}))

# %%
r = A.e3_channel_ltv(p); show(r)
display(r["table"].assign(value=lambda x: x.value.round(2), ltv=lambda x: x.ltv.round(0), cac=lambda x: x.cac.round(0))
        .rename(columns={"label": "channel", "value": "LTV ÷ CAC", "ltv": "LTV (₹)", "cac": "CAC (₹)"}).iloc[::-1])

# %% [md]
# ## 6. The finalised sales pipeline
# Each stage of the funnel with the pilot result, the target for the next phase, and the chart that justifies it.

# %%
A.pipeline_plan(s, p)

# %% [md]
# ## 7. Key insights
# 1. **The market is real but narrow.** About one in five survey respondents is positive at the price they saw, and the
#    pilot converted 12% of leads. Demand is concentrated in households with a balcony or terrace and ₹1 lakh+ income.
# 2. **Price is the strongest lever and it was tested properly.** Both the randomised survey price and the field A/B
#    test show sharp drops above ₹4,999; the field drop was steeper than the survey predicted. Launch at ₹4,999.
# 3. **The demo is the conversion engine.** Two thirds of leads never see a demo; those who do buy at about three times
#    the average rate, more so in person and when contacted within two hours.
# 4. **Channel choice decides profitability.** Referrals and housing-society demos acquire buyers for a fraction of
#    the cost of Instagram or Google; paid social and search do not pay back over a customer's life.
# 5. **Retention is won in the first five weeks.** A first harvest within 35 days and an onboarding call roughly halve
#    monthly churn and lift satisfaction, which in turn feeds referrals.
# 6. **Survey answers need correction factors.** Purchase intent overstated actual buying by about half; subscription
#    intent by about 30%. Future forecasts should apply these ratios.

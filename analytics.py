"""
analytics.py - descriptive and diagnostic analysis for the home-farming start-up
================================================================================
One function per chart. Each returns a dict with the numbers behind the chart AND a three-part explanation
(what it shows / why it looks like this / so what for the business) computed from the data, so the notebook,
the Streamlit app and the report always quote the same figures. Renderers live in charts.py.

All data is SYNTHETIC (see 01_generate_data.py).
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- unit-economics assumptions (stated, not data)
UNIT_COST, FULFIL, PAY_RET, GST_KIT = 2400, 350, .05, .18           # kit landed cost, delivery, payment+returns, GST
ADDON_COST = {"expansion_tier": 1000, "smart_sensor": 800}            # cost incl. shipping
REFILL_PRICE, REFILL_MARGIN = 299, 141                                # ₹299/month incl. 5% GST -> ₹141 contribution
SPACE_ORDER = ["No balcony", "Small balcony", "Large balcony", "Terrace or garden"]
INCOME_ORDER = ["Below Rs 50,000", "Rs 50,000-1 lakh", "Rs 1-2 lakh", "Rs 2-3.5 lakh", "Above Rs 3.5 lakh", "Not disclosed"]
INTENT_ORDER = ["Very unlikely", "Unlikely", "Not sure", "Likely", "Very likely"]
CHANNELS = ["Housing-society demo", "Referral", "Corporate wellness", "Website (organic)", "Google search", "Instagram ads"]
FEATURES = {"f_vertical": "Vertical / stackable", "f_selfwater": "Self-watering", "f_hydroponic": "Hydroponic", "f_soil": "Soil / cocopeat",
            "f_smart": "Smart sensor + app", "f_organic": "Organic seeds", "f_subscription": "Refill subscription",
            "f_install": "Installation visits", "f_growlight": "Grow light"}
WORRIES = {"w_price": "Price", "w_space": "Lack of space", "w_time": "Time and upkeep", "w_dying": "Plants dying",
           "w_pests": "Pests or mess", "w_leak": "Leakage / society rules"}
NICE = {"income": "Income band", "household_size": "Household size", "space_level": "Growing space", "sun_level": "Sunlight",
        "veg_spend": "Monthly veg spend", "spoil_level": "Greens spoil", "att_health": "Wants pesticide-free", "att_eco": "Eco-minded",
        "att_tech": "Tech-comfortable", "att_time_poor": "Time-poor", "att_hobby": "Enjoys plants as hobby",
        "experience_level": "Gardening experience", "children": "Children at home", "price_shown": "Price shown (random)",
        "wtp_system": "Max price for kit", "intent_score": "Purchase intent", "subscribe_score": "Subscription intent"}


# ---------------------------------------------------------------- helpers
def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def inr(x):
    return ("-₹" if x < 0 else "₹") + f"{abs(x):,.0f}"


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion (better than +-2SE for small n or extreme rates)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rate_table(df, by, flag, order=None, min_n=1):
    g = df.groupby(by, observed=True)[flag].agg(["sum", "size"]).rename(columns={"sum": "k", "size": "n"})
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    g = g[g.n >= min_n]
    g["rate"] = g.k / g.n
    ci = [wilson(k, n) for k, n in zip(g.k, g.n)]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g.reset_index()


def chi2_p(df, by, flag):
    t = pd.crosstab(df[by], df[flag])
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan
    return stats.chi2_contingency(t)[1]


def p_text(pv):
    if pd.isna(pv):
        return "not testable on this subset"
    return "p < 0.001" if pv < .001 else f"p = {pv:.3f}"


def sig_word(pv):
    return "statistically significant" if pv < .05 else "not statistically significant"


def result(id, section, title, question, kind, data, shows, why, so_what, **kw):
    return dict(id=id, section=section, title=title, question=question, kind=kind, data=data,
                text=dict(shows=shows, why=why, so_what=so_what), **kw)


def small(n, need=60):
    return f" (only {n} rows in this selection - read with caution)" if n < need else ""


# ================================================================ 1. SURVEY - DESCRIPTIVE
def d1_profile(s):
    t = pd.crosstab(s.city_tier.map({1: "Metro (tier 1)", 2: "Tier 2", 3: "Tier 3"}), s.space_label, normalize="index") * 100
    t = t.reindex(columns=[c for c in SPACE_ORDER if c in t.columns])
    tight = t.get("No balcony", 0) + t.get("Small balcony", 0)
    n_tier = s.city_tier.value_counts().sort_index()
    m = tight.get("Metro (tier 1)", np.nan); t3 = tight.get("Tier 3", np.nan)
    return result("d1", "Survey - descriptive", "Who answered: growing space by city tier",
                  "Do customers in different cities have room for a growing system?", "stacked_hbar", t,
                  shows=f"{len(s):,} usable responses: {n_tier.get(1, 0):,} from metros, {n_tier.get(2, 0):,} from tier-2 and {n_tier.get(3, 0):,} from tier-3 cities. "
                        f"Each bar splits one city tier by the space people have to grow in.",
                  why=f"{m:.0f}% of metro respondents have only a window sill or a small balcony, against {t3:.0f}% in tier-3 cities, where terraces "
                      f"and independent houses are common. Space is scarce exactly where incomes and online buying are highest.",
                  so_what="The product must work in under 40 sq ft to sell in metros - a vertical, balcony-rail format, not a raised bed. "
                          "Tier-3 households have space but, as later charts show, lower budgets.",
                  unit="% of respondents in tier", ordinal=True)


def d2_intent(s):
    c = s.intent.value_counts(normalize=True).reindex(INTENT_ORDER).fillna(0) * 100
    top2, bot2 = c[["Likely", "Very likely"]].sum(), c[["Very unlikely", "Unlikely"]].sum()
    return result("d2", "Survey - descriptive", "Purchase intent at the price shown",
                  "How many people say they would buy?", "vbar", c.rename("value").reset_index().rename(columns={"intent": "label"}),
                  shows=f"Answers to Q15 (\"How likely are you to buy at ₹X?\"). Top-2-box (Likely + Very likely) = {top2:.0f}%; "
                        f"bottom-2-box = {bot2:.0f}%; {c['Not sure']:.0f}% are not sure.",
                  why="Each respondent saw one randomly chosen price between ₹2,999 and ₹9,999, so this mixes cheap and expensive offers. "
                      "Stated intent is also known to overstate real buying - people say \"likely\" to be agreeable.",
                  so_what=f"About {top2:.0f} in 100 people are positive: a real but niche market. The "
                          "\"Not sure\" block is the group demos and trials must convert; it is larger than the committed buyers.",
                  unit="% of respondents", colors="intent")


def d3_wtp(s):
    w = s.wtp_system.dropna()
    med, mean = w.median(), w.mean()
    a49, a59 = (w >= 4999).mean(), (w >= 5999).mean()
    return result("d3", "Survey - descriptive", "Maximum price people would pay for a starter kit",
                  "What price can the market bear?", "hist", w.to_frame("value"), bins=list(range(0, 13000, 1000)),
                  refs=[(med, f"median {inr(med)}"), (4999, "₹4,999 launch price")],
                  shows=f"Distribution of Q16a for {len(w):,} people who answered. Median {inr(med)}, mean {inr(mean)}. "
                        f"{pct(a49)} would pay ₹4,999 or more; {pct(a59)} would pay ₹5,999 or more.",
                  why="Spikes at round numbers (₹3,000, ₹5,000) are normal - people type round amounts. The mean sits close to the median "
                      "after the extra-zero typos were removed in cleaning, so the distribution is close to symmetric.",
                  so_what=f"A ₹4,999 kit is within budget for only about {pct(a49)} of the market; every ₹1,000 above that loses a large "
                          f"share. A cheaper entry kit (₹2,999-3,999) would open the larger group whose limit sits around {inr(med)}.",
                  unit="respondents", xlab="Maximum price for starter kit (₹)")


def d4_features(s):
    f = (s[list(FEATURES)].mean() * 100).rename(index=FEATURES).sort_values()
    top = f.sort_values(ascending=False)
    return result("d4", "Survey - descriptive", "Features customers want", "What should the product include?", "hbar",
                  f.rename("value").reset_index().rename(columns={"index": "label"}),
                  shows=f"Share of respondents ticking each feature in Q12 (they could tick several; average {s[list(FEATURES)].sum(axis=1).mean():.1f}). "
                        f"Top three: {top.index[0]} ({top.iloc[0]:.0f}%), {top.index[1]} ({top.iloc[1]:.0f}%), {top.index[2]} ({top.iloc[2]:.0f}%).",
                  why="The most-ticked features remove effort or save space - the two biggest barriers in a flat. Tech features "
                      "(smart sensor) and services (installation) appeal to a smaller, richer group.",
                  so_what=f"The base kit should be vertical and self-watering. The smart sensor ({f.get('Smart sensor + app', 0):.0f}%) belongs "
                          f"in an add-on, not the base price. Only {f.get('Refill subscription', 0):.0f}% ask for a subscription "
                          f"unprompted, so the refill must be sold on convenience, not assumed.",
                  unit="% of respondents ticking")


def d5_worries(s):
    w = (s[list(WORRIES)].mean() * 100).rename(index=WORRIES).sort_values()
    quit_, other = s.loc[s.tried_and_quit == 1, "w_dying"].mean(), s.loc[s.tried_and_quit == 0, "w_dying"].mean()
    lowinc, hiinc = s.loc[s.income <= 2, "w_price"].mean(), s.loc[s.income >= 3, "w_price"].mean()
    none = (s.n_worries == 0).mean()
    top = w.sort_values(ascending=False)
    return result("d5", "Survey - descriptive", "Biggest worries about growing at home", "What stops people from buying?", "hbar",
                  w.rename("value").reset_index().rename(columns={"index": "label"}),
                  shows=f"Share ticking each worry in Q14 (up to two). Top worries: {top.index[0]} ({top.iloc[0]:.0f}%) and "
                        f"{top.index[1]} ({top.iloc[1]:.0f}%). {pct(none)} ticked no worry at all.",
                  why=f"\"Plants dying\" is a fear learned from failure: {pct(quit_)} of people who tried growing and gave up tick it, "
                      f"against {pct(other)} of everyone else. Price worry is income-driven: {pct(lowinc)} of households under ₹1 lakh a month "
                      f"tick it, against {pct(hiinc)} above.",
                  so_what="Marketing should promise a result, not a product: a guaranteed first harvest, pre-seeded pods and a "
                          "replace-if-it-dies policy answer the top fear directly and cost little.",
                  unit="% of respondents ticking")


def d6_kept_up(s):
    t = s[s.experience_level >= 1]
    tab = pd.crosstab(t.experience, t.kept_up_6m, normalize="index").reindex(columns=["Kept up", "Stopped"]).fillna(0) * 100
    tab = tab.reindex([e for e in ["Kept a few plants", "Grown vegetables before", "Grow vegetables now"] if e in tab.index])
    kept = (t.kept_up_6m == "Kept up").mean()
    monthly = 1 - kept ** (1 / 6) if kept > 0 else np.nan
    return result("d6", "Survey - descriptive", "Did people who tried growing keep it up for 6 months?",
                  "How big is the drop-off risk after purchase?", "stacked_hbar", tab,
                  shows=f"Among {len(t):,} respondents who have tried growing (Q10-Q11), {pct(kept)} kept it up for six months or more; "
                        f"{pct(1 - kept)} stopped.",
                  why=f"Light gardeners (\"kept a few plants\") stopped {tab.loc['Kept a few plants', 'Stopped']:.0f}% of the time, people who "
                      f"grow vegetables now only {tab.loc['Grow vegetables now', 'Stopped']:.0f}%: the habit, not the equipment, is what lasts. "
                      "Losing about half of starters within six months is typical of hobby products.",
                  so_what=f"If subscribers behave like this, monthly churn would be about {pct(monthly, 1)} - a refill customer would last only "
                          f"~{1 / monthly:.0f} months. The pilot tests whether onboarding and refills can do better.",
                  unit="% of those who tried", series_colors=["#2a78d6", "#e34948"], keep=kept, monthly=monthly)


# ================================================================ 2. SURVEY - DIAGNOSTIC
CORR_COLS = ["income", "household_size", "space_level", "sun_level", "veg_spend", "spoil_level", "att_health", "att_eco", "att_tech",
             "att_time_poor", "att_hobby", "experience_level", "price_shown", "wtp_system", "intent_score", "subscribe_score"]


def g1_corr(s):
    c = s[CORR_COLS].corr(method="spearman").round(2)
    c.index = c.columns = [NICE[x] for x in CORR_COLS]
    off = c.where(~np.eye(len(c), dtype=bool)).stack()
    off = off[[a < b for a, b in off.index]]
    strongest = off.abs().sort_values(ascending=False).head(3).index
    pi = c["Purchase intent"].drop(["Purchase intent", "Subscription intent", "Max price for kit"]).sort_values()
    return result("g1", "Survey - diagnostic", "Correlation matrix of survey answers",
                  "Which answers move together?", "heatmap", c,
                  shows="Spearman rank correlation between every pair of numeric answers (+1 = rise together, -1 = one rises as the other "
                        "falls, 0 = no link). Spearman is used because most answers are ordered scales, not true numbers.",
                  why="Strongest pairs: " + "; ".join(f"{a} & {b} ({c.loc[a, b]:+.2f})" for a, b in strongest) + ". Income, space and "
                      "spend cluster together (richer households have bigger homes and bigger food bills), so their separate effects "
                      "overlap. Price shown was assigned at random, so its correlation with intent is causal, not coincidence.",
                  so_what=f"Among the factors that could drive it, purchase intent links most strongly with {pi.index[-1].lower()} ({pi.iloc[-1]:+.2f}) and "
                          f"{pi.index[-2].lower()} ({pi.iloc[-2]:+.2f}), and falls with {pi.index[0].lower()} ({pi.iloc[0]:+.2f}). "
                          "The next chart ranks every driver.",
                  unit="Spearman r")


def g2_drivers(s):
    rows = []
    for col in [c for c in CORR_COLS if c not in ("intent_score", "subscribe_score", "wtp_system")]:
        d = s[[col, "intent_score"]].dropna()
        r, pv = stats.spearmanr(d[col], d.intent_score)
        rows.append(dict(label=NICE[col], value=r, p=pv))
    t = pd.DataFrame(rows).sort_values("value")
    pos = t[t.value > 0].sort_values("value", ascending=False); neg = t[t.value < 0].sort_values("value")
    weak = t[t.value.abs() < .05].label.tolist()
    return result("g2", "Survey - diagnostic", "What drives purchase intent?", "Which customer traits predict a 'yes'?", "hbar_signed", t,
                  shows="Each bar is the rank correlation of one factor with purchase intent (1-5). Blue bars raise intent, red bars lower it.",
                  why=f"The strongest positive signals are {pos.label.iloc[0].lower()} ({pos.value.iloc[0]:+.2f}), {pos.label.iloc[1].lower()} "
                      f"({pos.value.iloc[1]:+.2f}) and {pos.label.iloc[2].lower()} ({pos.value.iloc[2]:+.2f}); attitudes sit alongside income, "
                      f"not behind it. The price shown has the strongest negative link ({t.set_index('label').value.get('Price shown (random)', 0):+.2f}) "
                      "- and because it was randomised, higher price causes lower intent. "
                      + (f"Weak or no link: {', '.join(weak).lower()}." if weak else ""),
                  so_what="Target households that combine budget with the right attitude - health-conscious, eco-minded, with a balcony - "
                          "e.g. organic-store shoppers and society gardening groups. Price is the biggest single lever the company controls.",
                  unit="Spearman r with purchase intent")


def g3_price(s):
    g = s.groupby("price_shown").agg(would_buy=("would_buy", "mean"), afford=("can_afford_shown", "mean"), n=("would_buy", "size")).reset_index()
    g["rev100"] = g.would_buy * 100 * g.price_shown
    best = g.loc[g.rev100.idxmax()]
    lo_, hi_ = g.iloc[0], g.iloc[-1]
    long = pd.concat([g.assign(series="Would buy (top-2 box)", value=g.would_buy * 100),
                      g.assign(series="Max price >= price shown", value=g.afford * 100)])
    pv = chi2_p(s, "price_shown", "would_buy")
    r49 = g.set_index("price_shown").would_buy.get(4999, np.nan); r59 = g.set_index("price_shown").would_buy.get(5999, np.nan)
    return result("g3", "Survey - diagnostic", "Price test: intent falls as price rises",
                  "How sensitive is demand to price?", "line", long, x="price_shown", series="series", table=g,
                  shows=f"Each respondent saw one random price. At {inr(lo_.price_shown)} {pct(lo_.would_buy)} would buy; at "
                        f"{inr(hi_.price_shown)} only {pct(hi_.would_buy)}. The second line is the share whose stated maximum price "
                        f"covers the price shown. Difference across prices is {sig_word(pv)} ({p_text(pv)}).",
                  why="Both lines fall together - intent drops because the price passes people's budget, not because they dislike the "
                      "product. The gap between the lines shows some people would 'like' to buy at a price above what they said they'd pay.",
                  so_what=f"Moving from ₹4,999 to ₹5,999 cuts intent from {pct(r49)} to {pct(r59)}. Revenue per 100 respondents peaks at "
                          f"{inr(best.price_shown)} - but profit also depends on unit cost and acquisition cost, so the pilot tests "
                          "₹4,999-5,999 in the field.",
                  unit="% of respondents", xlab="Price shown (₹)")


def g4_wtp_income(s):
    d = s[["income_label", "wtp_system"]].dropna()
    d["income_label"] = d.income_label.astype(str)
    med = d.groupby("income_label").wtp_system.median().reindex([i for i in INCOME_ORDER if i in d.income_label.unique()])
    r, pv = stats.spearmanr(s.dropna(subset=["income", "wtp_system"]).income, s.dropna(subset=["income", "wtp_system"]).wtp_system)
    lo_, hi_ = med.drop("Not disclosed", errors="ignore").iloc[0], med.drop("Not disclosed", errors="ignore").iloc[-1]
    nd = med.get("Not disclosed", np.nan)
    return result("g4", "Survey - diagnostic", "Maximum kit price by income band", "Does income explain willingness to pay?", "box",
                  d.rename(columns={"income_label": "group", "wtp_system": "value"}), order=list(med.index),
                  refs=[(4999, "₹4,999")],
                  shows=f"Box = middle 50% of answers, line = median, whiskers = typical range. Median max price rises from {inr(lo_)} in the "
                        f"lowest band to {inr(hi_)} in the highest (Spearman r = {r:+.2f}, {p_text(pv)}).",
                  why="Income sets the budget ceiling, but the boxes overlap a lot: attitude and space explain the rest. "
                      + (f"People who did not disclose income have a median of {inr(nd)}, close to the middle bands - so leaving them "
                         "out would not bias results much." if not pd.isna(nd) else ""),
                  so_what="A ₹4,999 kit sits at or above the median for households earning under ₹1 lakh a month. Lead targeting "
                          "should focus on ₹1 lakh+ households, or an EMI / cheaper kit is needed for the rest.",
                  unit="Max price for kit (₹)")


def g5_space(s):
    t = rate_table(s, "space_label", "would_buy", SPACE_ORDER)
    pv = chi2_p(s, "space_label", "would_buy")
    return result("g5", "Survey - diagnostic", "Purchase intent by growing space", "Does more space mean more buyers?", "vbar_ci",
                  t.rename(columns={"space_label": "label"}),
                  shows=f"Share who would buy (top-2 box) in each space group, with 95% confidence intervals. It rises from "
                        f"{pct(t.rate.iloc[0])} (no balcony) to {pct(t.rate.iloc[-1])} (terrace or garden); {p_text(pv)}.",
                  why="More space means more light and more room to grow a meaningful amount of food, so the product's value is higher. "
                      "Space also tracks income, so part of this gap is an income effect.",
                  so_what="Window-sill households are the weakest segment and should not be the launch target; large-balcony and terrace "
                          "homes in gated societies are, which supports housing-society demos as a channel.",
                  unit="% would buy")


def g6_wtp_spend(s):
    d = s[["veg_spend", "wtp_system", "income"]].dropna()
    r, pv = stats.pearsonr(d.veg_spend, d.wtp_system)
    slope, icpt = np.polyfit(d.veg_spend, d.wtp_system, 1)
    samp = d.sample(min(len(d), 1500), random_state=1)
    return result("g6", "Survey - diagnostic", "Monthly vegetable spend vs maximum kit price",
                  "Do bigger food spenders value the kit more?", "scatter", samp, x="veg_spend", y="wtp_system", fit=(slope, icpt),
                  shows=f"Each dot is one respondent (sample of {len(samp):,}). Pearson r = {r:+.2f} ({p_text(pv)}); the trend line says each extra "
                        f"₹1,000 of monthly vegetable spend goes with {inr(slope * 1000)} more willingness to pay.",
                  why="Families who spend more on fresh produce see more value in growing some of it, and they are usually richer. The wide "
                      "scatter shows spend is only a moderate predictor - attitude matters as much.",
                  so_what="Monthly vegetable spend is a useful screening question for sales calls and a credible value message "
                          "(\"save on the herbs and greens you already buy\").",
                  unit="₹", xlab="Monthly vegetable & herb spend (₹)", ylab="Max price for kit (₹)")


def g7_subscribe(s):
    d = s.assign(time=np.where(s.att_time_poor >= 4, "Time-poor (4-5)", "Not time-poor (1-3)"))
    t = d.groupby(["refill_price_shown", "time"]).would_subscribe.mean().unstack() * 100
    long = t.stack().rename("value").reset_index().rename(columns={"refill_price_shown": "label", "time": "series"})
    long["label"] = "₹" + long.label.astype(str)
    overall = s.groupby("refill_price_shown").would_subscribe.mean()
    pv = chi2_p(s, "refill_price_shown", "would_subscribe")
    gap = (t.iloc[:, 1] - t.iloc[:, 0]).mean() if t.shape[1] == 2 else np.nan
    return result("g7", "Survey - diagnostic", "Refill subscription intent by price and time pressure",
                  "Will customers pay monthly, and who?", "grouped_vbar", long, series_order=["Not time-poor (1-3)", "Time-poor (4-5)"],
                  shows=f"Share saying they would probably or definitely subscribe, by refill price shown (random) and by how time-poor they are. "
                        f"Overall: {pct(overall.get(199, np.nan))} at ₹199, {pct(overall.get(299, np.nan))} at ₹299, "
                        f"{pct(overall.get(399, np.nan))} at ₹399 ({p_text(pv)}).",
                  why=f"Time-poor households are about {gap:.0f} points more willing at every price: a refill that arrives ready to plant "
                      "saves them effort, which is what they pay for. Price matters, but less than for the kit.",
                  so_what=f"Monthly revenue per 100 people is {inr(overall.get(199, 0) * 100 * 199)} at ₹199, {inr(overall.get(299, 0) * 100 * 299)} "
                          f"at ₹299 and {inr(overall.get(399, 0) * 100 * 399)} at ₹399. After GST, seeds and shipping (about ₹135 a box), contribution is "
                          f"{inr(overall.get(199, 0) * 100 * (199 / 1.05 * .97 - 135))} at ₹199 vs {inr(overall.get(299, 0) * 100 * (299 / 1.05 * .97 - 135))} "
                          "at ₹299, so ₹299 is the better price; ₹399 loses too many people. Sell the subscription to busy dual-income households as "
                          "'never think about seeds again'. Stated intent is optimistic; the pilot measures the real rate.",
                  unit="% would subscribe")


def g8_feature_lift(s):
    b, n = s[s.would_buy == 1], s[s.would_buy == 0]
    t = pd.DataFrame({"label": list(FEATURES.values()),
                      "value": [(b[f].mean() - n[f].mean()) * 100 for f in FEATURES]}).sort_values("value")
    top = t.sort_values("value", ascending=False)
    return result("g8", "Survey - diagnostic", "Which features separate likely buyers from the rest?",
                  "Which features should the pitch lead with?", "hbar_signed", t,
                  shows=f"Percentage-point difference in how often likely buyers ({len(b):,}) and others ({len(n):,}) tick each feature. "
                        f"Largest gaps: {top.label.iloc[0]} ({top.value.iloc[0]:+.0f} pts) and {top.label.iloc[1]} ({top.value.iloc[1]:+.0f} pts).",
                  why="A feature everybody wants (high in the feature-demand chart) but with a small gap here is a hygiene factor - expected, not persuasive. "
                      "A feature with a big gap is what excited buyers are looking for.",
                  so_what=f"Lead the pitch with {top.label.iloc[0].lower()} and {top.label.iloc[1].lower()}. "
                          + (f"Negative gaps ({', '.join(t[t.value < -2].label.str.lower())}) are features wanted mainly by households with "
                             f"little space or light - {pct(s[s.space_level <= 1].f_vertical.mean())} of window-sill and small-balcony homes "
                             f"want a vertical system against {pct(s[s.space_level >= 2].f_vertical.mean())} of others - and those households "
                             "buy least. They are still needed in the product for metros, but they will not sell it on their own."
                             if (t.value < -2).any() else "Features near zero are expected as standard, not reasons to buy."),
                  unit="percentage-point gap (buyers - others)")


# ================================================================ 3. PIPELINE - DESCRIPTIVE
def _active(p):
    sub = p[p.refill_subscribed == 1]
    return int((sub.churned == 0).sum())


def p1_funnel(p):
    stages = [("Leads", len(p)), ("Contacted", int(p.contacted.sum())), ("Demo done", int(p.demo_done.sum())),
              ("Purchased", int(p.purchased.sum())), ("Subscribed to refills", int(p.refill_subscribed.sum())),
              ("Still subscribed at 31 Jul", _active(p))]
    t = pd.DataFrame(stages, columns=["label", "count"])
    t["pct_of_leads"] = t["count"] / max(len(p), 1) * 100
    t["step_conv"] = t["count"] / t["count"].shift(1) * 100
    return result("p1", "Pipeline - descriptive", "Sales funnel of the 6-month pilot", "Where do leads drop out?", "funnel", t,
                  shows=f"{len(p):,} leads → {t['count'][1]:,} contacted ({t.step_conv[1]:.0f}%) → {t['count'][2]:,} demos "
                        f"({t.step_conv[2]:.0f}% of contacted) → {t['count'][3]:,} buyers ({t.step_conv[3]:.0f}% of demos) → "
                        f"{t['count'][4]:,} refill subscribers ({t.step_conv[4]:.0f}% of buyers). Lead-to-sale: {t.pct_of_leads[3]:.1f}%.{small(len(p))}",
                  why=f"The biggest loss is before the demo: {100 - t.pct_of_leads[2]:.0f}% of leads never see the product, mostly low-intent "
                      f"online leads that were not reached or not interested. Once a household has a demo, {t.step_conv[3]:.0f}% buy - "
                      f"{t.step_conv[3] / max(t.pct_of_leads[3], 1e-9):.1f}x the lead-level rate.",
                  so_what="The demo is the conversion engine. Sales effort should go into getting more leads to a demo (faster follow-up, "
                          "society events), not into more top-of-funnel ads.",
                  unit="leads")


def p2_volume(p):
    d = p.assign(ch=p.channel.where(p.channel.isin(CHANNELS), "Unknown"))
    t = pd.crosstab(d.lead_month, d.ch).reindex(columns=[c for c in CHANNELS + ["Unknown"] if c in d.ch.unique()])
    first, last = t.sum(axis=1).iloc[0], t.sum(axis=1).iloc[-1]
    ref_share = t.div(t.sum(axis=1), axis=0).get("Referral", pd.Series([0, 0]))
    ig = t.div(t.sum(axis=1), axis=0).get("Instagram ads", pd.Series([0, 0]))
    return result("p2", "Pipeline - descriptive", "Leads per month by channel", "How did lead volume and mix change?", "stacked_vbar", t,
                  shows=f"Leads grew from {first:,} in {t.index[0]} to {last:,} in {t.index[-1]}. Referral share went from "
                        f"{pct(ref_share.iloc[0])} to {pct(ref_share.iloc[-1])}; Instagram from {pct(ig.iloc[0])} to {pct(ig.iloc[-1])}.",
                  why="Paid ads start the pipeline; referrals can only grow once there are happy customers to refer, which is why they "
                      "build up from month 3. A small 'Unknown' block is leads whose source was not recorded.",
                  so_what="Referral growth is the healthiest signal in the pilot - it is the cheapest channel (see P4). Recording the "
                          "source for every lead is a process fix the sales team should make now.",
                  unit="leads")


def p3_channel_conv(p):
    t = rate_table(p[p.channel != "Unknown"], "channel", "purchased", CHANNELS)
    best, worst = t.sort_values("rate").iloc[-1], t.sort_values("rate").iloc[0]
    pv = chi2_p(p[p.channel != "Unknown"], "channel", "purchased")
    return result("p3", "Pipeline - descriptive", "Lead-to-purchase rate by channel", "Which channels bring buyers, not just leads?",
                  "hbar_ci", t.rename(columns={"channel": "label"}).iloc[::-1],
                  shows=f"Share of each channel's leads that bought, with 95% intervals. Best: {best.channel} {pct(best.rate, 1)} "
                        f"({int(best.k)} of {int(best.n):,}); worst: {worst.channel} {pct(worst.rate, 1)} ({int(worst.k)} of {int(worst.n):,}). "
                        f"The differences between channels are {sig_word(pv)} ({p_text(pv)}).",
                  why="Society demos and referrals arrive with trust and a chance to see the product; ad and website leads are often a "
                      f"single click with little intent (Instagram: {pct(t.set_index('channel').rate.get('Instagram ads', np.nan), 1)}). "
                      "The highest-volume channels are not the best-converting ones.",
                  so_what=f"A {worst.channel} lead is worth about {best.rate / max(worst.rate, 1e-9):.0f}x less than a {best.channel} lead. "
                          "Budget should be judged on cost per buyer, not cost per lead (next chart).",
                  unit="% of leads that bought")


def unit_economics(p):
    """Per-buyer contribution from the kit and add-ons (net of GST, payment/returns, unit and delivery cost)."""
    b = p[p.purchased == 1]
    kit = b.order_value_ex_gst * (1 - PAY_RET) - UNIT_COST - FULFIL - b.expansion_tier.fillna(0) * ADDON_COST["expansion_tier"] \
        - b.smart_sensor.fillna(0) * ADDON_COST["smart_sensor"]
    return kit


def churn_rate(sub):
    """Monthly churn = churn events / subscriber-months observed (handles customers still active at the cut-off)."""
    exposure = sub.months_active.clip(lower=0).fillna(0).sum()
    return (sub.churned == 1).sum() / exposure if exposure > 0 else np.nan


def p4_cac(p):
    d = p[p.channel != "Unknown"]
    g = d.groupby("channel").agg(cost=("acquisition_cost_inr", "sum"), buyers=("purchased", "sum"), leads=("purchased", "size"))
    g["cac"] = g.cost / g.buyers.replace(0, np.nan)
    g = g.reindex([c for c in CHANNELS if c in g.index])
    margin = unit_economics(p).mean()
    blended = p.acquisition_cost_inr.sum() / max(p.purchased.sum(), 1)          # all leads, incl. unknown source
    t = g.reset_index().rename(columns={"channel": "label", "cac": "value"}).sort_values("value", ascending=False)
    above = t[t.value > margin].label.tolist()
    return result("p4", "Pipeline - descriptive", "Cost to acquire one buyer (CAC) by channel",
                  "Which channels are affordable?", "hbar", t, refs=[(margin, f"kit margin per buyer {inr(margin)}")],
                  shows=f"Marketing spend per channel ÷ buyers from that channel. Blended CAC = {inr(blended)}. The line marks the average "
                        f"contribution from the first sale ({inr(margin)} after GST, unit cost, delivery and returns).",
                  why="CAC = cost per lead ÷ lead-to-purchase rate. " + "; ".join(
                          f"{c}: {inr(g.loc[c, 'cost'] / g.loc[c, 'leads'])} per lead ÷ {pct(g.loc[c, 'buyers'] / g.loc[c, 'leads'], 1)} = {inr(g.loc[c, 'cac'])}"
                          for c in ["Instagram ads", "Housing-society demo", "Referral"] if c in g.index)
                      + ". A cheap lead that rarely buys is an expensive buyer.",
                  so_what=("Channels above the line lose money on the first sale and only pay back if the customer subscribes: "
                           + ", ".join(above) + ". " if above else "Every channel pays back on the first sale. ")
                          + "Shift budget from the costliest channels to society demos and referral rewards.",
                  unit="₹ per buyer", margin=margin, blended=blended)


def p5_basket(p):
    b = p[p.purchased == 1]
    kind = np.select([(b.expansion_tier == 1) & (b.smart_sensor == 1), b.expansion_tier == 1, b.smart_sensor == 1],
                     ["Kit + expansion + sensor", "Kit + expansion tier", "Kit + smart sensor"], "Kit only")
    order = ["Kit only", "Kit + expansion tier", "Kit + smart sensor", "Kit + expansion + sensor"]
    t = pd.Series(kind).value_counts(normalize=True).reindex(order).fillna(0) * 100
    aov = b.order_value_inr.mean()
    disc = b.discount_used.mean()
    big = b.groupby("space_label", observed=True).expansion_tier.mean()
    return result("p5", "Pipeline - descriptive", "What buyers put in the basket", "Do customers buy add-ons?", "vbar",
                  t.rename("value").reset_index().rename(columns={"index": "label"}),
                  shows=f"{len(b):,} orders. Average order value {inr(aov)} (incl. GST). {t['Kit only']:.0f}% bought the kit alone; "
                        f"{100 - t['Kit only']:.0f}% added at least one extra. {pct(disc)} used a 10% discount.",
                  why=f"Add-ons follow space: {pct(big.get('Large balcony', 0))} of large-balcony and {pct(big.get('Terrace or garden', 0))} of "
                      f"terrace buyers took the expansion tier, against {pct(big.get('Small balcony', 0))} with a small balcony.",
                  so_what="Add-ons lift revenue per buyer at good margin. Offer the expansion tier as a 'grow later' upgrade after the "
                          "first harvest rather than at checkout, when price resistance is highest.",
                  unit="% of orders")


# ================================================================ 4. PIPELINE - DIAGNOSTIC
def q1_price(p):
    d = p[p.demo_done == 1]
    t = rate_table(d, "price_label", "purchased", ["₹4,999", "₹5,499", "₹5,999"])
    m = unit_economics(p)
    b = p[p.purchased == 1].assign(m=m)
    margin_by = b.groupby("price_label").m.mean()
    t["contrib_per_demo"] = t.rate * t.price_label.map(margin_by)
    pv = chi2_p(d, "price_label", "purchased")
    best = t.loc[t.contrib_per_demo.idxmax()]; best_label = best["price_label"]
    return result("q1", "Pipeline - diagnostic", "A/B price test: purchase rate after a demo", "Which launch price works best in the field?",
                  "vbar_ci", t.rename(columns={"price_label": "label"}), table=t,
                  shows=f"Leads were randomly offered ₹4,999, ₹5,499 or ₹5,999. After a demo, {pct(t.rate.iloc[0])} bought at ₹4,999, "
                        f"{pct(t.rate.iloc[1])} at ₹5,499 and {pct(t.rate.iloc[2])} at ₹5,999 ({sig_word(pv)}, {p_text(pv)}).",
                  why="Because price was randomised, the difference is caused by price, not by who saw it. Each ₹500 step loses roughly "
                      f"{(t.rate.iloc[0] - t.rate.iloc[-1]) / 2 * 100:.0f} points of conversion.",
                  so_what=f"Kit contribution per demo is highest at {best_label} ({inr(best.contrib_per_demo)} per demo). The extra margin "
                          "from a higher price does not make up for the lost buyers, which also lose future refill income.",
                  unit="% of demos that bought")


def q2_response(p):
    d = p[p.contacted == 1]
    t = rate_table(d, "response_bucket", "purchased", ["Under 2 h", "2-6 h", "6-24 h", "Over 24 h"])
    pv = chi2_p(d, "response_bucket", "purchased")
    slow = (p.hours_to_first_contact > 24).mean()
    return result("q2", "Pipeline - diagnostic", "Purchase rate by speed of first contact", "Does replying fast win more sales?", "vbar_ci",
                  t.rename(columns={"response_bucket": "label"}),
                  shows=f"Of contacted leads, {pct(t.rate.iloc[0], 1)} bought when called within 2 hours vs {pct(t.rate.iloc[-1], 1)} "
                        f"after more than 24 hours ({p_text(pv)}). {pct(slow)} of all leads waited more than a day.",
                  why="Interest fades quickly after someone fills in a form; a late call reaches a lead who has moved on or bought "
                      "elsewhere. Note this is observational - busy reps may also be slower and weaker - but the direction is consistent.",
                  so_what="Set a 2-hour response target with automatic WhatsApp acknowledgement and lead routing to a free rep. "
                          "It costs little and lifts conversion across every channel.",
                  unit="% of contacted leads that bought")


def q3_rep(p):
    d = p[p.demo_done == 1]
    t = d.groupby(["sales_rep", "demo_type"]).purchased.mean().unstack() * 100
    overall = d.groupby("sales_rep").purchased.agg(["mean", "size"]).sort_values("mean")
    t = t.reindex(overall.index)
    long = t.stack().rename("value").reset_index().rename(columns={"sales_rep": "label", "demo_type": "series"})
    ip, vc = d[d.demo_type == "In person"].purchased.mean(), d[d.demo_type == "Video call"].purchased.mean()
    pv = chi2_p(d, "sales_rep", "purchased")
    return result("q3", "Pipeline - diagnostic", "Demo-to-purchase by sales rep and demo type",
                  "Is conversion about the rep, the format, or both?", "grouped_vbar", long, series_order=["Video call", "In person"],
                  shows=f"In-person demos convert {pct(ip)} vs {pct(vc)} on video. Across reps, demo conversion ranges from "
                        f"{pct(overall['mean'].iloc[0])} ({overall.index[0]}) to {pct(overall['mean'].iloc[-1])} ({overall.index[-1]}); "
                        f"rep differences are {sig_word(pv)} ({p_text(pv)}).",
                  why="Touching the product and seeing a live harvest removes the 'will it work for me' doubt. Rep skill adds on top: "
                      "the best reps beat the weakest in both formats, so it is not just that they got easier leads.",
                  so_what=f"Pair the weakest reps with {overall.index[-1]} for shadowing and script the top rep's demo. Push in-person "
                          "demos where lead density allows (societies); keep video for scattered leads.",
                  unit="% of demos that bought")


def q4_drivers(p):
    d = p[p.demo_done == 1].assign(in_person=lambda x: (x.demo_type == "In person").astype(int),
                                   exp=lambda x: x.gardening_experience.map({"No experience": 0, "Kept a few plants": 1, "Grown vegetables before": 2, "Grow vegetables now": 3}))
    cols = {"price_offered": "Price offered", "hours_to_first_contact": "Hours to first contact", "space_level": "Growing space",
            "income_level": "Income band", "household_size": "Household size", "in_person": "In-person demo", "exp": "Gardening experience"}
    rows = []
    for c, lab in cols.items():
        x = d[[c, "purchased"]].dropna()
        r, pv = stats.spearmanr(x[c], x.purchased)
        rows.append(dict(label=lab, value=r, p=pv))
    t = pd.DataFrame(rows).sort_values("value")
    return result("q4", "Pipeline - diagnostic", "What predicts a purchase after a demo?", "Which levers matter most at the close?",
                  "hbar_signed", t,
                  shows=f"Rank correlation of each factor with buying, among {len(d):,} leads who had a demo. Blue raises the chance of "
                        "purchase, red lowers it.",
                  why=f"Customer fit ({t.set_index('label').value.idxmax().lower()}) and price are the strongest factors. "
                      "Household size and experience barely matter once someone has seen a demo. Income is blank for a quarter of leads, so "
                      "its bar uses fewer rows.",
                  so_what="Qualify leads on space and income before booking a demo, keep price at the low test point, and demo in "
                          "person - these three levers are in the company's hands.",
                  unit="Spearman r with purchase")


def km_curve(sub, months=6):
    """Kaplan-Meier share still subscribed after m months. months_active = month of cancellation for churners,
    months observed so far for customers still active at the cut-off (censored)."""
    t = sub.months_active.clip(lower=0).fillna(0).values
    e = (sub.churned == 1).values
    surv, out = 1.0, [1.0]
    for m in range(1, months + 1):
        at_risk = (t >= m).sum()
        events = ((t == m) & e).sum()
        surv *= (1 - events / at_risk) if at_risk > 0 else 1
        out.append(surv)
    return out


def q5_retention(p):
    sub = p[p.refill_subscribed == 1]
    groups = {"First harvest within 35 days": sub[sub.fast_first_harvest == 1], "First harvest after 35 days": sub[sub.fast_first_harvest == 0]}
    rows = []
    for g, d in groups.items():
        for m, v in enumerate(km_curve(d)):
            rows.append(dict(series=g, month=m, value=v * 100))
    t = pd.DataFrame(rows)
    cr = {g: churn_rate(d) for g, d in groups.items()}
    cr_all = churn_rate(sub)
    onb = {k: churn_rate(sub[sub.onboarding_call == k]) for k in (1, 0)}
    s6 = t[t.month == 6].set_index("series").value
    return result("q5", "Pipeline - diagnostic", "Refill subscribers still active, by speed of first harvest",
                  "What keeps subscribers from cancelling?", "line", t, x="month", series="series",
                  shows=f"Kaplan-Meier curve: share of {len(sub):,} subscribers still paying each month (customers still active at 31 July "
                        f"are counted correctly, not as churners). Overall monthly churn is {pct(cr_all, 1)}.",
                  why=f"Subscribers whose first harvest came within 35 days churn at {pct(cr['First harvest within 35 days'], 1)} a month, "
                      f"against {pct(cr['First harvest after 35 days'], 1)} for slower starts. An early harvest proves the system works "
                      f"and builds the habit. Onboarding calls help too: {pct(onb[1], 1)} vs {pct(onb[0], 1)} monthly churn.",
                  so_what="The first five weeks decide lifetime value. Ship fast-growing microgreens in the first refill, call every "
                          "new customer in week 1, and track 'days to first harvest' as the key operating metric.",
                  unit="% still subscribed", xlab="Months since subscribing", churn=cr_all, s6=s6)


def q6_nps(p):
    b = p[p.purchased == 1].dropna(subset=["nps_score", "days_to_first_harvest"])
    b = b.assign(bucket=pd.cut(b.days_to_first_harvest, [0, 28, 35, 42, 50, 200], labels=["≤28 days", "29-35", "36-42", "43-50", ">50 days"]))
    t = b.groupby("bucket", observed=True).nps_score.agg(["mean", "size"]).reset_index().rename(columns={"bucket": "label", "mean": "value", "size": "n"})
    r, pv = stats.spearmanr(b.days_to_first_harvest, b.nps_score)
    prom, det = (b.nps_score >= 9).mean(), (b.nps_score <= 6).mean()
    return result("q6", "Pipeline - diagnostic", "Customer satisfaction (NPS score) by days to first harvest",
                  "Does a quick first harvest make happier customers?", "vbar", t,
                  shows=f"Average 0-10 recommendation score for {len(b):,} buyers, grouped by how long their first harvest took. Spearman r = "
                        f"{r:+.2f} ({p_text(pv)}). Net Promoter Score = {prom * 100:.0f}% promoters − {det * 100:.0f}% detractors = "
                        f"{(prom - det) * 100:+.0f}.",
                  why="Satisfaction falls steadily the longer customers wait for something to eat. Promoters create the referrals that "
                      "proved to be the cheapest channel, so slow harvests also raise future acquisition cost.",
                  so_what="Faster first harvest improves three numbers at once - churn, NPS and referrals. It is the single most valuable "
                          "operational fix found in the pilot.",
                  unit="average score (0-10)")


# ================================================================ 5. END-TO-END VALIDATION
def e1_price_validation(s, p):
    sv = s[s.price_shown.isin([4999, 5999])].groupby("price_shown").would_buy.mean()
    pv_ = p[p.demo_done == 1].groupby("price_offered").purchased.mean()
    rows = [dict(series="Survey: would buy (stated)", x=k, value=v / sv[4999] * 100) for k, v in sv.items()]
    rows += [dict(series="Pilot: bought after demo (actual)", x=k, value=v / pv_[4999] * 100) for k, v in pv_.items()]
    t = pd.DataFrame(rows)
    drop_s, drop_p = 1 - sv[5999] / sv[4999], 1 - pv_[5999] / pv_[4999]
    return result("e1", "End-to-end validation", "Price sensitivity: survey vs pilot (₹4,999 = 100)",
                  "Did the survey predict how buyers react to price?", "line", t, x="x", series="series", refs_y=[(100, "")],
                  shows=f"Both lines are indexed so ₹4,999 = 100, which lets a stated rate and a real rate share one scale. Raising the price "
                        f"to ₹5,999 cut stated intent by {pct(drop_s)} in the survey and actual purchases by {pct(drop_p)} in the pilot.",
                  why="The survey captured the direction and the rough size of price sensitivity. "
                      + ("Real buyers were even more price-sensitive than the survey suggested - spending real money sharpens attention to price."
                         if drop_p > drop_s else "Real buyers were slightly less price-sensitive than stated - a demo builds value that a survey cannot."),
                  so_what="The survey is a reliable guide to price direction, so it can be used to screen future price changes before a field "
                          "test. The launch price should stay at ₹4,999.",
                  unit="index (₹4,999 = 100)", xlab="Kit price (₹)")


def e2_stated_vs_actual(s, p):
    stated_buy = s[s.price_shown.between(4999, 5999)].would_buy.mean()
    actual_buy = p.purchased.mean()
    stated_sub = s[(s.refill_price_shown == REFILL_PRICE) & (s.would_buy == 1)].would_subscribe.mean()
    actual_sub = p[p.purchased == 1].refill_subscribed.mean()
    tried = s[s.experience_level >= 1]
    stated_keep = (tried.kept_up_6m == "Kept up").mean()
    sub = p[p.refill_subscribed == 1]
    actual_keep = km_curve(sub)[6]
    rows = [("Buys the kit", "% of survey (₹4,999-5,999) / % of pilot leads", stated_buy, actual_buy),
            ("Subscribes to refills", "% of likely buyers / % of actual buyers, at ₹299", stated_sub, actual_sub),
            ("Still growing after 6 months", "% of triers / % of subscribers", stated_keep, actual_keep)]
    t = pd.DataFrame([dict(label=a, series=sr, value=v * 100) for a, _, sv_, av in rows for sr, v in (("Survey said", sv_), ("Pilot showed", av))])
    tab = pd.DataFrame(rows, columns=["metric", "base", "survey", "pilot"]).assign(ratio=lambda x: x.pilot / x.survey)
    return result("e2", "End-to-end validation", "What customers said vs what they did",
                  "How far can survey answers be trusted for the business plan?", "grouped_hbar", t, table=tab,
                  series_order=["Survey said", "Pilot showed"],
                  shows=f"Three funnel stages measured twice. Buying: {pct(stated_buy)} said likely vs {pct(actual_buy, 1)} of pilot leads bought "
                        f"(ratio {actual_buy / stated_buy:.2f}). Subscribing: {pct(stated_sub)} vs {pct(actual_sub)} (ratio {actual_sub / stated_sub:.2f}). "
                        f"Six-month survival: {pct(stated_keep)} vs {pct(actual_keep)}.",
                  why="A 'likely' in a survey costs nothing, so stated intent overshoots: only about half as many leads bought as survey "
                      f"respondents said they would. Subscription was compared like for like (likely buyers vs actual buyers) and came in at "
                      f"{actual_sub / stated_sub:.2f}x stated - close to the 0.70 realism factor used in the earlier business plan. "
                      + ("Six-month survival beat past gardening behaviour, so onboarding and refills appear to help people keep going."
                         if actual_keep > stated_keep else "Six-month survival was no better than past gardening behaviour."),
                  so_what="Use these ratios as correction factors whenever the survey feeds a forecast: subscription intent × "
                          f"{actual_sub / stated_sub:.2f}, and never treat top-2-box intent as a purchase rate.",
                  unit="%")


def e3_channel_ltv(p):
    kit = unit_economics(p)
    b = p[p.purchased == 1].assign(kit=kit)
    churn_all = churn_rate(p[p.refill_subscribed == 1])
    life = 1 / churn_all if churn_all and churn_all > 0 else 12
    rows = []
    for ch in CHANNELS:
        d = p[p.channel == ch]; bb = b[b.channel == ch]
        if bb.empty:
            continue
        cac = d.acquisition_cost_inr.sum() / len(bb)
        ltv = bb.kit.mean() + bb.refill_subscribed.mean() * REFILL_MARGIN * min(life, 24)
        rows.append(dict(label=ch, value=ltv / cac, ltv=ltv, cac=cac, buyers=len(bb)))
    t = pd.DataFrame(rows).sort_values("value")
    good = t[t.value >= 2.5].sort_values("value", ascending=False).label.tolist(); bad = t[t.value < 1].label.tolist()
    rank = t.sort_values("value", ascending=False)
    return result("e3", "End-to-end validation", "Lifetime value ÷ acquisition cost by channel",
                  "Which channels should the sales pipeline be built on?", "hbar", t, refs=[(1, "break-even (1x)"), (3, "healthy (3x)")],
                  shows=f"Customer lifetime value (kit margin + expected refill margin, using the pilot's {pct(churn_all, 1)} monthly churn and "
                        f"₹{REFILL_MARGIN} margin per box, capped at 24 months) divided by CAC. Above 3x is healthy; below 1x loses money.",
                  why="Lifetime value per buyer is similar across channels (" + ", ".join(f"{inr(v)}" for v in [t.ltv.min(), t.ltv.max()]).replace(", ", " to ")
                      + "), so the ranking is driven by CAC: " + "; ".join(f"{r.label} {r.value:.1f}x" for r in rank.itertuples())
                      + ". Paid search and social buy attention, not intent." + (f" The smallest channel has only {int(t.buyers.min())} buyers, so its ratio is uncertain." if t.buyers.min() < 30 else ""),
                  so_what=("Build the pipeline on " + " and ".join(good) + ". " if good else "")
                          + ("Cut or redesign " + ", ".join(bad) + "." if bad else "No channel is loss-making over the customer's life, but the weaker ones should be capped."),
                  unit="LTV ÷ CAC (x)", table=t)


def pipeline_plan(s, p):
    """Final sales-pipeline table: current pilot rate per stage, a target, and the lever found in the analysis."""
    d = p
    contact = d.contacted.mean(); demo = d.demo_done.sum() / max(d.contacted.sum(), 1)
    buy = d.purchased.sum() / max(d.demo_done.sum(), 1); sub = d[d.purchased == 1].refill_subscribed.mean()
    churn = churn_rate(d[d.refill_subscribed == 1])
    fast = q2_response(p)["data"]; fast_rate = fast.rate.iloc[0] if len(fast) else np.nan
    return pd.DataFrame([
        ("1. Lead capture", "Leads / month", f"{len(d) / 6:,.0f}", "Shift mix to society demos + referrals", "Channel CAC (P4), LTV:CAC (E3)"),
        ("2. First contact", "Contacted", pct(contact), "90%, called within 2 hours", f"Purchase rate {pct(fast_rate, 1)} when called < 2 h (Q2)"),
        ("3. Demo", "Contacted → demo", pct(demo), "In-person where possible", "In-person converts better (Q3)"),
        ("4. Close", "Demo → purchase", pct(buy), "Hold ₹4,999; coach weaker reps", "Price A/B test (Q1), rep spread (Q3)"),
        ("5. Subscribe", "Buyer → refill", pct(sub), "Offer at ₹299 with first box free", "Survey ratio (E2), time-poor segment (G7)"),
        ("6. Retain", "Monthly churn", pct(churn, 1), "Below 6% via first harvest ≤35 days", "Harvest speed + onboarding (Q5, Q6)"),
    ], columns=["Stage", "Measure", "Pilot result", "Target / action", "Evidence"])


SURVEY_CHARTS = [d1_profile, d2_intent, d3_wtp, d4_features, d5_worries, d6_kept_up, g1_corr, g2_drivers, g3_price, g4_wtp_income,
                 g5_space, g6_wtp_spend, g7_subscribe, g8_feature_lift]
PIPELINE_CHARTS = [p1_funnel, p2_volume, p3_channel_conv, p4_cac, p5_basket, q1_price, q2_response, q3_rep, q4_drivers, q5_retention, q6_nps]
E2E_CHARTS = [e1_price_validation, e2_stated_vs_actual, e3_channel_ltv]


def load(base="."):
    s = pd.read_csv(f"{base}/data/clean/survey_clean.csv")
    p = pd.read_csv(f"{base}/data/clean/pipeline_clean.csv", parse_dates=["lead_created", "purchase_date"])
    s["space_label"] = pd.Categorical(s.space_label, SPACE_ORDER, ordered=True)
    s["income_label"] = pd.Categorical(s.income_label, INCOME_ORDER, ordered=True)
    p["space_label"] = pd.Categorical(p.space_label, SPACE_ORDER, ordered=True)
    return s, p

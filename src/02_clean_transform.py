# %% [md]
# # Step 2 – Data cleaning and transformation
# **Urban home-farming start-up (India) · synthetic data**
#
# This notebook takes the two raw exports made by `01_generate_data.py` and turns them into analysis-ready tables.
#
# | Raw file | What it is | Rows |
# |---|---|---|
# | `data/raw/survey_raw.csv` | Consumer survey, exported from a form tool (question text as headers, text answers) | 3,045 |
# | `data/raw/pipeline_raw.csv` | CRM export of a 6-month pilot in Bengaluru and Pune (lead → demo → purchase → refill → retention) | 4,326 |
#
# Every step follows the same pattern: **find the problem → count it → fix it → prove it is fixed.**
# The generator also wrote an answer key of the problems it planted (`planted_noise_log.csv`,
# `survey_answer_key_planted_issues.csv`). It is used only at the end, to score how well the cleaning rules worked.
#
# **Cleaning principles used throughout**
# 1. Never invent an answer the respondent did not give. Missing prices stay missing and are flagged.
# 2. Fix what can be recovered (typed amounts, spellings, date formats, a purchase date that can be rebuilt).
# 3. Remove a whole response only when it cannot be trusted (duplicate, speeder, impossible or contradictory answers).
# 4. Keep a log of every change so each number in the report can be traced back.

# %%
import re
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

LOG = []                                    # one row per cleaning action -> data/clean/cleaning_log.csv
def log(dataset, step, issue, rows, action):
    LOG.append(dict(dataset=dataset, step=step, issue=issue, rows_affected=int(rows), action=action))

survey_raw = pd.read_csv("data/raw/survey_raw.csv")
pipe_raw = pd.read_csv("data/raw/pipeline_raw.csv")
print("survey  :", survey_raw.shape)
print("pipeline:", pipe_raw.shape)

# %% [md]
# ## Part A – Survey data
# ### A1. First look: structure, types and missing values
# The form tool exports question text as column names and almost everything as text. Before changing anything,
# record the starting state so the before/after comparison at the end is honest.

# %%
survey_raw.head(3).T

# %%
def profile(df):
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(), "missing": df.isna().sum(),
                         "missing_%": (df.isna().mean() * 100).round(1), "unique": df.nunique()})
profile(survey_raw)

# %% [md]
# **Reading the profile.** Amount questions (Q7, Q16a, Q16b) and household size are stored as text (`object`),
# because people typed values like *"6k"*, *"Rs 6,000"* and *"4 members"*. The multi-select questions (Q12–Q14)
# are blank when nothing was ticked, which is a valid answer ("no worries"), not missing data.
# Real gaps are in Q7 spend and the two price questions (Q16a/Q16b).

# %% [md]
# ### A2. Rename columns to short, code-friendly names

# %%
RENAME = {
    "Timestamp": "submitted", "Response ID": "response_id", "Device ID": "device_id", "Time taken (min)": "minutes_taken",
    "Q1 City": "city_raw", "Q2 Monthly household income": "income_raw", "Q3 Household size": "household_raw",
    "Q4 Children under 15": "children", "Q5 Growing space": "space", "Q6 Direct sunlight": "sunlight",
    "Q7 Monthly veg & herb spend (Rs)": "veg_spend_raw", "Q8 Greens spoil before use": "spoilage",
    "Q9a Prefer pesticide-free produce (1-5)": "att_health", "Q9b Reduce environmental impact (1-5)": "att_eco",
    "Q9c Comfortable with apps/smart devices (1-5)": "att_tech", "Q9d Little time for chores (1-5)": "att_time_poor",
    "Q9e Would enjoy plants as a hobby (1-5)": "att_hobby", "Q10 Gardening experience": "experience",
    "Q11 Kept up growing 6+ months": "kept_up_6m", "Q12 Features wanted": "features", "Q13 Would like to grow": "crops",
    "Q14 Biggest worries": "worries", "Q15 Price shown (Rs)": "price_shown", "Q15 Likelihood to buy": "intent",
    "Q16a Max price for starter system (Rs)": "wtp_system_raw", "Q16b Max price for monthly refill (Rs)": "wtp_refill_raw",
    "Q17 Refill price shown (Rs)": "refill_price_shown", "Q17 Would subscribe": "subscribe"}
s = survey_raw.rename(columns=RENAME).copy()
assert set(RENAME.values()) == set(s.columns)
s["submitted"] = pd.to_datetime(s.submitted, format="%d/%m/%Y %H:%M:%S")
print(s.columns.tolist())

# %% [md]
# ### A3. Remove duplicate submissions
# The same device submitting twice is the classic duplicate in online surveys. Keep the **first** submission
# and drop later ones.

# %%
dup_mask = s.sort_values("submitted").duplicated("device_id", keep="first").reindex(s.index)
print("Responses from a device that had already answered:", dup_mask.sum())
s.loc[dup_mask, ["response_id", "device_id", "submitted", "minutes_taken", "city_raw", "intent"]].head()

# %%
removed = {"Duplicate submission (same device)": s.loc[dup_mask, "response_id"].tolist()}
log("survey", "A3", "Duplicate submission (same device)", dup_mask.sum(), "Dropped later copy, kept first")
s = s.loc[~dup_mask].copy()
print("rows now:", len(s))

# %% [md]
# ### A4. Standardise city names and derive city tier
# Q1 was a free-text box, so one city appears under several spellings (*Bangalore, BLR, bangalore␣*).
# A lookup table maps every spelling to one canonical name and to its tier (1 = metro, 2 = large city, 3 = smaller city).

# %%
print("distinct raw spellings:", s.city_raw.nunique())
s.city_raw.value_counts().head(25)

# %%
TIER = {**{c: 1 for c in ["Mumbai", "Delhi NCR", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune"]},
        **{c: 2 for c in ["Jaipur", "Lucknow", "Indore", "Coimbatore", "Kochi", "Chandigarh", "Nagpur", "Surat"]},
        **{c: 3 for c in ["Mysuru", "Nashik", "Udaipur", "Dehradun", "Guntur"]}}
ALIAS = {"bombay": "Mumbai", "delhi": "Delhi NCR", "new delhi": "Delhi NCR", "gurgaon": "Delhi NCR", "noida": "Delhi NCR",
         "bangalore": "Bengaluru", "blr": "Bengaluru", "madras": "Chennai", "hyd": "Hyderabad", "calcutta": "Kolkata", "poona": "Pune"}
CANON = {c.lower(): c for c in TIER} | ALIAS

def clean_city(x):
    return CANON.get(str(x).strip().lower(), np.nan)

s["city"] = s.city_raw.map(clean_city)
s["city_tier"] = s.city.map(TIER)
changed = (s.city != s.city_raw).sum()
log("survey", "A4", "City spelt in several ways", changed, "Mapped to canonical name + tier")
print("unmapped cities:", s.city.isna().sum(), "| spellings fixed:", changed, "| distinct after:", s.city.nunique())
s.groupby("city_tier").city.agg(["nunique", "size"])

# %% [md]
# ### A5. Convert typed numbers to numbers
# * **Household size**: mostly digits, but some typed *"4 members"* or *"four"*.
# * **Rupee amounts** (Q7, Q16a, Q16b): typed as `6000`, `Rs 6,000`, `6k`, `6000/-` or `₹6,000`.
#
# One parser handles every format; anything it cannot read becomes missing (and is counted).

# %%
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
def to_number(x):
    if pd.isna(x) or str(x).strip() == "":
        return np.nan
    t = str(x).strip().lower()
    if t in WORDS:
        return float(WORDS[t])
    t = t.replace("members", "").replace("₹", "").replace("rs", "").replace(",", "").replace("/-", "").strip()
    mult = 1000 if t.endswith("k") else 1
    t = t.rstrip("k").strip()
    try:
        return float(t) * mult
    except ValueError:
        return np.nan

for raw_col, new_col in [("household_raw", "household_size"), ("veg_spend_raw", "veg_spend"),
                         ("wtp_system_raw", "wtp_system"), ("wtp_refill_raw", "wtp_refill")]:
    is_text = s[raw_col].notna() & ~s[raw_col].astype(str).str.fullmatch(r"\d+(\.\d+)?")
    s[new_col] = s[raw_col].map(to_number)
    failed = s[raw_col].notna() & s[new_col].isna()
    log("survey", "A5", f"{raw_col}: typed as text", is_text.sum(), "Parsed to number")
    print(f"{raw_col:16s} text-formatted: {is_text.sum():4d} | could not parse: {failed.sum()}")

s.loc[s.wtp_system_raw.astype(str).str.contains(r"[kK]|Rs|₹|/-", na=False), ["wtp_system_raw", "wtp_system"]].drop_duplicates().head(8)

# %% [md]
# ### A6. Encode ordered answers as numbers
# Ordered text answers (income band, space, sunlight, spoilage, experience, intent, subscription) are mapped to
# ordinal scores so they can be averaged and correlated. *Prefer not to say* becomes missing, not a fake score,
# and is kept as an explicit category for charts.

# %%
ORD = {
    "income": ({"Below Rs 50,000": 1, "Rs 50,000-1 lakh": 2, "Rs 1-2 lakh": 3, "Rs 2-3.5 lakh": 4, "Above Rs 3.5 lakh": 5}, "income_raw"),
    "space_level": ({"No balcony (window sill)": 0, "Small balcony (<40 sq ft)": 1, "Large balcony (40-100 sq ft)": 2, "Terrace or garden (>100 sq ft)": 3}, "space"),
    "sun_level": ({"<2 hours": 1, "2-4 hours": 2, "4-6 hours": 3, ">6 hours": 4}, "sunlight"),
    "spoil_level": ({"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}, "spoilage"),
    "experience_level": ({"No experience": 0, "Kept a few plants": 1, "Grown vegetables before": 2, "Grow vegetables now": 3}, "experience"),
    "intent_score": ({"Very unlikely": 1, "Unlikely": 2, "Not sure": 3, "Likely": 4, "Very likely": 5}, "intent"),
    "subscribe_score": ({"Definitely not": 1, "Probably not": 2, "Not sure": 3, "Probably yes": 4, "Definitely yes": 5}, "subscribe"),
}
for new, (mapping, src) in ORD.items():
    s[new] = s[src].map(mapping)
    unknown = s[src].notna() & s[new].isna() & (s[src] != "Prefer not to say")
    assert unknown.sum() == 0, f"unmapped labels in {src}"
s["income_disclosed"] = (s.income_raw != "Prefer not to say").astype(int)
s["children"] = s.children.map({"Yes": 1, "No": 0})
s["kept_up_6m"] = s.kept_up_6m.map({"Never tried": "Never tried", "Yes, kept it up 6+ months": "Kept up", "No, stopped within 6 months": "Stopped"})
log("survey", "A6", "Income 'Prefer not to say'", (s.income_disclosed == 0).sum(), "Set to missing + flag; shown as 'Not disclosed'")
s[["income_raw", "income", "space", "space_level", "intent", "intent_score"]].drop_duplicates().head(6)

# %% [md]
# ### A7. Split multi-select answers into 0/1 columns (one-hot)
# Q12–Q14 store ticked options in one cell separated by `;`. Each option becomes its own 0/1 column, which is the
# format correlation, market-basket and regression analysis need.

# %%
OPTIONS = {
    "features": {"Vertical / stackable design": "f_vertical", "Self-watering reservoir": "f_selfwater", "Hydroponic (no soil)": "f_hydroponic",
                 "Soil or cocopeat based": "f_soil", "Smart sensor with app alerts": "f_smart", "Organic seeds and nutrients": "f_organic",
                 "Monthly refill subscription": "f_subscription", "Installation and maintenance visits": "f_install", "Grow light for indoor use": "f_growlight"},
    "crops": {"Herbs (coriander mint tulsi)": "g_herbs", "Leafy greens (spinach methi lettuce)": "g_leafy", "Microgreens": "g_micro",
              "Tomatoes and chillies": "g_tomato", "Exotic greens (basil kale cherry tomato)": "g_exotic"},
    "worries": {"Price": "w_price", "Lack of space": "w_space", "Time and upkeep": "w_time", "Plants dying": "w_dying",
                "Pests or mess": "w_pests", "Water leakage or society rules": "w_leak"}}
for col, opts in OPTIONS.items():
    ticked = s[col].fillna("").str.split(r";\s*")
    for label, short in opts.items():
        s[short] = ticked.map(lambda xs: int(label in xs))
    s[f"n_{col}"] = s[list(opts.values())].sum(axis=1)
    unknown = set(o for xs in ticked for o in xs if o) - set(opts)
    print(f"{col:9s} options: {len(opts)} | unknown labels: {unknown or 'none'} | mean ticked: {s[f'n_{col}'].mean():.2f}")
s[["features", "f_vertical", "f_selfwater", "f_smart", "n_features"]].head(3)

# %% [md]
# ### A8. Quality checks – which responses cannot be trusted?
# Rules are set *before* looking at results, and each rule is logical, not statistical:
#
# | Rule | Why it signals a bad response |
# |---|---|
# | Finished in under 2 minutes | Reading 17 questions takes at least ~2.5 minutes; faster means not reading |
# | Same answer on all 7 rating questions **and** under 2 minutes | Straight-lining: clicking the same column without reading |
# | Household size 0 or above 15 | Physically implausible for an urban flat |
# | Spends ₹0 on vegetables with 3+ people at home | Contradicts itself |
# | Max price under ₹200 for a starter kit, yet "Very likely" to buy at the shown price | Contradicts itself |

# %%
RATING = ["att_health", "att_eco", "att_tech", "att_time_poor", "att_hobby", "intent_score", "subscribe_score"]
rules = {
    "Speeder (under 2 minutes)": s.minutes_taken < 2,
    "Straight-liner and speeder": (s[RATING].nunique(axis=1) == 1) & (s.minutes_taken < 2),
    "Impossible household size": (s.household_size < 1) | (s.household_size > 15),
    "Rs 0 veg spend with 3+ people": (s.veg_spend == 0) & (s.household_size >= 3),
    "Max price under Rs 200 but 'Very likely'": (s.wtp_system < 200) & (s.intent_score == 5),
}
flags = pd.DataFrame(rules)
print(flags.sum().to_string())
bad = flags.any(axis=1)
print("\nresponses failing at least one rule:", bad.sum())
print("straight-lining without speeding (kept, only flagged):", ((s[RATING].nunique(axis=1) == 1) & (s.minutes_taken >= 2)).sum())

# %%
for rule, m in rules.items():
    removed.setdefault(rule, []).extend(s.loc[m & ~s.response_id.isin(sum(removed.values(), [])), "response_id"])
    log("survey", "A8", rule, m.sum(), "Response dropped")
s = s.loc[~bad].copy()
print("rows now:", len(s))

# %% [md]
# ### A9. Outliers in amounts – 3×IQR fences
# For the typed rupee amounts a statistical fence is used: values more than **3 × IQR** beyond the quartiles are
# "extreme" (the usual 1.5× fence would cut genuine high earners and big families). Flagged values are set to
# missing **for that question only** – the rest of the response is kept, because a typo in one box does not make
# the other answers wrong.

# %%
def iqr_fence(x, k=3.0):
    v = x
    q1, q3 = v.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out = (v < lo) | (v > hi)
    return out.fillna(False), (max(lo, 0), hi)

pre_fence = s[["veg_spend", "wtp_system", "wtp_refill"]].copy()
for col in ["veg_spend", "wtp_system", "wtp_refill"]:
    out, (lo, hi) = iqr_fence(s[col])
    print(f"{col:11s} fence ₹{lo:,.0f} – ₹{hi:,.0f} | outside: {out.sum():3d} | values: {sorted(s.loc[out, col].astype(int).tolist())[:12]}")
    s[f"{col}_outlier"] = out.astype(int)
    s.loc[out, col] = np.nan
    log("survey", "A9", f"{col} outside 3xIQR fence", out.sum(), "Value set to missing, response kept")

# %% [md]
# **Reading the fences.** The veg-spend fence removes the ₹65,000–₹1,50,000 entries (a month's vegetables for a
# family is a few thousand rupees). The starter-kit fence catches the *extra zero* typos (₹40,000–₹70,000 for a
# kit priced around ₹5,000). The lower fences fall below zero, so they are shown as ₹0. A max price of **₹0 is kept**:
# "I would not pay anything for this" is a real answer, not an error.

# %% [md]
# ### A10. Missing values – what is filled and what is left alone
# | Field | Missing | Treatment | Reason |
# |---|---|---|---|
# | Monthly veg spend | blank + outliers | **Median of the same household size and income band** | Spend depends mainly on family size and income, so a group median is a fair estimate; a flag column records it |
# | Income | *Prefer not to say* | Left missing, shown as "Not disclosed" | Guessing someone's income would bias the income analysis |
# | Max price for kit / refill | blank + outliers | **Left missing** | A price answer is an opinion; inventing it would distort the demand curve |

# %%
before = s[["veg_spend", "wtp_system", "wtp_refill", "income"]].isna().sum()
s["veg_spend_imputed"] = s.veg_spend.isna().astype(int)
grp_median = s.groupby([s.household_size.clip(upper=7), s.income.fillna(0)]).veg_spend.transform("median")
s["veg_spend"] = s.veg_spend.fillna(grp_median).fillna(s.groupby(s.household_size.clip(upper=7)).veg_spend.transform("median"))
log("survey", "A10", "Veg spend missing (blank or outlier)", s.veg_spend_imputed.sum(), "Filled with household-size x income median")
log("survey", "A10", "Max kit price missing", s.wtp_system.isna().sum(), "Left missing (not imputed)")
log("survey", "A10", "Max refill price missing", s.wtp_refill.isna().sum(), "Left missing (not imputed)")
pd.DataFrame({"missing_before": before, "missing_after": s[["veg_spend", "wtp_system", "wtp_refill", "income"]].isna().sum()})

# %% [md]
# ### A11. Feature engineering (transformation)
# New columns that turn raw answers into business measures:
#
# | New column | Formula | Business meaning |
# |---|---|---|
# | `income_mid_inr` | midpoint of the income band | Lets income be used as a number |
# | `spend_per_person` | veg spend ÷ household size | Intensity of fresh-produce use |
# | `would_buy` | intent = Likely or Very likely (top-2 box) | Standard purchase-intent measure |
# | `would_subscribe` | subscribe = Probably/Definitely yes | Refill subscription intent |
# | `wtp_gap` | max price − price shown | Positive = the shown price is affordable to them |
# | `can_afford_shown` | max price ≥ price shown | Whether the test price is within their budget |
# | `refill_affordable` | max refill price ≥ refill price shown | Same for the refill |
# | `green_index` | mean of health and eco attitudes | Strength of the "pesticide-free, sustainable" motive |
# | `novice`, `tried_and_quit` | experience = none; stopped within 6 months | Retention-risk markers |
# | `price_band`, `space_label`, `income_label` | ordered categories | Clean labels for charts |

# %%
INCOME_MID = {1: 35000, 2: 75000, 3: 150000, 4: 275000, 5: 450000}
s["income_mid_inr"] = s.income.map(INCOME_MID)
s["income_label"] = pd.Categorical(s.income_raw.replace({"Prefer not to say": "Not disclosed"}),
    ["Below Rs 50,000", "Rs 50,000-1 lakh", "Rs 1-2 lakh", "Rs 2-3.5 lakh", "Above Rs 3.5 lakh", "Not disclosed"], ordered=True)
s["space_label"] = pd.Categorical(s.space.str.replace(r" \(.*\)", "", regex=True),
    ["No balcony", "Small balcony", "Large balcony", "Terrace or garden"], ordered=True)
s["spend_per_person"] = (s.veg_spend / s.household_size).round(0)
s["would_buy"] = (s.intent_score >= 4).astype(int)
s["would_subscribe"] = (s.subscribe_score >= 4).astype(int)
s["wtp_gap"] = s.wtp_system - s.price_shown
s["can_afford_shown"] = np.where(s.wtp_system.isna(), np.nan, (s.wtp_system >= s.price_shown).astype(float))
s["refill_affordable"] = np.where(s.wtp_refill.isna(), np.nan, (s.wtp_refill >= s.refill_price_shown).astype(float))
s["green_index"] = s[["att_health", "att_eco"]].mean(axis=1)
s["novice"] = (s.experience_level == 0).astype(int)
s["tried_and_quit"] = (s.kept_up_6m == "Stopped").astype(int)
s["price_band"] = pd.Categorical("₹" + s.price_shown.map("{:,}".format), [f"₹{p:,}" for p in [2999, 3999, 4999, 5999, 7499, 9999]], ordered=True)
s["submitted_date"] = s.submitted.dt.date
s["submitted_hour"] = s.submitted.dt.hour

skew = pd.DataFrame({"skew with outliers": pre_fence.skew(), "skew after 3xIQR fence": s[["veg_spend", "wtp_system", "wtp_refill"]].skew()}).round(2)
skew.index = ["monthly veg spend", "max kit price", "max refill price"]
skew

# %% [md]
# **Is a log transform needed?** Skew above about 1 means a long tail that would distort means and correlations.
# With the typos still in, max kit price and veg spend are heavily skewed. After the 3×IQR fence both are close to
# symmetric (|skew| < 0.5), so the amounts can be analysed in plain rupees and **no log transform is applied** –
# that keeps every chart in units a manager reads directly. (A log would now over-correct and create a left skew.)

# %% [md]
# ### A12. Score the cleaning against the answer key
# The generator logged which responses it deliberately broke. Comparing the rows removed (A3, A8) and the values
# blanked (A9) with that key shows whether the rules were too loose or too strict.

# %%
key = pd.read_csv("data/raw/survey_answer_key_planted_issues.csv")
dropped_ids = set(sum(removed.values(), []))
blanked_ids = set(s.loc[s[["veg_spend_outlier", "wtp_system_outlier", "wtp_refill_outlier"]].any(axis=1), "response_id"])
caught = dropped_ids | blanked_ids
key["caught"] = key.response_id.isin(caught)
score = key.groupby("planted_issue").caught.agg(planted="size", caught="sum")
score.loc["TOTAL"] = score.sum()
false_alarms = caught - set(key.response_id)
print(score.to_string())
print(f"\nRecall: {key.caught.mean():.1%} | Rows flagged that were not planted (false alarms): {len(false_alarms)}")

# %% [md]
# **Recall** = share of planted problems the rules caught. **False alarms** = responses touched by a rule that the
# generator did not break; for those only one typed amount was blanked, the response itself was kept.

# %% [md]
# ## Part B – Pilot sales-pipeline (CRM) data
# ### B1. First look

# %%
pipe_raw.head(3).T

# %%
profile(pipe_raw)

# %% [md]
# **Reading the profile.** Sales staff typed the CRM by hand, so the problems are different from the survey:
# dates in three formats, Yes/No written as *Y, 1, TRUE, yes*, channel and city spelt several ways, order values
# typed as text, and placeholder numbers (999 hours). Columns such as `purchase_date` or `nps_score` are blank
# for leads that never bought – that is structural, not missing.

# %%
p = pipe_raw.copy()

# %% [md]
# ### B2. Remove duplicate leads
# The same phone number entered twice (two reps, or one lead coming through two ads). Keep the earliest lead ID.

# %%
p = p.sort_values("lead_id")
dupl = p.duplicated("phone_hash", keep="first")
print("duplicate leads:", dupl.sum())
log("pipeline", "B2", "Duplicate lead (same phone)", dupl.sum(), "Dropped later lead ID")
p = p.loc[~dupl].copy()

# %% [md]
# ### B3. Parse dates written in three formats
# `2027-04-21 14:50`, `04/04/2027` (day first, Indian style) and `Apr 21 2027`. Each format is tried explicitly –
# letting pandas guess would read `04/05/2027` as 5 April in some rows and 4 May in others.

# %%
def parse_mixed(x):
    for fmt in ("%Y-%m-%d %H:%M", "%d/%m/%Y", "%b %d %Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(x, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

fmt_counts = p.lead_created.astype(str).map(lambda x: "yyyy-mm-dd hh:mm" if re.match(r"\d{4}-", x) else "dd/mm/yyyy" if "/" in x else "Mon dd yyyy").value_counts()
print(fmt_counts.to_string())
p["lead_created"] = p.lead_created.map(parse_mixed)
p["purchase_date"] = pd.to_datetime(p.purchase_date, format="%Y-%m-%d")
print("unparsed lead dates:", p.lead_created.isna().sum())
log("pipeline", "B3", "Lead date not in yyyy-mm-dd", fmt_counts.drop("yyyy-mm-dd hh:mm").sum(), "Parsed with explicit formats")

# %% [md]
# ### B4. Standardise text categories and Yes/No flags

# %%
CITY_P = {"bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru", "pune": "Pune", "poona": "Pune"}
CHANNEL = {"instagram ads": "Instagram ads", "ig ads": "Instagram ads", "instagram": "Instagram ads",
           "google search": "Google search", "google": "Google search",
           "housing-society demo": "Housing-society demo", "society demo": "Housing-society demo", "housing society demo": "Housing-society demo",
           "referral": "Referral", "referral program": "Referral",
           "corporate wellness": "Corporate wellness", "corporate": "Corporate wellness",
           "website (organic)": "Website (organic)", "website": "Website (organic)", "website/organic": "Website (organic)"}
before_city, before_ch = p.city.nunique(), p.channel.nunique()
p["city"] = p.city.str.strip().str.lower().map(CITY_P)
p["channel"] = p.channel.str.strip().str.lower().map(CHANNEL).fillna("Unknown")
print(f"city spellings {before_city} -> {p.city.nunique()} | channel spellings {before_ch} -> {p.channel.nunique()} (incl. 'Unknown')")
log("pipeline", "B4", "City spelt in several ways", (pipe_raw.city.str.strip().isin(["Bengaluru", "Pune"]) == False).sum(), "Mapped to canonical")
log("pipeline", "B4", "Channel missing", (p.channel == "Unknown").sum(), "Labelled 'Unknown' (kept)")

TRUE_SET, FALSE_SET = {"yes", "y", "1", "true"}, {"no", "n", "0", "false"}
def to_bool(x):
    if pd.isna(x) or str(x).strip() == "":
        return np.nan
    t = str(x).strip().lower()
    return 1.0 if t in TRUE_SET else 0.0 if t in FALSE_SET else np.nan

BOOL_COLS = ["contacted", "demo_done", "purchased", "expansion_tier", "smart_sensor", "discount_used", "onboarding_call", "refill_subscribed", "churned"]
for c in BOOL_COLS:
    variants = p[c].dropna().astype(str).unique()
    p[c] = p[c].map(to_bool)
    if len(variants) > 2:
        print(f"{c:18s} variants {sorted(variants)} -> 0/1")
log("pipeline", "B4", "Yes/No written as Y, 1, TRUE, yes", sum((~pipe_raw[c].isin(["Yes", "No"]) & pipe_raw[c].notna()).sum() for c in BOOL_COLS), "Converted to 0/1")
p["income_band"] = p.income_band.fillna("Not disclosed")
log("pipeline", "B4", "Income not given", (p.income_band == "Not disclosed").sum(), "Labelled 'Not disclosed' (kept)")

# %% [md]
# ### B5. Order value: text to number, and the extra-zero typo
# The highest legitimate basket is ₹5,999 + ₹1,999 expansion + ₹1,499 sensor = **₹9,497**. Anything above that
# cannot be a real order. Values that become legitimate when divided by 10 are fixed; the fix is logged.

# %%
p["order_value_inr"] = p.order_value_inr.map(to_number)
MAX_BASKET = 5999 + 1999 + 1499
too_big = p.order_value_inr > MAX_BASKET
print("orders above the maximum basket:", too_big.sum(), p.loc[too_big, "order_value_inr"].tolist())
p.loc[too_big, "order_value_inr"] = p.loc[too_big, "order_value_inr"] / 10
p["order_value_fixed"] = too_big.astype(int)
log("pipeline", "B5", "Order value typed as text", (pipe_raw.order_value_inr.notna() & ~pipe_raw.order_value_inr.astype(str).str.fullmatch(r"\d+(\.0)?")).sum(), "Parsed to number")
log("pipeline", "B5", "Order value with extra zero", too_big.sum(), "Divided by 10")

# %% [md]
# ### B6. Impossible values and placeholders
# | Problem | Rule | Fix |
# |---|---|---|
# | Purchase dated before the lead existed | `purchase_date < lead_created` | **Rebuilt** as lead date + days-to-close (both fields are trusted) |
# | NPS of 11, 99 or −1 | NPS must be 0–10 | Set to missing |
# | 999 hours to first contact | system placeholder for "not recorded" | Set to missing, then filled with the channel median |
# | Acquisition cost blank | — | Filled with the channel median (cost is set by the channel) |
# | Negative months active | must be ≥ 0 | Set to missing |

# %%
bad_date = p.purchase_date < p.lead_created.dt.normalize()
p.loc[bad_date, "purchase_date"] = (p.loc[bad_date, "lead_created"] + pd.to_timedelta(p.loc[bad_date, "days_to_close"], unit="D")).dt.normalize()
bad_nps = ~p.nps_score.between(0, 10) & p.nps_score.notna()
p.loc[bad_nps, "nps_score"] = np.nan
ph = p.hours_to_first_contact >= 999
p.loc[ph, "hours_to_first_contact"] = np.nan
p["hours_to_first_contact"] = p.hours_to_first_contact.fillna(p.groupby("channel").hours_to_first_contact.transform("median"))
cost_na = p.acquisition_cost_inr.isna()
p["acquisition_cost_inr"] = p.acquisition_cost_inr.fillna(p.groupby("channel").acquisition_cost_inr.transform("median"))
neg = p.months_active < 0
p.loc[neg, "months_active"] = np.nan
for issue, n, act in [("Purchase date before lead date", bad_date.sum(), "Rebuilt from lead date + days to close"),
                      ("NPS outside 0-10", bad_nps.sum(), "Set to missing"), ("Contact time placeholder 999", ph.sum(), "Missing -> channel median"),
                      ("Acquisition cost missing", cost_na.sum(), "Channel median"), ("Negative months active", neg.sum(), "Set to missing")]:
    log("pipeline", "B6", issue, n, act)
    print(f"{issue:32s} {n:4d}  -> {act}")

# %% [md]
# ### B7. Logical consistency of the funnel
# A lead can only buy after a demo, and only be demoed after being contacted. Refill, churn and NPS fields must be
# blank for non-buyers. These checks confirm the funnel is internally consistent after cleaning.

# %%
checks = {
    "purchased but no demo": ((p.purchased == 1) & (p.demo_done != 1)).sum(),
    "demo but not contacted": ((p.demo_done == 1) & (p.contacted != 1)).sum(),
    "subscribed but not purchased": ((p.refill_subscribed == 1) & (p.purchased != 1)).sum(),
    "purchased but no order value": ((p.purchased == 1) & p.order_value_inr.isna()).sum(),
    "stage_reached disagrees with flags": (p.stage_reached != np.select([p.purchased == 1, p.demo_done == 1, p.contacted == 1],
                                                                       ["Purchased", "Demo done", "Contacted"], "New lead")).sum(),
}
pd.Series(checks, name="violations")

# %% [md]
# ### B8. Feature engineering for the pipeline

# %%
CUTOFF = pd.Timestamp("2027-07-31")
p["lead_week"] = ((p.lead_created - pd.Timestamp("2027-01-01")).dt.days // 7 + 1).clip(1, 26)
p["lead_month"] = p.lead_created.dt.to_period("M").astype(str)
p["funnel_stage"] = p.contacted.fillna(0) + p.demo_done.fillna(0) + p.purchased.fillna(0) + p.refill_subscribed.fillna(0)
p["response_bucket"] = pd.cut(p.hours_to_first_contact, [0, 2, 6, 24, np.inf], labels=["Under 2 h", "2-6 h", "6-24 h", "Over 24 h"])
p["space_level"] = p.space_type.map({"No balcony (window sill)": 0, "Small balcony (<40 sq ft)": 1, "Large balcony (40-100 sq ft)": 2, "Terrace or garden (>100 sq ft)": 3})
p["space_label"] = pd.Categorical(p.space_type.str.replace(r" \(.*\)", "", regex=True), ["No balcony", "Small balcony", "Large balcony", "Terrace or garden"], ordered=True)
p["income_level"] = p.income_band.map({"Below Rs 50,000": 1, "Rs 50,000-1 lakh": 2, "Rs 1-2 lakh": 3, "Rs 2-3.5 lakh": 4, "Above Rs 3.5 lakh": 5})
p["order_value_ex_gst"] = (p.order_value_inr / 1.18).round(0)
p["addons"] = p.expansion_tier.fillna(0) + p.smart_sensor.fillna(0)
p["fast_first_harvest"] = np.where(p.days_to_first_harvest.isna(), np.nan, (p.days_to_first_harvest <= 35).astype(float))
p["nps_group"] = pd.cut(p.nps_score, [-1, 6, 8, 10], labels=["Detractor (0-6)", "Passive (7-8)", "Promoter (9-10)"])
p["months_observable"] = np.where(p.purchased == 1, np.floor((CUTOFF - p.purchase_date).dt.days / 30.4), np.nan)
p["price_label"] = "₹" + p.price_offered.map("{:,}".format)
p.loc[:, ["lead_id", "lead_created", "lead_week", "channel", "response_bucket", "funnel_stage", "order_value_ex_gst", "fast_first_harvest", "nps_group"]].head()

# %% [md]
# ### B9. Score the pipeline cleaning against the noise log

# %%
noise = pd.read_csv("data/raw/planted_noise_log.csv")
pipe_noise = noise.iloc[noise.index[noise.issue == "--- pipeline ---"][0] + 1:].set_index("issue").rows
done = pd.DataFrame(LOG).query("dataset == 'pipeline'").set_index("issue").rows_affected
match = {"Duplicate lead (same phone, entered twice)": "Duplicate lead (same phone)", "Channel missing": "Channel missing",
         "Order value with an extra zero": "Order value with extra zero", "Purchase date before the pilot began": "Purchase date before lead date",
         "NPS outside 0-10": "NPS outside 0-10", "Contact time placeholder 999": "Contact time placeholder 999",
         "Acquisition cost missing": "Acquisition cost missing", "Negative months active": "Negative months active"}
pd.DataFrame({"planted": [pipe_noise[k] for k in match], "found_and_fixed": [done.get(v, 0) for v in match.values()]}, index=list(match))

# %% [md]
# Planted and found counts can differ slightly where a planted problem landed on a row that was later removed as a
# duplicate – the problem disappears with the row.

# %% [md]
# ## Part C – Save the clean data

# %%
SURVEY_COLS = (["response_id", "submitted", "submitted_date", "submitted_hour", "minutes_taken", "city", "city_tier", "income", "income_label",
                "income_disclosed", "income_mid_inr", "household_size", "children", "space_level", "space_label", "sun_level", "veg_spend",
                "veg_spend_imputed", "spend_per_person", "spoil_level", "att_health", "att_eco", "att_tech", "att_time_poor",
                "att_hobby", "green_index", "experience", "experience_level", "novice", "kept_up_6m", "tried_and_quit"]
               + list(OPTIONS["features"].values()) + list(OPTIONS["crops"].values()) + list(OPTIONS["worries"].values())
               + ["n_features", "n_crops", "n_worries", "price_shown", "price_band", "intent", "intent_score", "would_buy", "wtp_system",
                  "wtp_system_outlier", "wtp_gap", "can_afford_shown", "wtp_refill", "wtp_refill_outlier", "refill_price_shown",
                  "refill_affordable", "subscribe", "subscribe_score", "would_subscribe"])
PIPE_COLS = ["lead_id", "lead_created", "lead_week", "lead_month", "city", "channel", "household_size", "space_type", "space_level", "space_label",
             "income_band", "income_level", "gardening_experience", "sales_rep", "hours_to_first_contact", "response_bucket", "price_offered",
             "price_label", "contacted", "demo_done", "demo_type", "purchased", "purchase_date", "days_to_close", "order_value_inr",
             "order_value_ex_gst", "order_value_fixed", "expansion_tier", "smart_sensor", "addons", "discount_used", "onboarding_call",
             "days_to_first_harvest", "fast_first_harvest", "refill_subscribed", "months_active", "months_observable", "churned",
             "nps_score", "nps_group", "acquisition_cost_inr", "funnel_stage", "stage_reached"]
survey_clean, pipeline_clean = s[SURVEY_COLS].copy(), p[PIPE_COLS].sort_values("lead_created").copy()
survey_clean.to_csv("data/clean/survey_clean.csv", index=False)
pipeline_clean.to_csv("data/clean/pipeline_clean.csv", index=False)
cleaning_log = pd.DataFrame(LOG)
cleaning_log.to_csv("data/clean/cleaning_log.csv", index=False)
print("survey_clean  :", survey_clean.shape)
print("pipeline_clean:", pipeline_clean.shape)

# %% [md]
# ### Before and after

# %%
summary = pd.DataFrame({
    "rows": [len(survey_raw), len(survey_clean), len(pipe_raw), len(pipeline_clean)],
    "columns": [survey_raw.shape[1], survey_clean.shape[1], pipe_raw.shape[1], pipeline_clean.shape[1]],
    "text (object) columns": [(survey_raw.dtypes == object).sum(), (survey_clean.dtypes == object).sum(),
                              (pipe_raw.dtypes == object).sum(), (pipeline_clean.dtypes == object).sum()],
}, index=["Survey – raw", "Survey – clean", "Pipeline – raw", "Pipeline – clean"])
summary

# %%
cleaning_log

# %%
from IPython.display import Markdown
Markdown(f"""**Result.** The survey keeps **{len(survey_clean):,} trustworthy responses** out of {len(survey_raw):,} submissions; the pipeline keeps
**{len(pipeline_clean):,} unique leads** out of {len(pipe_raw):,} rows. Every change is in `data/clean/cleaning_log.csv`. Text answers are now
numeric scores, multi-selects are 0/1 columns, and new business measures (top-2-box intent, price gap, response-time bucket, fast-harvest flag,
NPS group) are ready for the descriptive and diagnostic analysis in Step 3.""")

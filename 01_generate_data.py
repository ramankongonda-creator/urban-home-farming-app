"""
01_generate_data.py  -  Synthetic data for an urban home-farming start-up (India)
================================================================================
SYNTHETIC / ILLUSTRATIVE DATA. Nothing here is real market evidence.

Creates two raw datasets that together cover the start-up's sales pipeline end to end:

  data/raw/survey_raw.csv     3,000 consumer survey responses (+45 duplicate submissions), exported
                              the way a form tool exports them: question text as headers, text answers,
                              typed amounts ("6k", "Rs 6,000"), city spellings, ticked options joined by ";".
                              -> validates demand, price and the refill-subscription idea BEFORE launch.
  data/raw/pipeline_raw.csv   ~4,300 leads from a 6-month pilot (Jan-Jun 2027, Bengaluru + Pune) exported
                              from a CRM: lead -> contacted -> demo -> purchase -> refill subscription -> retention.
                              -> tests whether stated intent turns into real sales AFTER launch.

Both files carry deliberate real-world problems (missing values, inconsistent spellings, mixed date
formats, impossible values, duplicates, speeders, straight-liners). Every planted problem is logged in
data/raw/planted_noise_log.csv and data/raw/survey_answer_key_planted_issues.csv so the cleaning step can
be checked against the truth.

Run:  python 01_generate_data.py      (fixed random seeds -> identical output every run)
"""
import os
import numpy as np
import pandas as pd

os.makedirs("data/raw", exist_ok=True)
rng = np.random.default_rng(42)
N = 3000

PRICES = [2999, 3999, 4999, 5999, 7499, 9999]
REFILL_PRICES = [199, 299, 399]
FEAT = {"f_vertical": "Vertical design", "f_selfwater": "Self-watering", "f_hydroponic": "Hydroponic",
        "f_soil": "Soil / cocopeat", "f_smart": "Smart sensor + app", "f_organic": "Organic seeds",
        "f_subscription": "Refill subscription", "f_install": "Installation visits", "f_growlight": "Grow light"}
CROP = {"g_herbs": "Herbs", "g_leafy": "Leafy greens", "g_micro": "Microgreens", "g_tomato": "Tomatoes & chillies", "g_exotic": "Exotic greens"}
WORRY = {"w_price": "Price", "w_space": "Lack of space", "w_time": "Time and upkeep", "w_dying": "Plants dying",
         "w_pests": "Pests or mess", "w_leak": "Leakage / society rules"}
INCOME_LAB = {1: "Below Rs 50,000", 2: "Rs 50,000-1 lakh", 3: "Rs 1-2 lakh", 4: "Rs 2-3.5 lakh", 5: "Above Rs 3.5 lakh", 9: "Prefer not to say"}
SPACE_LAB = {0: "No balcony (window sill)", 1: "Small balcony (<40 sq ft)", 2: "Large balcony (40-100 sq ft)", 3: "Terrace or garden (>100 sq ft)"}
SUN_LAB = {1: "<2 hours", 2: "2-4 hours", 3: "4-6 hours", 4: ">6 hours"}
EXP_LAB = {0: "No experience", 1: "Kept a few plants", 2: "Grown vegetables before", 3: "Grow vegetables now"}
KEEP_LAB = {0: "Never tried", 1: "Yes, kept it up 6+ months", 2: "No, stopped within 6 months"}
INT_LAB = {1: "Very unlikely", 2: "Unlikely", 3: "Not sure", 4: "Likely", 5: "Very likely"}
SUB_LAB = {1: "Definitely not", 2: "Probably not", 3: "Not sure", 4: "Probably yes", 5: "Definitely yes"}

# ============================================================== 1. SIMULATE (hidden personas)
PERS = ["Health-first families", "Eco-minded young professionals", "Terrace gardeners", "Affluent convenience seekers", "Budget-conscious households"]
persona = rng.choice(5, N, p=[.24, .20, .17, .13, .26])
P = lambda v: np.array(v, dtype=float)[persona]
cat = lambda probs, opts: np.array([rng.choice(opts, p=probs[k]) for k in persona])

city = cat([[.70, .25, .05], [.85, .13, .02], [.30, .50, .20], [.80, .18, .02], [.45, .38, .17]], [1, 2, 3])
income = cat([[.05, .30, .45, .15, .05], [.05, .25, .40, .22, .08], [.10, .35, .35, .15, .05], [0, .05, .25, .40, .30], [.45, .40, .13, .02, 0]], [1, 2, 3, 4, 5])
hh = np.clip(np.round(rng.normal(P([4.4, 1.9, 4.2, 3.4, 3.6]), 1.0)), 1, 9).astype(int)
kids = (rng.random(N) < P([.85, .10, .45, .45, .45])).astype(int)
space = cat([[.10, .55, .30, .05], [.30, .55, .13, .02], [0, .05, .20, .75], [.03, .25, .50, .22], [.25, .55, .15, .05]], [0, 1, 2, 3])
sun = np.clip(np.round(np.array([1.6, 2.2, 3.0, 3.6])[space] + rng.normal(0, .8, N)), 1, 4).astype(int)
lik = lambda m, sd=.85: np.clip(np.round(rng.normal(m, sd)), 1, 5).astype(int)
health, eco, tech = lik(P([4.4, 3.6, 3.8, 3.7, 3.0])), lik(P([3.5, 4.4, 3.9, 3.2, 2.9])), lik(P([3.3, 4.4, 2.6, 4.2, 2.8]))
timep, hobby = lik(P([3.9, 3.8, 2.4, 4.3, 3.0])), lik(P([3.2, 3.3, 4.5, 2.8, 2.9]))
exp_ = cat([[.45, .40, .12, .03], [.55, .38, .06, .01], [.03, .17, .40, .40], [.50, .38, .10, .02], [.40, .40, .15, .05]], [0, 1, 2, 3])
# Q11 retention proxy: of those who tried growing (exp>=1), did they keep it up 6+ months?
p_keep = 1 / (1 + np.exp(-(-0.55 + .55 * (hobby - 3) - .35 * (timep - 3) + .45 * (exp_ >= 2) + .25 * (sun >= 3))))
keep = np.where(exp_ == 0, 0, np.where(rng.random(N) < p_keep, 1, 2))
spend = np.clip(1500 + 650 * hh + 900 * income + 250 * (health - 3) + rng.normal(0, 750, N), 600, 16000)
spoil = np.clip(np.round(rng.normal(2.4 + .35 * (timep - 3) + .15 * (hh - 3) - .2 * (exp_ - 1), .8)), 1, 4).astype(int)
wtp = np.clip(900 + 620 * income + 450 * space + 260 * (sun - 2) + 420 * (health - 3) + 380 * (eco - 3) + 330 * (tech - 3)
              + 200 * (timep - 3) + 150 * (hobby - 3) + .10 * (spend - 6000) + P([500, 300, 200, 900, -500]) + rng.normal(0, 950, N), 400, 25000)
refill_wtp = np.clip(110 + 45 * income + 40 * (health - 3) + 30 * (tech - 3) + 25 * (timep - 3) + 20 * (hobby - 3) + rng.normal(0, 75, N), 0, 1500)
shown = rng.permutation(np.resize(PRICES, N)); refill_shown = rng.permutation(np.resize(REFILL_PRICES, N))
z = lambda x: (x - x.mean()) / x.std()
interest = (0.15 + .55 * z(health) + .50 * z(eco) + .30 * z(tech) + .25 * z(hobby) + .45 * z(space.astype(float)) + .30 * z(sun.astype(float))
            + .30 * kids + .25 * z(spoil.astype(float)) - .35 * (exp_ == 3) + rng.normal(0, .55, N))
p_buy = 1 / (1 + np.exp(-interest)) / (1 + np.exp(-(wtp - shown) / 950))
likely = rng.random(N) < p_buy
intent = np.where(likely, rng.choice([4, 5], N, p=[.62, .38]), rng.choice([1, 2, 3], N, p=[.22, .38, .40]))
sub_score = (-1.25 + 1.0 * likely + .35 * (tech - 3) + .30 * (timep - 3) + .25 * (health - 3) + .25 * (exp_ <= 1)
             - .35 * (keep == 2) + (refill_wtp - refill_shown) / 70 + rng.normal(0, .7, N))
sub = np.clip(np.round(3 + 1.0 * sub_score), 1, 5).astype(int)
pr = lambda x: (rng.random(N) < np.clip(x, .02, .97)).astype(int)
small, novice = (space <= 1).astype(int), (exp_ <= 1).astype(int)
f = dict(f_vertical=pr(.30 + .50 * small + .05 * (space == 2)), f_selfwater=pr(.20 + .14 * (timep - 3) + .20 * novice),
         f_hydroponic=pr(.15 + .14 * (tech - 3) + .20 * novice - .15 * (exp_ >= 2)), f_soil=pr(.12 + .45 * (exp_ >= 2) + .10 * (space == 3)),
         f_smart=pr(.10 + .17 * (tech - 3) + .05 * (income - 3)), f_organic=pr(.20 + .17 * (health - 3) + .12 * kids),
         f_subscription=pr(.08 + .25 * (sub >= 4) + .10 * novice + .05 * (timep - 3)), f_install=pr(.06 + .12 * (timep - 3) + .07 * (income - 3)),
         f_growlight=pr(.05 + .40 * (sun == 1) + .15 * (sun == 2) + .10 * (space == 0)))
g = dict(g_herbs=pr(.65), g_leafy=pr(.45 + .10 * (health - 3) + .10 * (hh >= 4)), g_micro=pr(.12 + .25 * kids + .08 * (health - 3) + .05 * (tech - 3)),
         g_tomato=pr(.15 + .35 * (exp_ >= 2) + .15 * (sun >= 3)), g_exotic=pr(.10 + .10 * (income - 3) + .12 * (city == 1)))
wp = dict(w_price=.15 + .20 * (income <= 2), w_space=.10 + .25 * (space == 0) + .10 * (space == 1), w_time=.10 + .15 * (timep - 3),
          w_dying=.10 + .25 * novice + .15 * (keep == 2), w_pests=.12 * np.ones(N), w_leak=.06 + .10 * (space <= 2) * (city == 1))
w = {k: pr(v) for k, v in wp.items()}
Wm, Pm = np.column_stack(list(w.values())), np.column_stack([np.broadcast_to(v, N) for v in wp.values()])
for i in np.where(Wm.sum(1) > 2)[0]:                           # "tick up to two": keep the two most probable ticks
    ticked = np.where(Wm[i] == 1)[0]; keep2 = ticked[np.argsort(-Pm[i, ticked])[:2]]
    for j, k in enumerate(w): w[k][i] = int(j in keep2)
minutes = np.clip(rng.lognormal(np.log(6.0), .28, N), 2.4, 25)
start = pd.Timestamp("2026-10-12 08:00")
day = rng.integers(0, 21, N); hour = np.clip(rng.normal(19, 3.5, N), 6, 23.9)
ts = start.normalize() + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 60, unit="m")

df = pd.DataFrame(dict(
    response_id=[f"R{i:05d}" for i in range(1, N + 1)], device_id=[f"D{rng.integers(10**7, 10**8)}" for _ in range(N)],
    submitted=ts.floor("min"), minutes_taken=minutes.round(1),
    city_tier=city, income_band=income, household_size=hh, children=kids, space_level=space, sunlight_band=sun,
    monthly_veg_spend_inr=spend, spoilage=spoil, att_health=health, att_eco=eco, att_tech=tech, att_time_poor=timep, att_hobby=hobby,
    experience=exp_, kept_up_6m=keep, **f, **g, **w, price_shown_inr=shown, intent_1to5=intent,
    wtp_system_inr=wtp, wtp_refill_inr=refill_wtp, refill_price_shown_inr=refill_shown, subscribe_1to5=sub, _persona=persona))

# ============================================================== 2. REAL-WORLD NOISE
ATT = ["att_health", "att_eco", "att_tech", "att_time_poor", "att_hobby"]
noise_log = []
def heap(x, base_probs):                                        # people round numbers they type
    r = rng.random(len(x)); out = x.copy()
    out = np.where(r < base_probs[0], np.round(x / 1000) * 1000, np.where(r < base_probs[1], np.round(x / 500) * 500, np.round(x / 100) * 100))
    return out
df["wtp_system_inr"] = heap(df.wtp_system_inr.values, (.45, .80))
df["monthly_veg_spend_inr"] = heap(df.monthly_veg_spend_inr.values, (.40, .75))
df["wtp_refill_inr"] = np.round(df.wtp_refill_inr / 50) * 50
yes = rng.random(N) < .07                                        # yes-saying (acquiescence) bias
df.loc[yes, ATT] = np.clip(df.loc[yes, ATT] + 1, 1, 5); noise_log.append(("Yes-saying bias (+1 on all attitude items)", int(yes.sum())))
for col, rate, lab in [("income_band", .05, "Q2 'Prefer not to say'"), ("monthly_veg_spend_inr", .03, "Q7 left blank"),
                       ("wtp_system_inr", .04, "Q16a left blank"), ("wtp_refill_inr", .05, "Q16b left blank")]:
    m = rng.random(N) < rate
    df.loc[m, col] = 9 if col == "income_band" else np.nan
    noise_log.append((lab, int(m.sum())))
# planted problems
idx = rng.permutation(N); cur = 0
def take(k):
    global cur; s = idx[cur:cur + k]; cur += k; return s
planted = {}
s = take(10); df.loc[s, "wtp_system_inr"] = df.loc[s, "wtp_system_inr"].fillna(4000) * 10; planted.update({i: "Extra zero in Q16a max price" for i in s})
s = take(6); df.loc[s, "household_size"] = rng.choice([0, 18, 25, 40], 6); planted.update({i: "Impossible household size" for i in s})
s = take(8); df.loc[s, "monthly_veg_spend_inr"] = rng.choice([65000, 90000, 150000], 8); planted.update({i: "Extreme monthly veg spend" for i in s})
s = take(6); df.loc[s, "monthly_veg_spend_inr"] = 0; df.loc[s, "household_size"] = np.maximum(df.loc[s, "household_size"], 3); planted.update({i: "Rs 0 spend for 3+ people" for i in s})
s = take(8); df.loc[s, "wtp_system_inr"] = rng.choice([0, 50, 100], 8); df.loc[s, "intent_1to5"] = 5; planted.update({i: "Max price under Rs 200 but 'Very likely'" for i in s})
s = take(18); v = rng.choice([1, 5], 18)
for i, vv in zip(s, v):
    df.loc[i, ATT + ["intent_1to5", "subscribe_1to5"]] = vv; df.loc[i, list(FEAT) + list(CROP)] = int(vv == 5); df.loc[i, "minutes_taken"] = round(rng.uniform(1.0, 1.9), 1)
planted.update({i: "Straight-liner (all same answers) and speeder" for i in s})
s = take(22); df.loc[s, "minutes_taken"] = rng.uniform(.8, 1.9, 22).round(1); planted.update({i: "Speeder (under 2 minutes)" for i in s})
dup_src = take(45)
dups = df.loc[dup_src].copy()
dups["response_id"] = [f"R{N + 1 + k:05d}" for k in range(len(dups))]
dups["submitted"] = dups.submitted + pd.to_timedelta(rng.integers(3, 600, len(dups)), unit="m")
dups["minutes_taken"] = (dups.minutes_taken * rng.uniform(.6, 1.1, len(dups))).round(1)
df = pd.concat([df, dups], ignore_index=True).sort_values("submitted").reset_index(drop=True)
planted_ids = {df.loc[df.index[df.response_id == f"R{i + 1:05d}"][0], "response_id"]: r for i, r in planted.items()}
planted_ids.update({r: "Duplicate submission (same device)" for r in dups.response_id})
noise_log += [("Planted outliers / bad responses", len(planted)), ("Duplicate submissions", len(dups))]
raw = df.copy()


# ============================================================== 3. EXPORT THE SURVEY AS A FORM TOOL WOULD (text answers)
FORM_FEAT = {"f_vertical": "Vertical / stackable design", "f_selfwater": "Self-watering reservoir", "f_hydroponic": "Hydroponic (no soil)",
             "f_soil": "Soil or cocopeat based", "f_smart": "Smart sensor with app alerts", "f_organic": "Organic seeds and nutrients",
             "f_subscription": "Monthly refill subscription", "f_install": "Installation and maintenance visits", "f_growlight": "Grow light for indoor use"}
FORM_CROP = {"g_herbs": "Herbs (coriander mint tulsi)", "g_leafy": "Leafy greens (spinach methi lettuce)", "g_micro": "Microgreens",
             "g_tomato": "Tomatoes and chillies", "g_exotic": "Exotic greens (basil kale cherry tomato)"}
FORM_WORRY = {"w_price": "Price", "w_space": "Lack of space", "w_time": "Time and upkeep", "w_dying": "Plants dying",
              "w_pests": "Pests or mess", "w_leak": "Water leakage or society rules"}
CITY_NAMES = {1: ["Mumbai", "Delhi NCR", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune"],
              2: ["Jaipur", "Lucknow", "Indore", "Coimbatore", "Kochi", "Chandigarh", "Nagpur", "Surat"],
              3: ["Mysuru", "Nashik", "Udaipur", "Dehradun", "Guntur"]}
CITY_VARIANTS = {"Mumbai": ["Mumbai", "mumbai", "Bombay", "MUMBAI "], "Delhi NCR": ["Delhi NCR", "Delhi", "Gurgaon", "Noida", "new delhi"],
                 "Bengaluru": ["Bengaluru", "Bangalore", "bangalore ", "BLR"], "Chennai": ["Chennai", "chennai", "Madras"],
                 "Hyderabad": ["Hyderabad", "hyderabad", "Hyd"], "Kolkata": ["Kolkata", "Calcutta", "kolkata"], "Pune": ["Pune", "pune", "Poona"]}
rr = np.random.default_rng(7)

def money_text(v, style_probs=(.80, .07, .06, .04, .03)):
    """Render a typed rupee amount the way people actually type it."""
    if pd.isna(v): return ""
    v = int(round(v)); k = rr.choice(5, p=style_probs)
    if k == 0: return str(v)
    if k == 1: return f"Rs {v:,}"
    if k == 2: return f"{v/1000:g}k" if v % 100 == 0 and v >= 1000 else str(v)
    if k == 3: return f"{v}/-"
    return f"₹{v:,}"

def hh_text(v):
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    k = rr.random()
    if k < .94 or v not in words: return str(int(v))
    return f"{int(v)} members" if k < .97 else words[int(v)]

def city_text(tier):
    c = rr.choice(CITY_NAMES[int(tier)])
    if c in CITY_VARIANTS and rr.random() < .35: return rr.choice(CITY_VARIANTS[c])
    return c

def ticks(row, mapping):
    return "; ".join(v for k, v in mapping.items() if row[k] == 1)

form = pd.DataFrame({
    "Timestamp": raw.submitted.dt.strftime("%d/%m/%Y %H:%M:%S"),
    "Response ID": raw.response_id, "Device ID": raw.device_id, "Time taken (min)": raw.minutes_taken,
    "Q1 City": [city_text(t) for t in raw.city_tier],
    "Q2 Monthly household income": raw.income_band.map({**INCOME_LAB, 9: "Prefer not to say"}),
    "Q3 Household size": [hh_text(v) for v in raw.household_size],
    "Q4 Children under 15": raw.children.map({1: "Yes", 0: "No"}),
    "Q5 Growing space": raw.space_level.map(SPACE_LAB),
    "Q6 Direct sunlight": raw.sunlight_band.map(SUN_LAB),
    "Q7 Monthly veg & herb spend (Rs)": [money_text(v) for v in raw.monthly_veg_spend_inr],
    "Q8 Greens spoil before use": raw.spoilage.map({1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often"}),
    "Q9a Prefer pesticide-free produce (1-5)": raw.att_health, "Q9b Reduce environmental impact (1-5)": raw.att_eco,
    "Q9c Comfortable with apps/smart devices (1-5)": raw.att_tech, "Q9d Little time for chores (1-5)": raw.att_time_poor,
    "Q9e Would enjoy plants as a hobby (1-5)": raw.att_hobby,
    "Q10 Gardening experience": raw.experience.map(EXP_LAB),
    "Q11 Kept up growing 6+ months": raw.kept_up_6m.map(KEEP_LAB),
    "Q12 Features wanted": raw.apply(lambda r: ticks(r, FORM_FEAT), axis=1),
    "Q13 Would like to grow": raw.apply(lambda r: ticks(r, FORM_CROP), axis=1),
    "Q14 Biggest worries": raw.apply(lambda r: ticks(r, FORM_WORRY), axis=1),
    "Q15 Price shown (Rs)": raw.price_shown_inr, "Q15 Likelihood to buy": raw.intent_1to5.map(INT_LAB),
    "Q16a Max price for starter system (Rs)": [money_text(v) for v in raw.wtp_system_inr],
    "Q16b Max price for monthly refill (Rs)": [money_text(v, (.85, .05, 0, .06, .04)) for v in raw.wtp_refill_inr],
    "Q17 Refill price shown (Rs)": raw.refill_price_shown_inr, "Q17 Would subscribe": raw.subscribe_1to5.map(SUB_LAB),
})
form.to_csv("data/raw/survey_raw.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([dict(response_id=k, planted_issue=v) for k, v in planted_ids.items()]).to_csv("data/raw/survey_answer_key_planted_issues.csv", index=False)
print("survey_raw.csv:", form.shape)

# ============================================================== 4. PILOT SALES PIPELINE (6-month CRM export)
# A hypothetical 6-month pilot in Bengaluru and Pune, Jan-Jun 2027, run after the survey.
pr = np.random.default_rng(2027)
NL = 4200
CH = ["Instagram ads", "Google search", "Housing-society demo", "Referral", "Corporate wellness", "Website (organic)"]
week = np.clip(np.floor(pr.beta(1.6, 1.2, NL) * 26), 0, 25).astype(int)             # volume ramps up over 26 weeks
ch_w = np.array([[.36, .20, .18, .04, .10, .12], [.30, .17, .21, .14, .07, .11]])
chan = np.array([pr.choice(6, p=ch_w[0] + (ch_w[1] - ch_w[0]) * (wk / 25)) for wk in week])
created = pd.Timestamp("2027-01-01") + pd.to_timedelta(np.minimum(week * 7 + pr.integers(0, 7, NL), 180), unit="D") + pd.to_timedelta(pr.integers(8 * 60, 22 * 60, NL), unit="m")
city = np.where(pr.random(NL) < .6, "Bengaluru", "Pune")
space = pr.choice(4, NL, p=[.12, .48, .28, .12])
hh = np.clip(np.round(pr.normal(3.4, 1.3, NL)), 1, 8).astype(int)
income = pr.choice(5, NL, p=[.08, .25, .35, .22, .10]) + 1
exp_l = pr.choice(4, NL, p=[.45, .35, .12, .08])
price = pr.choice([4999, 5499, 5999], NL)                                            # A/B price test
rep = pr.choice(["Ananya", "Rohit", "Meera", "Karthik", "Sneha", "Vikram"], NL)
rep_skill = pd.Series(rep).map({"Ananya": .35, "Rohit": 0, "Meera": .2, "Karthik": -.25, "Sneha": .1, "Vikram": -.4}).values
resp_h = np.round(pr.lognormal(np.log(np.array([6, 4, 2, 1.5, 5, 8])[chan]), .9), 1)     # hours to first contact
cpl = np.array([210, 280, 230, 40, 180, 30])[chan] * pr.lognormal(0, .25, NL)             # marketing cost per lead
sig = lambda x: 1 / (1 + np.exp(-x))
p_contact = sig(np.array([.6, 1.2, 2.4, 2.6, 1.6, .9])[chan] - .25 * np.log1p(resp_h))
contacted = pr.random(NL) < p_contact
demo_type = np.where(chan == 2, "In person", np.where(pr.random(NL) < .3, "In person", "Video call"))
p_demo = sig(np.array([-.9, -.4, 1.1, .8, .3, -.6])[chan] + .25 * (space - 1) + rep_skill * .8 - .2 * np.log1p(resp_h))
demo = contacted & (pr.random(NL) < p_demo)
p_buy = sig(-.85 + np.array([-.35, -.1, .55, .7, .2, -.2])[chan] + .35 * (space - 1) + .30 * (income - 3) + .5 * (demo_type == "In person")
            - (price - 4999) / 800 + rep_skill + pr.normal(0, .4, NL))
purchased = demo & (pr.random(NL) < p_buy)
days_close = np.where(purchased, np.clip(np.round(pr.gamma(2.2, 4.5, NL) + 2 * (resp_h > 24)), 1, 60), np.nan)
purchase_dt = created + pd.to_timedelta(np.nan_to_num(days_close), unit="D")
addon_exp = purchased & (pr.random(NL) < .18 + .12 * (space >= 2))
addon_sensor = purchased & (pr.random(NL) < .08 + .04 * (income >= 4))
discount = purchased & (pr.random(NL) < .15)
order = np.where(purchased, (price + 1999 * addon_exp + 1499 * addon_sensor) * np.where(discount, .9, 1), np.nan)
onboard = purchased & (pr.random(NL) < .62)
harvest = np.where(purchased, np.clip(np.round(pr.normal(np.where(onboard, 31, 44) + 4 * (exp_l == 0), 7)), 18, 80), np.nan)
p_sub = sig(-.55 + .55 * onboard + .25 * (income - 3) + .3 * (chan == 3) - .02 * (np.nan_to_num(harvest) - 35))
subscribed = purchased & (pr.random(NL) < p_sub)
cutoff = pd.Timestamp("2027-07-31")
avail = np.where(purchased, np.floor((cutoff - purchase_dt).days / 30.4), 0).astype(int)
hazard = np.clip(.115 - .055 * (np.nan_to_num(harvest) <= 35) - .02 * onboard + .025 * (exp_l == 0), .02, .3)
life = np.where(subscribed, pr.geometric(hazard), 0)
months_active = np.where(subscribed, np.minimum(life, avail), 0)
churned = subscribed & (life <= avail)
nps = np.where(purchased, np.clip(np.round(pr.normal(8.2 - .08 * (np.nan_to_num(harvest) - 35) - 1.5 * churned, 1.6)), 0, 10), np.nan)
acq = np.round(cpl + np.where(purchased & (chan == 3), 300, 0))                       # referral reward paid on purchase
stage = np.select([purchased, demo, contacted], ["Purchased", "Demo done", "Contacted"], "New lead")

pl = pd.DataFrame(dict(
    lead_id=[f"L{i:05d}" for i in range(1, NL + 1)], phone_hash=[f"P{pr.integers(10**8, 10**9)}" for _ in range(NL)],
    lead_created=created, city=city, channel=np.array(CH)[chan], household_size=hh,
    space_type=pd.Series(space).map(SPACE_LAB).values, income_band=pd.Series(income).map(INCOME_LAB).values,
    gardening_experience=pd.Series(exp_l).map(EXP_LAB).values, sales_rep=rep, hours_to_first_contact=resp_h,
    price_offered=price, contacted=np.where(contacted, "Yes", "No"), demo_done=np.where(demo, "Yes", "No"),
    demo_type=np.where(demo, demo_type, ""), purchased=np.where(purchased, "Yes", "No"),
    purchase_date=np.where(purchased, purchase_dt.strftime("%Y-%m-%d"), ""), days_to_close=days_close, order_value_inr=order,
    expansion_tier=np.where(purchased, np.where(addon_exp, "Yes", "No"), ""), smart_sensor=np.where(purchased, np.where(addon_sensor, "Yes", "No"), ""),
    discount_used=np.where(purchased, np.where(discount, "Yes", "No"), ""),
    onboarding_call=np.where(purchased, np.where(onboard, "Yes", "No"), ""), days_to_first_harvest=harvest,
    refill_subscribed=np.where(purchased, np.where(subscribed, "Yes", "No"), ""), months_active=np.where(subscribed, months_active, np.nan),
    churned=np.where(subscribed, np.where(churned, "Yes", "No"), ""), nps_score=nps, acquisition_cost_inr=acq, stage_reached=stage))

# ---- real CRM mess
mess = []
def pick(frac, mask=None):
    idx = pl.index if mask is None else pl.index[mask]
    return pr.choice(idx, int(round(len(idx) * frac)), replace=False)
CITY_V = {"Bengaluru": ["Bangalore", "bangalore ", "BLR", "Bengaluru "], "Pune": ["pune", "PUNE ", "Poona"]}
i = pick(.18); pl.loc[i, "city"] = [pr.choice(CITY_V[c]) for c in pl.loc[i, "city"]]; mess.append(("City spelt inconsistently", len(i)))
CH_V = {"Instagram ads": ["Instagram Ads", "IG ads", "instagram"], "Google search": ["Google Search", "google"], "Housing-society demo": ["Society demo", "housing society demo"],
        "Referral": ["referral ", "Referral program"], "Corporate wellness": ["Corporate", "corporate wellness"], "Website (organic)": ["Website", "website/organic"]}
i = pick(.15); pl.loc[i, "channel"] = [pr.choice(CH_V[c]) for c in pl.loc[i, "channel"]]; mess.append(("Channel spelt inconsistently", len(i)))
i = pick(.02); pl.loc[i, "channel"] = ""; mess.append(("Channel missing", len(i)))
i = pick(.25); pl.loc[i, "income_band"] = ""; mess.append(("Income not given", len(i)))
dates = pl.lead_created.dt.strftime("%Y-%m-%d %H:%M").copy()
i = pick(.12); dates[i] = pl.loc[i, "lead_created"].dt.strftime("%d/%m/%Y"); mess.append(("Lead date in dd/mm/yyyy", len(i)))
i = pick(.05); dates[i] = pl.loc[i, "lead_created"].dt.strftime("%b %d %Y"); mess.append(("Lead date as 'Mon dd yyyy'", len(i)))
pl["lead_created"] = dates
for col in ["purchased", "contacted", "demo_done"]:
    i = pick(.10); pl.loc[i, col] = [pr.choice(["Y", "yes", "1", "TRUE"]) if v == "Yes" else pr.choice(["N", "no", "0", "FALSE"]) for v in pl.loc[i, col]]
mess.append(("Yes/No stored as Y, 1, TRUE and similar", 3 * int(round(NL * .10))))
buy = pl.purchased.isin(["Yes", "Y", "yes", "1", "TRUE"]) & pl.order_value_inr.notna()
ov = pl.order_value_inr.astype(object)
i = pick(.20, buy); ov[i] = [pr.choice([f"Rs {int(v):,}", f"{v:.2f}", f"₹{int(v):,}"]) for v in pl.loc[i, "order_value_inr"]]; mess.append(("Order value typed as text", len(i)))
i = pick(.006, buy); ov[i] = [str(int(v) * 10) for v in pl.loc[i, "order_value_inr"]]; mess.append(("Order value with an extra zero", len(i)))
pl["order_value_inr"] = ov
i = pick(.004, buy); pl.loc[i, "purchase_date"] = "2026-12-15"; mess.append(("Purchase date before the pilot began", len(i)))
i = pick(.02, buy); pl.loc[i, "nps_score"] = pr.choice([11, 99, -1], len(i)); mess.append(("NPS outside 0-10", len(i)))
i = pick(.01); pl.loc[i, "hours_to_first_contact"] = 999; mess.append(("Contact time placeholder 999", len(i)))
i = pick(.01); pl.loc[i, "acquisition_cost_inr"] = np.nan; mess.append(("Acquisition cost missing", len(i)))
sub_m = pl.refill_subscribed == "Yes"
i = pick(.01, sub_m); pl.loc[i, "months_active"] = -1; mess.append(("Negative months active", len(i)))
dsrc = pick(.03)
dup = pl.loc[dsrc].copy(); dup["lead_id"] = [f"L{NL + 1 + k:05d}" for k in range(len(dup))]
pl = pd.concat([pl, dup], ignore_index=True); mess.append(("Duplicate lead (same phone, entered twice)", len(dup)))
pl = pl.sample(frac=1, random_state=3).reset_index(drop=True)
pl.to_csv("data/raw/pipeline_raw.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(noise_log + [("--- pipeline ---", "")] + mess, columns=["issue", "rows"]).to_csv("data/raw/planted_noise_log.csv", index=False)
print("pipeline_raw.csv:", pl.shape)
print("purchase rate (true):", round(purchased.mean(), 3), "subscribed share of buyers:", round(subscribed.sum() / purchased.sum(), 3))

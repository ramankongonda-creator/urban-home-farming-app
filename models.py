"""
models.py - PREDICTIVE and PRESCRIPTIVE analytics: the four algorithms + break-even model
=========================================================================================
1. Classification  - predict who will buy (survey intent model) and which pilot lead will convert (lead scoring)
2. Clustering      - segment respondents into high / medium / low conversion-probability groups (k-means)
3. Association     - which features are wanted together -> which features to launch first (Apriori)
4. Regression      - how much a customer is willing to pay (OLS + random-forest check)
5. Prescriptive    - break-even and payback under the pilot as-is vs the data-driven strategy

All functions take the CLEAN data (data/clean/*.csv) and return plain pandas objects, used by
04_predictive_prescriptive.ipynb and the Streamlit app. Data is SYNTHETIC.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import analytics as A

SEED = 42
ATT = ["att_health", "att_eco", "att_tech", "att_time_poor", "att_hobby"]
FEAT = list(A.FEATURES)
CROP = {"g_herbs": "Herbs", "g_leafy": "Leafy greens", "g_micro": "Microgreens", "g_tomato": "Tomatoes & chillies", "g_exotic": "Exotic greens"}

# ------------------------------------------------------------------ which column feeds which algorithm (and why)
COLUMN_MAP = pd.DataFrame([
    ("Classification (who will buy)", "Target: would_buy (Q15 top-2-box). Inputs: income, income_disclosed, household_size, children, space_level, "
     "sun_level, veg_spend, spoil_level, att_health/eco/tech/time_poor/hobby, experience_level, price_shown",
     "Profile + attitude + the price they saw. Max-price and subscription answers are left out: a new lead has not given them yet."),
    ("Classification (lead scoring, pilot)", "Target: purchased. Inputs: channel, hours_to_first_contact, price_offered, space_level, income_level, "
     "household_size, demo_type, sales_rep, city",
     "Everything the sales team knows about a lead before the close."),
    ("Clustering (segments)", "att_health/eco/tech/time_poor/hobby, space_level, sun_level, income, veg_spend, household_size, children, experience_level",
     "Who they are and how they live - not the outcome. Outcomes (would_buy, max price) are used afterwards to profile and rank the segments."),
    ("Association rules (feature bundles)", "f_* (9 features wanted, Q12) and g_* (5 crops, Q13) as 0/1 baskets",
     "Tick-box answers are baskets: 'people who want A and B also want C' decides which features launch together."),
    ("Regression (willingness to pay)", "Target: wtp_system (Q16a, ₹). Inputs: income, space_level, sun_level, veg_spend, att_*, experience_level, "
     "household_size, children", "A rupee amount is continuous, so regression (not classification) predicts it."),
], columns=["Algorithm", "Columns used", "Why these columns"])

CLS_X = ["income", "income_disclosed", "household_size", "children", "space_level", "sun_level", "veg_spend", "spoil_level"] + ATT + \
        ["experience_level", "price_shown"]
NICE = {**A.NICE, "income_disclosed": "Income disclosed", "price_shown": "Price shown"}


def _prep(s):
    d = s.copy()
    d["income"] = d.income.fillna(d.income.median())          # 'prefer not to say' -> median band + flag column
    return d


def _metrics(name, y, prob, thr=.5):
    pred = (prob >= thr).astype(int)
    return dict(model=name, accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, prob), cm=confusion_matrix(y, pred))


# ================================================================== 1. CLASSIFICATION
def classification(s):
    d = _prep(s)
    X, y = d[CLS_X], d.would_buy
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    models = {"Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
              "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=SEED),
              "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1)}
    rows, fitted = [], {}
    for name, m in models.items():
        lin = name.startswith("Logistic")
        m.fit(sc.transform(Xtr) if lin else Xtr, ytr)
        prob = m.predict_proba(sc.transform(Xte) if lin else Xte)[:, 1]
        cv = cross_val_score(m, sc.transform(X) if lin else X, y, cv=5, scoring="roc_auc")
        r = _metrics(name, yte, prob); r["cv_auc"] = cv.mean(); rows.append(r); fitted[name] = m
    res = pd.DataFrame(rows)
    lr = fitted["Logistic regression"]
    odds = pd.DataFrame({"factor": [NICE.get(c, c) for c in CLS_X], "coef": lr.coef_[0]})
    odds["odds_ratio_per_sd"] = np.exp(odds.coef)
    odds = odds.sort_values("coef")
    rf = fitted["Random forest"]
    imp = pd.DataFrame({"factor": [NICE.get(c, c) for c in CLS_X], "importance": rf.feature_importances_}).sort_values("importance")
    best = res.sort_values("auc", ascending=False).iloc[0]
    return dict(results=res, odds=odds, importance=imp, best=best, n_test=len(yte), base_rate=y.mean())


def lead_scoring(p):
    d = p[p.contacted == 1].copy()
    d["in_person"] = (d.demo_type == "In person").astype(int)
    d["income_level"] = d.income_level.fillna(d.income_level.median())
    X = pd.get_dummies(d[["channel", "sales_rep", "city"]], drop_first=True).astype(int)
    X = pd.concat([X, d[["hours_to_first_contact", "price_offered", "space_level", "income_level", "household_size", "in_person"]]], axis=1)
    y = d.purchased.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    rows = [_metrics("Logistic regression", yte, lr.predict_proba(sc.transform(Xte))[:, 1]),
            _metrics("Random forest", yte, rf.predict_proba(Xte)[:, 1])]
    prob = lr.predict_proba(sc.transform(X))[:, 1]
    d["score"] = prob
    d["decile"] = pd.qcut(d.score.rank(method="first"), 10, labels=range(10, 0, -1)).astype(int)   # 1 = best 10%
    gains = d.groupby("decile").agg(leads=("purchased", "size"), buyers=("purchased", "sum")).sort_index()
    gains["conversion"] = gains.buyers / gains.leads
    gains["cum_share_of_buyers"] = gains.buyers.cumsum() / gains.buyers.sum()
    return dict(results=pd.DataFrame(rows), gains=gains.reset_index(), n_test=len(yte), base_rate=y.mean())


# ================================================================== 2. CLUSTERING
CLU_X = ATT + ["space_level", "sun_level", "income", "veg_spend", "household_size", "children", "experience_level"]


def clustering(s, k=None):
    d = _prep(s)
    Z = StandardScaler().fit_transform(d[CLU_X])
    scan = []
    for kk in range(2, 9):
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(Z)
        scan.append(dict(k=kk, inertia=km.inertia_, silhouette=silhouette_score(Z, km.labels_, sample_size=2000, random_state=SEED)))
    scan = pd.DataFrame(scan)
    k = k or 4
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
    d["cluster"] = km.labels_
    prof = d.groupby("cluster").agg(size=("cluster", "size"), would_buy=("would_buy", "mean"), would_subscribe=("would_subscribe", "mean"),
                                   median_wtp=("wtp_system", "median"), income=("income", "mean"), space=("space_level", "mean"),
                                   health=("att_health", "mean"), eco=("att_eco", "mean"), tech=("att_tech", "mean"),
                                   time_poor=("att_time_poor", "mean"), hobby=("att_hobby", "mean"), experience=("experience_level", "mean"),
                                   metro=("city_tier", lambda x: (x == 1).mean()))
    prof["share"] = prof["size"] / len(d)
    prof = prof.sort_values("would_buy", ascending=False)
    tiers = ["High", "Medium", "Medium", "Low", "Low", "Low", "Low", "Low"]
    prof["probability_tier"] = tiers[:len(prof)]
    prof["name"] = [_name(r) for r in prof.itertuples()]
    d["segment"] = d.cluster.map(prof.name)
    return dict(scan=scan, profile=prof.reset_index(), labels=d[["response_id", "cluster", "segment"]], k=k,
                silhouette=float(scan.set_index("k").silhouette[k]))


def _name(r):
    if r.space >= 2 and r.income >= 3.2:
        return "Affluent space-rich growers"
    if r.hobby >= 3.8 and r.space >= 1.8:
        return "Hobby terrace gardeners"
    if r.time_poor >= 3.7 and r.tech >= 3.6:
        return "Busy tech-savvy professionals"
    if r.income <= 2.2:
        return "Budget-conscious households"
    if r.health >= 3.9:
        return "Health-first families"
    return "Mainstream apartment dwellers"


# ================================================================== 3. ASSOCIATION RULES
def association(s, min_support=.06, min_conf=.45):
    names = {**A.FEATURES, **CROP}
    basket = s[FEAT + list(CROP)].astype(bool).rename(columns=names)
    items = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(items, metric="confidence", min_threshold=min_conf)
    rules = rules[rules.lift > 1.05].copy()
    rules["if"] = rules.antecedents.map(lambda x: " + ".join(sorted(x)))
    rules["then"] = rules.consequents.map(lambda x: " + ".join(sorted(x)))
    rules = rules[["if", "then", "support", "confidence", "lift"]].sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    feat_rules = rules[rules["if"].map(lambda t: all(x in A.FEATURES.values() for x in t.split(" + ")))
                       & rules.then.map(lambda t: all(x in A.FEATURES.values() for x in t.split(" + ")))]
    buyers = s[s.would_buy == 1]
    launch = pd.DataFrame({"feature": list(A.FEATURES.values()),
                           "wanted_by_all": [s[f].mean() for f in FEAT], "wanted_by_likely_buyers": [buyers[f].mean() for f in FEAT]})
    launch = launch.sort_values("wanted_by_likely_buyers", ascending=False).reset_index(drop=True)
    launch["phase"] = ["Launch"] * 4 + ["Phase 2 (month 6)"] * 3 + ["Phase 3 / add-on"] * (len(launch) - 7)
    return dict(rules=rules, feature_rules=feat_rules.reset_index(drop=True), n_itemsets=len(items), launch=launch)


# ================================================================== 4. REGRESSION
REG_X = ["income", "space_level", "sun_level", "veg_spend"] + ATT + ["experience_level", "household_size", "children"]


def regression(s):
    d = _prep(s).dropna(subset=["wtp_system"])
    X, y = d[REG_X], d.wtp_system
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=SEED)
    ols = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
    pred = ols.predict(sm.add_constant(Xte))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    rfp = rf.predict(Xte)
    perf = pd.DataFrame([dict(model="Linear regression (OLS)", r2_test=r2_score(yte, pred), mae=mean_absolute_error(yte, pred), r2_train=ols.rsquared),
                         dict(model="Random forest regressor", r2_test=r2_score(yte, rfp), mae=mean_absolute_error(yte, rfp),
                              r2_train=r2_score(ytr, rf.predict(Xtr)))])
    coef = pd.DataFrame({"factor": [NICE.get(c, c) for c in REG_X], "rupees_per_unit": ols.params[REG_X].values, "p_value": ols.pvalues[REG_X].values})
    coef["significant"] = coef.p_value < .05
    coef = coef.sort_values("rupees_per_unit")
    # outliers left in: what the cleaning step bought us
    raw = s.copy()
    return dict(perf=perf, coef=coef, r2=ols.rsquared, adj_r2=ols.rsquared_adj, n=len(d),
                actual_vs_pred=pd.DataFrame({"actual": yte.values, "predicted": pred.values}))


# ================================================================== 5. PRESCRIPTIVE - break-even
FIXED = dict(upfront=2_500_000, monthly_y1=350_000, monthly_y2=600_000)   # assumptions (₹): tooling+launch; team, rent, software


def scenario_inputs(p):
    sub = p[p.refill_subscribed == 1]
    kit = A.unit_economics(p).mean()
    good = p[p.channel.isin(["Housing-society demo", "Referral"])]
    return {
        "Pilot as-is": dict(cac=p.acquisition_cost_inr.sum() / p.purchased.sum(), kit_margin=kit,
                            attach=p[p.purchased == 1].refill_subscribed.mean(), churn=A.churn_rate(sub), buyers_m1=80),
        "Data-driven strategy": dict(cac=good.acquisition_cost_inr.sum() / good.purchased.sum(), kit_margin=kit,
                                     attach=min(.55, p[p.purchased == 1].refill_subscribed.mean() + .10),
                                     churn=A.churn_rate(sub[sub.fast_first_harvest == 1]), buyers_m1=80),
    }


def breakeven(inp, months=48, growth=.06, cap=600):
    rows, subs, cum = [], 0.0, -FIXED["upfront"]
    for m in range(1, months + 1):
        new = min(inp["buyers_m1"] * (1 + growth) ** (m - 1), cap)
        subs = subs * (1 - inp["churn"]) + new * inp["attach"]
        fixed = FIXED["monthly_y1"] if m <= 12 else FIXED["monthly_y2"]
        contrib = new * (inp["kit_margin"] - inp["cac"]) + subs * A.REFILL_MARGIN
        profit = contrib - fixed
        cum += profit
        rows.append(dict(month=m, new_buyers=new, subscribers=subs, contribution=contrib, fixed=fixed, profit=profit, cumulative=cum))
    t = pd.DataFrame(rows)
    pos = t[t.profit > 0]
    be = int(pos.month.iloc[0]) if len(pos) else None
    pay = t[t.cumulative >= 0]
    pb = int(pay.month.iloc[0]) if len(pay) else None
    return t, be, pb


def prescriptive(s, p):
    out = {}
    for name, inp in scenario_inputs(p).items():
        t, be, pb = breakeven(inp)
        out[name] = dict(inputs=inp, table=t, breakeven=be, payback=pb, min_cash=t.cumulative.min())
    return out

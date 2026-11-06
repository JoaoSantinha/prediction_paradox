"""
Simulation behind Figure 2 of "The prediction-intervention paradox in clinical AI".

Setting, loosely modelled on QRISK3-style primary prevention:
  - outcome is a major adverse cardiovascular event (MACE) within 10 years
  - statins are started when predicted risk reaches 10%
  - statins cut risk by 30% in relative terms, the same for every patient
  - the development cohort comes from standard care, where some patients are
    already on statins, so its outcomes are not untreated risk

Writes figure2_all.svg, one SVG per panel (figure2_panelA.svg ... F.svg)
and figure2_numbers.json.
"""

import json

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(20260802)

# -----------------------------------------------------------------------------
# How the data are generated
# -----------------------------------------------------------------------------
RRR = 0.30              # relative risk reduction from a statin
THRESH = 0.10           # predicted 10-year risk at which guidelines say treat
ADHERE_TREAT = 0.85     # chance of a statin when the model says treat
ADHERE_NOTREAT = 0.05   # chance of a statin when the model says don't; keeps overlap
N_DEV = 80_000
N_MONTH = 2_500
MONTHS_PRE, MONTHS_POST = 24, 36
DRIFT_MONTH = 25        # month at which baseline risk really rises (not caused by the model)
DRIFT_FACTOR = 1.25


def draw_X(n):
    age = rng.normal(0, 1, n)
    sex = rng.binomial(1, 0.5, n)
    smoke = rng.binomial(1, 0.20, n)
    sbp = rng.normal(0, 1, n)
    chol = rng.normal(0, 1, n)
    diab = rng.binomial(1, 0.09, n)
    return np.column_stack([age, sex, smoke, sbp, chol, diab])


def untreated_risk(X, drift=1.0):
    lp = (-3.05 + 0.95 * X[:, 0] + 0.35 * X[:, 1] + 0.45 * X[:, 2]
          + 0.30 * X[:, 3] + 0.25 * X[:, 4] + 0.60 * X[:, 5])
    return np.clip(expit(lp) * drift, 1e-6, 0.999)


def standard_care_propensity(X):
    """Probability of a statin under usual care; rises with risk but targets it loosely."""
    lp = (-1.15 + 0.75 * X[:, 0] + 0.30 * X[:, 2] + 0.35 * X[:, 3]
          + 0.20 * X[:, 4] + 0.60 * X[:, 5])
    return expit(lp)


def outcome(p0, A):
    return rng.binomial(1, p0 * (1 - RRR * A))


# -----------------------------------------------------------------------------
# Stage 1: model development under standard care
# -----------------------------------------------------------------------------
X_dev = draw_X(N_DEV)
p0_dev = untreated_risk(X_dev)
e_sc_dev = standard_care_propensity(X_dev)
A_dev = rng.binomial(1, e_sc_dev)
Y_dev = outcome(p0_dev, A_dev)

# Fit the deployed model to the outcomes seen under standard care. What it
# estimates is risk under that treatment policy, not risk without treatment.
model = LogisticRegression(max_iter=2000).fit(X_dev, Y_dev)
# Treatment model for standard care. It serves twice: to undo treatment in the
# hypothetical-strategy prediction, and as the target policy in off-policy evaluation.
ps_sc = LogisticRegression(max_iter=2000).fit(X_dev, A_dev)


def R_hat(X):
    """Risk predicted by the model as developed, i.e. under standard care."""
    return model.predict_proba(X)[:, 1]


def R_hat_untreated(X):
    """Hypothetical-strategy prediction: risk if no one were given a statin.
    Since R_sc(X) = p0(X) * (1 - RRR * e_sc(X)), we recover p0(X) = R_sc(X) / (1 - RRR * e_sc(X))."""
    return np.clip(R_hat(X) / (1 - RRR * ps_sc.predict_proba(X)[:, 1]), 1e-6, 0.999)


# -----------------------------------------------------------------------------
# One cohort per month, before and after the model goes live
# -----------------------------------------------------------------------------
records = []
for m in range(-MONTHS_PRE + 1, MONTHS_POST + 1):
    drift = DRIFT_FACTOR if m >= DRIFT_MONTH else 1.0
    X = draw_X(N_MONTH)
    p0 = untreated_risk(X, drift)
    r = R_hat(X)
    r0 = R_hat_untreated(X)
    if m <= 0:                                   # usual care still applies
        e_real = standard_care_propensity(X)
    else:                                        # treatment now follows the model
        rec = (r >= THRESH).astype(float)
        e_real = ADHERE_TREAT * rec + ADHERE_NOTREAT * (1 - rec)
    A = rng.binomial(1, e_real)
    Y = outcome(p0, A)
    records.append(dict(month=m, X=X, p0=p0, r=r, r0=r0, A=A, Y=Y,
                        e_sc=standard_care_propensity(X), e_real=e_real))

pre = [d for d in records if d["month"] <= 0]
post = [d for d in records if d["month"] > 0]
post_stable = [d for d in post if d["month"] < DRIFT_MONTH]


def stack(ds, key):
    return np.concatenate([d[key] for d in ds]) if ds[0][key].ndim == 1 else np.vstack([d[key] for d in ds])


# -----------------------------------------------------------------------------
# True policy values in the population (known exactly because we simulate)
# -----------------------------------------------------------------------------
X_pop = draw_X(400_000)
p0_pop = untreated_risk(X_pop)
e_sc_pop = standard_care_propensity(X_pop)
rec_pop = (R_hat(X_pop) >= THRESH).astype(float)
e_model_pop = ADHERE_TREAT * rec_pop + ADHERE_NOTREAT * (1 - rec_pop)

V_sc = float(np.mean(p0_pop * (1 - RRR * e_sc_pop)))
V_model = float(np.mean(p0_pop * (1 - RRR * e_model_pop)))
V_untreated = float(np.mean(p0_pop))
delta_V = V_model - V_sc

hi = rec_pop == 1
V_sc_hi = float(np.mean((p0_pop * (1 - RRR * e_sc_pop))[hi]))
V_model_hi = float(np.mean((p0_pop * (1 - RRR * e_model_pop))[hi]))
V_sc_lo = float(np.mean((p0_pop * (1 - RRR * e_sc_pop))[~hi]))
V_model_lo = float(np.mean((p0_pop * (1 - RRR * e_model_pop))[~hi]))

# share treated in each decile of predicted risk, across the whole population
r_pop = R_hat(X_pop)
edges_pop = np.quantile(r_pop, np.linspace(0, 1, 11))
alloc_x, alloc_sc, alloc_m = [], [], []
for lo_, hi_ in zip(edges_pop[:-1], edges_pop[1:]):
    m = (r_pop >= lo_) & (r_pop <= hi_)
    alloc_x.append(r_pop[m].mean())
    alloc_sc.append(e_sc_pop[m].mean())
    alloc_m.append(e_model_pop[m].mean())
alloc_x, alloc_sc, alloc_m = map(np.array, (alloc_x, alloc_sc, alloc_m))

# -----------------------------------------------------------------------------
# Doubly robust off-policy estimate of the comparator V(pi_sc), from
# post-deployment data alone. Standard care is no longer being delivered, so
# this value cannot be observed directly and has to be estimated.
# -----------------------------------------------------------------------------
def dr_offpolicy(d_list):
    X = stack(d_list, "X"); A = stack(d_list, "A"); Y = stack(d_list, "Y")
    r = stack(d_list, "r"); e_sc = stack(d_list, "e_sc")
    lr = logit(np.clip(r, 1e-4, 1 - 1e-4))
    rec = (r >= THRESH).astype(float)
    feats = np.column_stack([X, lr, lr ** 2])
    e_post = np.clip(LogisticRegression(max_iter=3000).fit(
        np.column_stack([feats, rec]), A).predict_proba(np.column_stack([feats, rec]))[:, 1], .02, .98)
    m1 = LogisticRegression(max_iter=3000).fit(feats[A == 1], Y[A == 1])
    m0 = LogisticRegression(max_iter=3000).fit(feats[A == 0], Y[A == 0])
    mu1 = m1.predict_proba(feats)[:, 1]
    mu0 = m0.predict_proba(feats)[:, 1]
    mu_obs = np.where(A == 1, mu1, mu0)
    w = np.where(A == 1, e_sc / e_post, (1 - e_sc) / (1 - e_post))
    psi = e_sc * mu1 + (1 - e_sc) * mu0 + w * (Y - mu_obs)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


dr_est, dr_se = dr_offpolicy(post_stable)

# month-by-month series -------------------------------------------------------
months = np.array([d["month"] for d in records])
obs_rate = np.array([d["Y"].mean() for d in records])
obs_se = np.array([d["Y"].std(ddof=1) / np.sqrt(len(d["Y"])) for d in records])
pred_mean = np.array([d["r"].mean() for d in records])

# DR estimate of V(pi_sc) over a rolling 6-month window, after go-live only
dr_months, dr_vals, dr_ses = [], [], []
for i, d in enumerate(records):
    if d["month"] <= 0:
        continue
    win = [records[j] for j in range(max(0, i - 5), i + 1) if records[j]["month"] > 0]
    if len(win) < 3:
        continue
    v, s = dr_offpolicy(win)
    dr_months.append(d["month"]); dr_vals.append(v); dr_ses.append(s)
dr_months, dr_vals, dr_ses = map(np.array, (dr_months, dr_vals, dr_ses))

# checking whether the predictions still hold ---------------------------------
# naive O/E: observed event rate over mean predicted risk
# counterfactual O/E: IPW estimate of E[Y(0)] from untreated patients (weight
#                     1/P(A=0|X)), over the mean hypothetical-strategy prediction
naive_oe, cf_oe, cf_lo, cf_hi = [], [], [], []
for d in records:
    naive_oe.append(d["Y"].mean() / d["r"].mean())
    w = (1 - d["A"]) / np.clip(1 - d["e_real"], 0.05, 1)
    ey0 = np.sum(w * d["Y"]) / np.sum(w)
    se = np.sqrt(np.sum((w * (d["Y"] - ey0)) ** 2)) / np.sum(w)
    cf_oe.append(ey0 / d["r0"].mean())
    cf_lo.append((ey0 - 1.96 * se) / d["r0"].mean())
    cf_hi.append((ey0 + 1.96 * se) / d["r0"].mean())
naive_oe, cf_oe, cf_lo, cf_hi = map(np.array, (naive_oe, cf_oe, cf_lo, cf_hi))

# calibration across deciles of predicted risk --------------------------------
def calib(d_list, target="obs"):
    r = stack(d_list, "r"); Y = stack(d_list, "Y"); p0 = stack(d_list, "p0")
    edges = np.quantile(r, np.linspace(0, 1, 11))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r <= hi)
        xs.append(r[m].mean())
        ys.append(Y[m].mean() if target == "obs" else p0[m].mean())
    return np.array(xs), np.array(ys)


cal_pre_x, cal_pre_y = calib(pre, "obs")
_, cal_pre_y0 = calib(pre, "p0")
cal_post_x, cal_post_y = calib(post_stable, "obs")

# event rates among patients predicted to be high risk ------------------------
def high_risk_rate(d_list):
    r = stack(d_list, "r"); Y = stack(d_list, "Y")
    m = r >= THRESH
    y = Y[m]
    return y.mean(), y.std(ddof=1) / np.sqrt(len(y)), m.mean()


hr_pre, hr_pre_se, hr_frac = high_risk_rate(pre)
hr_post, hr_post_se, _ = high_risk_rate(post_stable)
hr_pred = float(stack(pre, "r")[stack(pre, "r") >= THRESH].mean())

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 6.8, "figure.dpi": 300,
    "svg.fonttype": "none",
})
C_OBS, C_PRED, C_CF, C_SC = "#2166ac", "#7f7f7f", "#1a9850", "#d6604d"

fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.6))
(axA, axB), (axC, axD), (axE, axF) = axes

# A ---------------------------------------------------------------------------
axA.plot([0, .35], [0, .35], color="0.75", lw=.8, ls="--")
axA.plot(cal_pre_x, cal_pre_y, "o-", ms=3.4, color=C_OBS, label="Observed under standard care")
axA.plot(cal_pre_x, cal_pre_y0, "s-", ms=3.4, color=C_CF, label="Untreated risk $Y(0)$")
axA.set(xlabel="Predicted risk", ylabel="Event rate", title="A  Development data (standard care)")
axA.legend(loc="upper left")

# B ---------------------------------------------------------------------------
axB.plot([0, .35], [0, .35], color="0.75", lw=.8, ls="--")
axB.plot(cal_post_x, cal_post_y, "o-", ms=3.4, color=C_OBS)
axB.axvline(THRESH, color="0.6", lw=.7, ls=":")
axB.annotate("treatment intensified\n(observed < predicted)", xy=(cal_post_x[-2], cal_post_y[-2]),
             xytext=(0.12, 0.055), fontsize=6.5, color="0.25",
             arrowprops=dict(arrowstyle="->", lw=.6, color="0.5"))
axB.annotate("treatment withdrawn\n(observed > predicted)", xy=(cal_post_x[2], cal_post_y[2]),
             xytext=(0.005, 0.20), fontsize=6.5, color="0.25",
             arrowprops=dict(arrowstyle="->", lw=.6, color="0.5"))
axB.set(xlabel="Predicted risk", ylabel="Observed event rate",
        title="B  Post-deployment (model-guided policy)")

# C ---------------------------------------------------------------------------
axC.plot(alloc_x, alloc_sc, "s-", ms=3.4, color=C_SC, label=r"Standard care $\pi_{\rm sc}$")
axC.plot(alloc_x, alloc_m, "o-", ms=3.4, color=C_OBS, label=r"Model-guided $\pi_{M}$")
axC.axvline(THRESH, color="0.6", lw=.7, ls=":")
axC.text(THRESH * 1.06, 0.92, "decision\nthreshold", fontsize=6.2, color="0.45", va="top")
axC.set(xlabel="Predicted risk", ylabel="Proportion treated", ylim=(0, 1),
        title="C  Treatment allocation under each policy")
axC.legend(loc="center left")

# D ---------------------------------------------------------------------------
xs = np.arange(2)
axD.bar(xs - .17, [V_sc, V_sc_hi], width=.32, color=C_SC, label=r"$V(\pi_{\rm sc})$")
axD.bar(xs + .17, [V_model, V_model_hi], width=.32, color=C_OBS, label=r"$V(\pi_{M})$")
axD.set_xticks(xs)
axD.set_xticklabels([f"Whole population", f"Predicted high risk\n({hr_frac*100:.0f}% of cohort)"])
axD.set(ylabel="Expected event rate", title="D  Policy value")
for x, a, b in zip(xs, [V_sc, V_sc_hi], [V_model, V_model_hi]):
    axD.text(x - .17, a + .002, f"{a*100:.1f}%", ha="center", fontsize=6.8)
    axD.text(x + .17, b + .002, f"{b*100:.1f}%", ha="center", fontsize=6.8)
    axD.text(x, max(a, b) + .014, f"$\\Delta$ {100*(b-a):+.2f} pp", ha="center",
             fontsize=6.6, color="0.25")
axD.set_ylim(0, V_sc_hi * 1.25)
axD.legend(loc="upper left")

# E ---------------------------------------------------------------------------
axE.axvspan(0.5, MONTHS_POST + .5, color="0.94")
axE.errorbar(months, obs_rate, yerr=1.96 * obs_se, fmt="o-", ms=2.2, lw=.9,
             color=C_OBS, ecolor="0.8", label="Observed outcome under policy in force")
axE.plot(dr_months, dr_vals, "-", color=C_SC, lw=1.2, label=r"DR off-policy estimate of $V(\pi_{\rm sc})$")
axE.fill_between(dr_months, dr_vals - 1.96 * dr_ses, dr_vals + 1.96 * dr_ses,
                 color=C_SC, alpha=.20, lw=0)
axE.axvline(0.5, color="k", lw=.8)
axE.axvline(DRIFT_MONTH - .5, color="0.5", lw=.8, ls=":")
axE.text(DRIFT_MONTH - .3, axE.get_ylim()[1] * .99, " genuine risk drift", fontsize=6.2, color="0.4", va="top")
axE.set(xlabel="Month relative to deployment", ylabel="Event rate",
        title="E  Policy-value monitoring")
axE.legend(loc="lower left")

# F ---------------------------------------------------------------------------
axF.axvspan(0.5, MONTHS_POST + .5, color="0.94")
axF.axhline(1, color="0.75", lw=.8, ls="--")
axF.plot(months, naive_oe, "o-", ms=2.2, lw=.9, color=C_PRED,
         label="Naive O/E (observed vs predicted)")
axF.plot(months, cf_oe, "o-", ms=2.2, lw=.9, color=C_CF,
         label=r"Counterfactual O/E against $Y(0)$ (IPW)")
axF.fill_between(months, cf_lo, cf_hi, color=C_CF, alpha=.18, lw=0)
axF.axvline(0.5, color="k", lw=.8)
axF.axvline(DRIFT_MONTH - .5, color="0.5", lw=.8, ls=":")
axF.set(xlabel="Month relative to deployment", ylabel="Observed / expected",
        title="F  Prediction-validity monitoring")
axF.legend(loc="lower left")

fig.tight_layout(pad=0.9)
fig.savefig("figure2_all.svg", dpi=300, bbox_inches="tight")

renderer = fig.canvas.get_renderer()
for ax, name in zip([axA, axB, axC, axD, axE, axF], list("ABCDEF")):
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"figure2_panel{name}.svg", bbox_inches=bbox, dpi=300)

# -----------------------------------------------------------------------------
numbers = dict(
    threshold=THRESH, RRR=RRR, adherence=[ADHERE_TREAT, ADHERE_NOTREAT],
    high_risk_fraction=round(float(hr_frac), 4),
    mean_untreated_risk=round(V_untreated, 4),
    V_standard_care=round(V_sc, 4), V_model=round(V_model, 4),
    delta_V=round(delta_V, 4), relative_reduction=round(delta_V / V_sc, 4),
    high_risk_rate_pre=round(float(hr_pre), 4), high_risk_rate_post=round(float(hr_post), 4),
    V_sc_highrisk=round(V_sc_hi, 4), V_model_highrisk=round(V_model_hi, 4),
    V_sc_lowrisk=round(V_sc_lo, 4), V_model_lowrisk=round(V_model_lo, 4),
    treated_fraction_sc=round(float(e_sc_pop.mean()), 3),
    treated_fraction_model=round(float(e_model_pop.mean()), 3),
    high_risk_predicted=round(hr_pred, 4),
    DR_estimate_V_sc_postdeployment=[round(dr_est, 4), round(dr_se, 5)],
    naive_OE_pre=round(float(naive_oe[:MONTHS_PRE].mean()), 3),
    naive_OE_post_stable=round(float(naive_oe[MONTHS_PRE:MONTHS_PRE + DRIFT_MONTH - 1].mean()), 3),
    naive_OE_after_drift=round(float(naive_oe[MONTHS_PRE + DRIFT_MONTH - 1:].mean()), 3),
    cf_OE_post_stable=round(float(cf_oe[MONTHS_PRE:MONTHS_PRE + DRIFT_MONTH - 1].mean()), 3),
    cf_OE_after_drift=round(float(cf_oe[MONTHS_PRE + DRIFT_MONTH - 1:].mean()), 3),
    low_risk_obs_vs_pred=[round(float(cal_post_x[0]), 4), round(float(cal_post_y[0]), 4)],
)
with open("figure2_numbers.json", "w") as f:
    json.dump(numbers, f, indent=2)
print(json.dumps(numbers, indent=2))

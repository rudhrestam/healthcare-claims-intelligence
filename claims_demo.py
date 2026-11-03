"""
Healthcare Claims Intelligence — demo.

Ask anyone who has worked around claims processing where the money leaks
and they rarely say "the big dramatic fraud case". They say denials:
a missing prior authorisation here, a coding mismatch there, a claim
submitted so late or priced so far above the usual charge that it was
never going to sail through. Individually small, in aggregate a steady
percentage of everything billed — and a lot of it is preventable.

This demo builds that world out of synthetic data:

  1. Generate ~6,000 fake claims with plausible denial signals baked in.
  2. Clean the mess real extracts always contain (duplicates, negative
     amounts, missing codes) and report exactly what was dropped.
  3. Train a gradient-boosting classifier to predict denial, and —
     more usefully — rank *why* claims get denied, by driver and by
     category.
  4. Estimate what fixing the single most fixable driver (missing
     prior-auth) would be worth, labelled honestly as an estimate.

Everything is synthetic, the seed is fixed, and no real claims, patient,
or payer data is used anywhere.

Run:  python claims_demo.py
Writes a copy of the report to claims_report.txt next to this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
N_CLAIMS = 6_000
N_PROVIDERS = 30

# Typical allowed amount per procedure category (synthetic, illustrative).
TYPICAL_AMOUNTS = {
    "office visit": 140.0,
    "lab panel": 95.0,
    "imaging": 620.0,
    "physical therapy": 110.0,
    "preventive screening": 180.0,
    "behavioral health session": 150.0,
    "minor surgery": 1_450.0,
    "emergency visit": 1_150.0,
}
# Procedures payers usually want authorised before the fact.
AUTH_REQUIRED = {"imaging", "minor surgery", "emergency visit"}
DIAGNOSES = [
    "musculoskeletal", "respiratory", "cardiovascular", "metabolic",
    "preventive", "mental health", "injury", "digestive",
]
PROVIDER_TYPES = ["clinic", "hospital", "lab", "imaging center"]
AGE_BANDS = ["0-17", "18-34", "35-49", "50-64", "65+"]

# Planted data-quality problems — the cleaning step has to find these.
N_DUPLICATES = 50
N_NEGATIVE_AMOUNTS = 30
N_MISSING_CODES = 40

# Share of missing-auth denied dollars assumed recoverable if the auth
# workflow were fixed. An assumption for the estimate, not a finding.
AUTH_FIX_RATE = 0.65


def generate_claims(rng):
    """Build the synthetic claims table, corruptions included."""
    procedures = list(TYPICAL_AMOUNTS)
    proc = rng.choice(procedures, size=N_CLAIMS,
                      p=[0.26, 0.20, 0.12, 0.10, 0.10, 0.08, 0.07, 0.07])
    typical = np.array([TYPICAL_AMOUNTS[p] for p in proc])

    # Each provider has a latent "paperwork quality" — sloppy billers
    # generate more missing auths, more coding errors, and more history.
    # The history is observable in real life (last quarter's denial rate
    # per provider is a report anyone can pull), so it is kept as its own
    # column and used as a feature; the latent quality itself is not.
    provider_id = rng.integers(0, N_PROVIDERS, size=N_CLAIMS)
    provider_sloppiness = rng.uniform(0.0, 1.0, size=N_PROVIDERS)
    sloppy = provider_sloppiness[provider_id]
    provider_history = np.clip(
        0.07 + 0.28 * provider_sloppiness + rng.normal(0, 0.03, size=N_PROVIDERS),
        0.02, 0.50)

    provider_type = np.array([
        PROVIDER_TYPES[0] if p in ("office visit", "physical therapy",
                                   "behavioral health session")
        else PROVIDER_TYPES[2] if p == "lab panel"
        else PROVIDER_TYPES[3] if p == "imaging"
        else rng.choice(["clinic", "hospital"])
        for p in proc
    ])

    needs_auth = np.isin(proc, list(AUTH_REQUIRED))
    missing_auth = (rng.random(N_CLAIMS)
                    < np.where(needs_auth, 0.10 + 0.28 * sloppy, 0.03 + 0.08 * sloppy)
                    ).astype(int)
    coding_mismatch = (rng.random(N_CLAIMS) < 0.04 + 0.20 * sloppy).astype(int)

    amount_ratio = rng.lognormal(mean=0.05, sigma=0.28, size=N_CLAIMS)
    amount = np.round(typical * amount_ratio, 2)
    lag_days = np.clip(rng.normal(14 + 10 * sloppy, 9, size=N_CLAIMS), 1, 90).round(0)
    prior_denials = rng.poisson(0.4 + 1.1 * sloppy)

    # The denial label: a logit over the plausible signals, plus noise,
    # so there is a real pattern for the model to find — and a ceiling
    # on how well anything can predict it.
    logit = (-3.35
             + 2.30 * missing_auth
             + 1.95 * coding_mismatch
             + 0.65 * np.minimum(prior_denials, 3)
             + 1.50 * np.clip(amount_ratio - 1.25, 0, None)
             + 0.030 * np.clip(lag_days - 30, 0, None)
             + 0.60 * sloppy
             + rng.normal(0, 0.30, size=N_CLAIMS))
    denied = (rng.random(N_CLAIMS) < 1 / (1 + np.exp(-logit))).astype(int)

    # Paid claims get a payment lag; denied ones never reach payment.
    payment_days = np.where(
        denied == 0,
        np.clip(rng.normal(24, 8, size=N_CLAIMS)
                + np.where(provider_type == "hospital", 6, 0), 5, 75).round(0),
        np.nan,
    )

    df = pd.DataFrame({
        "claim_id": [f"C{i:05d}" for i in range(N_CLAIMS)],
        "claim_amount": amount,
        "procedure_category": proc,
        "diagnosis_category": rng.choice(DIAGNOSES, size=N_CLAIMS),
        "provider_type": provider_type,
        "provider_id": [f"P{p + 1:02d}" for p in provider_id],
        "provider_denial_history": np.round(provider_history[provider_id], 3),
        "member_age_band": rng.choice(AGE_BANDS, size=N_CLAIMS,
                                      p=[0.14, 0.24, 0.24, 0.22, 0.16]),
        "submission_lag_days": lag_days.astype(int),
        "prior_denials_count": prior_denials,
        "missing_prior_auth": missing_auth,
        "coding_mismatch": coding_mismatch,
        "amount_ratio": np.round(amount_ratio, 3),
        "payment_days": payment_days,
        "denied": denied,
    })

    # --- Plant the mess ---------------------------------------------------
    clean_idx = rng.choice(N_CLAIMS, size=N_NEGATIVE_AMOUNTS + N_MISSING_CODES,
                           replace=False)
    neg_idx, miss_idx = clean_idx[:N_NEGATIVE_AMOUNTS], clean_idx[N_NEGATIVE_AMOUNTS:]
    df.loc[neg_idx, "claim_amount"] = -df.loc[neg_idx, "claim_amount"]
    df.loc[miss_idx, "procedure_category"] = None

    dup_pool = np.setdiff1d(np.arange(N_CLAIMS), clean_idx)
    duplicates = df.iloc[rng.choice(dup_pool, size=N_DUPLICATES, replace=False)]
    return pd.concat([df, duplicates], ignore_index=True).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)


def clean_claims(raw):
    """Drop duplicates, impossible amounts and missing codes; count each."""
    counts = {}
    df = raw.copy()
    counts["duplicate_rows"] = int(df.duplicated(subset=["claim_id"]).sum())
    df = df.drop_duplicates(subset=["claim_id"], keep="first")
    counts["negative_amounts"] = int((df["claim_amount"] <= 0).sum())
    df = df[df["claim_amount"] > 0]
    counts["missing_codes"] = int(df["procedure_category"].isna().sum())
    df = df.dropna(subset=["procedure_category", "diagnosis_category"])
    counts["rows_dropped"] = len(raw) - len(df)
    return df.reset_index(drop=True), counts


def build_features(df):
    """Engineer the model features from the cleaned claims."""
    X = pd.DataFrame(index=df.index)
    X["claim_amount"] = df["claim_amount"]
    X["amount_ratio"] = df["amount_ratio"]
    X["high_amount_flag"] = (df["amount_ratio"] > 1.4).astype(int)
    X["submission_lag_days"] = df["submission_lag_days"]
    X["late_submission_flag"] = (df["submission_lag_days"] > 30).astype(int)
    X["prior_denials_count"] = df["prior_denials_count"]
    X["repeat_denial_flag"] = (df["prior_denials_count"] >= 2).astype(int)
    X["provider_denial_history"] = df["provider_denial_history"]
    X["missing_prior_auth"] = df["missing_prior_auth"]
    X["coding_mismatch"] = df["coding_mismatch"]
    for col in ["procedure_category", "diagnosis_category",
                "provider_type", "member_age_band"]:
        X = pd.concat([X, pd.get_dummies(df[col], prefix=col, dtype=float)],
                      axis=1)
    return X


# Base feature each one-hot column belongs to, for grouped importances.
def base_feature(column):
    for prefix in ("procedure_category", "diagnosis_category",
                   "provider_type", "member_age_band"):
        if column.startswith(prefix):
            return prefix
    return column


def denial_rate_block(df, mask, label):
    sub = df[mask]
    rest = df[~mask]
    return (f"  {label:<38}{sub['denied'].mean() * 100:>5.1f}% denied "
            f"(n={len(sub):,})  vs {rest['denied'].mean() * 100:.1f}% without")


def main():
    rng = np.random.default_rng(SEED)
    raw = generate_claims(rng)
    df, cleaning = clean_claims(raw)

    X = build_features(df)
    y = df["denied"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.06,
                                       max_depth=3, random_state=SEED)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)

    # How the model would actually be used: not as an auto-denier, but as
    # a pre-submission review queue. At a 0.30 threshold it flags about a
    # fifth of claims for a human check and catches over half the denials.
    queue_pred = (proba >= 0.30).astype(int)
    queue_precision = precision_score(y_test, queue_pred, zero_division=0)
    queue_recall = recall_score(y_test, queue_pred, zero_division=0)
    queue_share = queue_pred.mean()

    # Permutation importance on the held-out test set, not the model's
    # built-in impurity scores: impurity importance flatters noisy
    # high-cardinality columns, permutation shows what actually costs
    # ROC-AUC when a column is shuffled — the honest driver ranking.
    perm = permutation_importance(model, X_test, y_test, n_repeats=5,
                                  random_state=SEED, scoring="roc_auc")
    importances = pd.Series(perm.importances_mean, index=X.columns)
    grouped = importances.groupby(importances.index.map(base_feature)).sum()
    grouped = grouped[grouped > 0].sort_values(ascending=False)

    denied_df = df[df["denied"] == 1]
    paid_df = df[df["denied"] == 0]
    denial_rate = df["denied"].mean()
    auth_denied_amount = denied_df.loc[
        denied_df["missing_prior_auth"] == 1, "claim_amount"].sum()
    recoverable = auth_denied_amount * AUTH_FIX_RATE

    reason_masks = {
        "Missing prior-auth": denied_df["missing_prior_auth"] == 1,
        "Coding mismatch": denied_df["coding_mismatch"] == 1,
        "Amount well above typical (ratio > 1.4)":
            denied_df["amount_ratio"] > 1.4,
        "Late submission (lag > 30 days)":
            denied_df["submission_lag_days"] > 30,
        "Repeat denials (2+ prior)": denied_df["prior_denials_count"] >= 2,
    }

    lines = []
    lines.append("=" * 68)
    lines.append("HEALTHCARE CLAIMS INTELLIGENCE REPORT  (synthetic data)")
    lines.append("=" * 68)
    lines.append(f"Claims generated (raw):     {len(raw):>7,}")
    lines.append(f"Bad rows cleaned:           {cleaning['rows_dropped']:>7,}  "
                 f"(duplicates {cleaning['duplicate_rows']}, "
                 f"negative amounts {cleaning['negative_amounts']}, "
                 f"missing codes {cleaning['missing_codes']})")
    lines.append(f"Claims analysed:            {len(df):>7,}")
    lines.append(f"Total billed:               ${df['claim_amount'].sum():>10,.0f}")
    lines.append("")
    lines.append(f"Overall denial rate:        {denial_rate * 100:.1f}%  "
                 f"({int(df['denied'].sum()):,} of {len(df):,} claims, "
                 f"${denied_df['claim_amount'].sum():,.0f} denied)")
    lines.append(f"Avg days to payment (paid): {paid_df['payment_days'].mean():.1f} days")
    lines.append("")
    lines.append("Denial prediction (GradientBoosting, 80/20 stratified split):")
    lines.append(f"  Accuracy:  {accuracy:.3f}")
    lines.append(f"  ROC-AUC:   {auc:.3f}")
    lines.append(f"  Precision (denied class): {precision:.3f}")
    lines.append(f"  Recall    (denied class): {recall:.3f}")
    lines.append(f"  As a review queue (threshold 0.30): flags {queue_share * 100:.1f}% "
                 f"of claims, precision {queue_precision:.3f}, "
                 f"recall {queue_recall:.3f}")
    lines.append("")
    lines.append("Top denial drivers (permutation importance on test set,")
    lines.append("ROC-AUC lost when shuffled; categories grouped):")
    for name, imp in grouped.head(8).items():
        lines.append(f"  {name:<26}{imp:.3f}")
    lines.append("")
    lines.append("Denial rate by driver:")
    lines.append(denial_rate_block(df, df["missing_prior_auth"] == 1,
                                   "Missing prior-auth"))
    lines.append(denial_rate_block(df, df["coding_mismatch"] == 1,
                                   "Coding mismatch"))
    lines.append(denial_rate_block(df, df["amount_ratio"] > 1.4,
                                   "Amount ratio > 1.4x typical"))
    lines.append(denial_rate_block(df, df["submission_lag_days"] > 30,
                                   "Submission lag > 30 days"))
    lines.append(denial_rate_block(df, df["prior_denials_count"] >= 2,
                                   "2+ prior denials"))
    lines.append(denial_rate_block(df, df["provider_denial_history"] >= 0.20,
                                   "Provider history denial rate >= 20%"))
    lines.append("")
    lines.append("Denial rate by procedure category:")
    by_proc = df.groupby("procedure_category")["denied"].agg(["mean", "size"])
    for proc_name, row in by_proc.sort_values("mean", ascending=False).iterrows():
        lines.append(f"  {proc_name:<28}{row['mean'] * 100:>5.1f}%  (n={int(row['size']):,})")
    lines.append("")
    lines.append("Top rejection reasons (share of denied claims; reasons overlap):")
    for label, mask in sorted(reason_masks.items(),
                              key=lambda kv: kv[1].sum(), reverse=True):
        lines.append(f"  {label:<42}{mask.mean() * 100:>5.1f}%")
    lines.append("")
    lines.append("Estimated recoverable revenue — fix missing prior-auth:")
    lines.append(f"  Denied amount tied to missing prior-auth: ${auth_denied_amount:,.0f}")
    lines.append(f"  Assumed fix rate:                         {AUTH_FIX_RATE:.0%}")
    lines.append(f"  Estimated recoverable (ESTIMATE):         ${recoverable:,.0f}")
    lines.append("")
    lines.append("Read on the estimate: it assumes an auth check at scheduling /")
    lines.append("order entry prevents or cures about two-thirds of missing-auth")
    lines.append("denials. It is a sizing number for a workflow fix, not revenue")
    lines.append("anyone has booked — real recovery depends on timely filing")
    lines.append("limits, appeal success, and how much was ever collectable.")
    lines.append("=" * 68)

    report = "\n".join(lines)
    print(report)
    out = Path(__file__).with_name("claims_report.txt")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {out.name}")


if __name__ == "__main__":
    main()

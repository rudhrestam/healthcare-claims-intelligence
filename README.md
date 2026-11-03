# Healthcare Claims Intelligence (Demo)

Ask anyone who has worked around claims processing where the money leaks and they rarely say "the big dramatic fraud case". They say denials: a missing prior authorisation here, a coding mismatch there, a claim submitted late or priced far above the usual charge for the procedure. Individually small, in aggregate a steady slice of everything billed — and a large share of it is preventable, which is what makes it worth measuring properly. This project builds a small synthetic claims world, cleans it the way real extracts need cleaning, and uses ML to rank the denial drivers so a team can see which fix is worth doing first.

> This is a personal portfolio project exploring ML for claims-denial analysis. It is built on 100% synthetic data with original code, and does not represent work done for any employer. No real claims, patient, or payer data is used anywhere.

## Why claims get denied (in this demo)

The generator bakes in the causes anyone in revenue cycle will recognise. None of them alone dooms a claim — they stack:

| Driver | What it means here |
|---|---|
| **Missing prior-auth** | Procedures payers usually want authorised up front (imaging, minor surgery, emergency visits here) submitted without one. The single strongest denial signal in the data. |
| **Coding mismatch** | The procedure and diagnosis coding on the claim don't line up with each other — the classic avoidable denial. |
| **Provider denial history** | Some billers are simply sloppier than others. Each provider has a latent paperwork quality that drives their auth gaps, coding errors, repeat denials — and a historical denial rate the model is allowed to see, because in real life last quarter's rate is a report anyone can pull. |
| **Amount vs. typical** | Every claim carries its billed amount as a ratio to the typical charge for that procedure category. Claims priced far above typical draw scrutiny. |
| **Submission lag** | Days between service and submission. Late claims deny more, long before any timely-filing deadline formally kills them. |
| **Prior denials** | A claim from a member/provider combination with denials behind it is more likely to be denied again — history rhymes. |

## What the demo does

`claims_demo.py` generates 6,000 synthetic claims (procedure and diagnosis categories, provider type, member age band, amounts, lags, and the flags above), then works through four steps:

1. **Cleaning, with receipts** — real extracts arrive dirty, so the generator plants them: duplicate rows, negative amounts, missing procedure codes. The cleaning step drops each class and reports exactly how many of each it found, instead of silently tidying up.
2. **Denial prediction** — a GradientBoosting classifier (80/20 stratified split) predicts denial from the engineered features: the flags, amount-vs-typical ratio, late-submission and repeat-denial indicators, provider denial history, and one-hot categories.
3. **Driver ranking, done honestly** — drivers are ranked by *permutation importance on the held-out test set* (how much ROC-AUC drops when a column is shuffled), not the model's built-in impurity scores, which flatter noisy high-cardinality columns. The report also shows raw denial rates with and without each driver, so the ranking can be sanity-checked against the plain numbers.
4. **A "fix this first" estimate** — the report sizes the most fixable driver: denied dollars tied to missing prior-auth, multiplied by an assumed 65% fix rate. It is labelled an estimate in the output, because it is one — a sizing number for a workflow fix (an auth check at scheduling/order entry), not booked revenue.

The model is also reported the way it would actually be used: not as an auto-denier at a 0.5 threshold, but as a pre-submission review queue. At a 0.30 threshold it flags about a fifth of claims for a human check and catches over half of the eventual denials.

## Example output

Actual output from a run (fixed seed, so you'll get the same numbers):

```
====================================================================
HEALTHCARE CLAIMS INTELLIGENCE REPORT  (synthetic data)
====================================================================
Claims generated (raw):       6,050
Bad rows cleaned:               120  (duplicates 50, negative amounts 30, missing codes 40)
Claims analysed:              5,930
Total billed:               $ 2,281,924

Overall denial rate:        21.8%  (1,294 of 5,930 claims, $604,121 denied)
Avg days to payment (paid): 24.7 days

Denial prediction (GradientBoosting, 80/20 stratified split):
  Accuracy:  0.815
  ROC-AUC:   0.795
  Precision (denied class): 0.651
  Recall    (denied class): 0.324
  As a review queue (threshold 0.30): flags 21.5% of claims, precision 0.549, recall 0.541

Top denial drivers (permutation importance on test set,
ROC-AUC lost when shuffled; categories grouped):
  missing_prior_auth        0.102
  coding_mismatch           0.081
  prior_denials_count       0.046
  amount_ratio              0.017
  provider_denial_history   0.008
  repeat_denial_flag        0.004
  claim_amount              0.002
  submission_lag_days       0.002

Denial rate by driver:
  Missing prior-auth                     63.9% denied (n=695)  vs 16.2% without
  Coding mismatch                        49.3% denied (n=904)  vs 16.9% without
  Amount ratio > 1.4x typical            28.0% denied (n=951)  vs 20.6% without
  Submission lag > 30 days               28.0% denied (n=736)  vs 20.9% without
  2+ prior denials                       35.8% denied (n=1,592)  vs 16.7% without
  Provider history denial rate >= 20%    28.3% denied (n=3,540)  vs 12.2% without

Denial rate by procedure category:
  imaging                      29.3%  (n=713)
  minor surgery                27.8%  (n=410)
  emergency visit              25.4%  (n=394)
  physical therapy             22.6%  (n=603)
  behavioral health session    20.3%  (n=477)
  lab panel                    19.9%  (n=1,178)
  office visit                 19.4%  (n=1,585)
  preventive screening         17.0%  (n=570)

Top rejection reasons (share of denied claims; reasons overlap):
  Repeat denials (2+ prior)                  44.0%
  Coding mismatch                            34.5%
  Missing prior-auth                         34.3%
  Amount well above typical (ratio > 1.4)    20.6%
  Late submission (lag > 30 days)            15.9%

Estimated recoverable revenue — fix missing prior-auth:
  Denied amount tied to missing prior-auth: $306,836
  Assumed fix rate:                         65%
  Estimated recoverable (ESTIMATE):         $199,443
```

Two things worth noticing. First, the model's recall at the default 0.5 threshold is modest (0.324) even with a respectable ROC-AUC — most single-flag claims sit below 50% denial probability, they only cross it when signals stack. That is exactly why the review-queue framing matters more than the headline metrics. Second, the procedure ranking makes sense on its face: imaging, minor surgery, and emergency visits — the categories that need authorisation — deny most; preventive screening denies least.

## Run it

```bash
pip install -r requirements.txt
python claims_demo.py
```

Prints the claims report and saves a copy to `claims_report.txt`.

## Tech stack

Python · NumPy · pandas · scikit-learn (GradientBoosting, permutation importance, accuracy / ROC-AUC / precision / recall evaluation)

## Ideas for extension

- Model the appeals path, not just the first denial: some denials are overturned, and the recoverable-revenue estimate gets much more honest once appeal success by reason is in the data.
- Replace the single fixable-driver estimate with a small what-if table — auth checks, coding edits at submission, lag alerts — each with its own assumed fix rate and cost, so the ranking is by net value rather than gross dollars.
- Add a fairness/consistency check on the review queue: a flagging model in a regulated workflow should be auditable for which claim populations it sends to review disproportionately.
- Feed the same pipeline a de-identified real extract shape (837-style fields): the cleaning step and feature engineering are deliberately written the way they'd need to survive contact with one.

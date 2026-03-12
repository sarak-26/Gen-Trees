"""
Enriched Employee Dataset Generator  (v2 — corrected)
=======================================================
Builds a rich, internally consistent employee dataset on top of the IBM HR
Analytics Kaggle dataset. All engineered features are semantically coherent,
scale-correct, and free of contradictions with the base IBM columns.

Dependencies:
    pip install pandas numpy

Usage:
    1. Download the IBM HR Analytics dataset from Kaggle:
       https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    2. Place the CSV in the same directory as this script.
       Supported filenames: ibm_hr.csv  OR  WA_Fn-UseC_-HR-Employee-Attrition.csv
    3. Run: python generate_employee_dataset.py

Output:
    enriched_employee_dataset.csv

═══════════════════════════════════════════════════════════════════════════════
WHAT WAS WRONG IN v1 AND HOW IT IS FIXED HERE
═══════════════════════════════════════════════════════════════════════════════

FIX 1 — Contradictory salary columns
  IBM has daily_rate / hourly_rate / monthly_rate / monthly_income with no
  defined relationship between them. Kept only monthly_income as the salary
  anchor; salary_band is now BUCKETED from monthly_income (guaranteed consistent).

FIX 2 — attrition vs voluntary_exit_flag were computed independently
  voluntary_exit_flag is now simply the binarised IBM attrition column.
  attrition_risk_score is a continuous predictor of attrition, not a
  separate binary outcome.

FIX 3 — overtime_flag=No but overtime_hours > 0
  overtime_hours_per_month is strictly 0 when overtime_flag = No.

FIX 4 — performance_rating (IBM) and performance_score (derived) coexisted
  performance_rating_raw is used as input then DROPPED from output.
  performance_score is the single enriched measure.

FIX 5 — work_life_balance (IBM) and work_life_score (derived) coexisted
  work_life_balance_raw is used as input then DROPPED from output.
  work_life_score is the enriched replacement.

FIX 6 — skill_rating inputs summed to >5 before clipping (distribution pileup)
  All inputs are normalised to [0,1] before combining, then scaled to [1,5].

FIX 7 — seniority_level used OR-logic allowing Director at job_level=1
  Now uses a weighted score so job_level and tenure both matter proportionally.

FIX 8 — promotion_eligibility=1 with years_since_last_promotion=0
  Hard constraint: ineligible if promoted within the last 12 months.

FIX 9 — career_mobility_score direction was inverted and label was misleading
  Renamed career_advancement_score; direction and label are now coherent.

FIX 10 — bonus_pct ignored department (Sales gets higher variable pay)
  Department modifier added to bonus formula.

═══════════════════════════════════════════════════════════════════════════════
GROUND TRUTH DEPENDENCY GRAPH  (your generator evaluation rubric)
═══════════════════════════════════════════════════════════════════════════════

  monthly_income                                      --> salary_band
  job_level + total_working_years                     --> seniority_level
  performance_rating_raw + job_involvement + training --> performance_score
  performance_score + seniority_level + department    --> bonus_pct
  performance_score + years_since_last_promotion      --> promotion_eligibility
  overtime_flag + job_level                           --> overtime_hours_per_month
  overtime_hours + job_satisfaction
    + distance_from_home + environment_satisfaction   --> attrition_risk_score
  attrition_raw                                       --> voluntary_exit_flag
  years_at_company + training_times + education       --> skill_rating
  monthly_income + years_at_company + bonus_pct       --> total_annual_compensation
  department                                          --> remote_work_pct
  work_life_balance_raw + remote_work_pct
    + overtime_hours                                  --> work_life_score
  work_life_score + overtime_hours
    + environment_satisfaction                        --> burnout_index
  relationship_satisfaction + years_with_curr_manager --> manager_rating
  job_involvement + skill_rating                      --> peer_rating
  manager_rating + peer_rating + performance_score    --> overall_360_score
  department + seniority_level                        --> team_size
  promotion_eligibility + years_since_last_promotion
    + years_at_company                                --> career_advancement_score
"""

import os
import pandas as pd
import numpy as np

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

OUTPUT_FILE = "enriched_employee_dataset.csv"

# ── 1. Load ───────────────────────────────────────────────────────────────────
for candidate in ["ibm_hr.csv", "WA_Fn-UseC_-HR-Employee-Attrition.csv"]:
    if os.path.exists(candidate):
        INPUT_FILE = candidate
        break
else:
    raise FileNotFoundError(
        "IBM HR CSV not found. Download from Kaggle and place it here.\n"
        "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
    )

print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

# Drop zero-variance / ID columns
df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
        errors="ignore", inplace=True)

# Drop the three rate columns that have no defined relationship to monthly_income
# Keeping them would let generators learn spurious correlations
df.drop(columns=["DailyRate", "HourlyRate", "MonthlyRate"], errors="ignore", inplace=True)

RENAME_MAP = {
    "Age":                      "age",
    "Attrition":                "attrition_raw",
    "BusinessTravel":           "business_travel",
    "Department":               "department",
    "DistanceFromHome":         "distance_from_home_km",
    "Education":                "education_level",
    "EducationField":           "education_field",
    "EnvironmentSatisfaction":  "environment_satisfaction",
    "Gender":                   "gender",
    "JobInvolvement":           "job_involvement",
    "JobLevel":                 "job_level",
    "JobRole":                  "job_role",
    "JobSatisfaction":          "job_satisfaction",
    "MaritalStatus":            "marital_status",
    "MonthlyIncome":            "monthly_income",
    "NumCompaniesWorked":       "num_companies_worked",
    "OverTime":                 "overtime_flag",
    "PercentSalaryHike":        "pct_salary_hike",
    "PerformanceRating":        "performance_rating_raw",  # input only, dropped later
    "RelationshipSatisfaction": "relationship_satisfaction",
    "StockOptionLevel":         "stock_option_level",
    "TotalWorkingYears":        "total_working_years",
    "TrainingTimesLastYear":    "training_times_last_year",
    "WorkLifeBalance":          "work_life_balance_raw",   # input only, dropped later
    "YearsAtCompany":           "years_at_company",
    "YearsInCurrentRole":       "years_in_current_role",
    "YearsSinceLastPromotion":  "years_since_last_promotion",
    "YearsWithCurrManager":     "years_with_curr_manager",
}
df.rename(columns=RENAME_MAP, inplace=True)
n = len(df)
print(f"  {n} rows, {df.shape[1]} base columns.\n")

# ── helper: normalise a Series to [0, 1] ─────────────────────────────────────
def norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(np.zeros(len(s)), index=s.index)

# Convenience binary array for overtime (not written to output)
overtime_bin = (df["overtime_flag"] == "Yes").values


# ══════════════════════════════════════════════════════════════════════════════
# ENGINEERED FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# ── FIX 2: voluntary_exit_flag is binarised attrition_raw ────────────────────
df["voluntary_exit_flag"] = (df["attrition_raw"] == "Yes").astype(int)


# ── FIX 1: salary_band bucketed FROM monthly_income ──────────────────────────
df["salary_band"] = pd.qcut(
    df["monthly_income"], q=6,
    labels=["Band_1", "Band_2", "Band_3", "Band_4", "Band_5", "Band_6"]
).astype(str)
_band_ord = df["salary_band"].map(
    {"Band_1": 1, "Band_2": 2, "Band_3": 3, "Band_4": 4, "Band_5": 5, "Band_6": 6}
)


# ── FIX 7: seniority_level via weighted score (no OR-logic edge cases) ────────
seniority_score = (
    norm(df["job_level"])           * 0.60 +
    norm(df["total_working_years"]) * 0.40
)
df["seniority_level"] = pd.cut(
    seniority_score,
    bins=[0.0, 0.25, 0.55, 0.78, 1.01],
    labels=["Junior", "Mid", "Senior", "Director"],
    include_lowest=True
).astype(str)
_seniority_ord = df["seniority_level"].map(
    {"Junior": 0, "Mid": 1, "Senior": 2, "Director": 3}
)


# ── FIX 4: performance_score (normalised inputs → full [1,5] range) ──────────
df["performance_score"] = (
    norm(df["performance_rating_raw"])      * 0.50 +
    norm(df["job_involvement"])             * 0.30 +
    norm(df["training_times_last_year"])    * 0.20
) * 4 + 1
df["performance_score"] = df["performance_score"].clip(1, 5).round(2)
df.drop(columns=["performance_rating_raw"], inplace=True)  # input consumed


# ── FIX 10: bonus_pct with department modifier ────────────────────────────────
dept_bonus_base = {"Sales": 4.0, "Research & Development": 2.5, "Human Resources": 1.5}
df["bonus_pct"] = (
    norm(df["performance_score"]) * 8.0 +
    _seniority_ord                * 1.5 +
    df["department"].map(dept_bonus_base).fillna(2.5) +
    rng.normal(0, 0.5, n)
).clip(0, 20).round(2)


# ── FIX 8: promotion_eligibility with hard constraint ────────────────────────
promo_score = (
    norm(df["performance_score"])          * 0.65 -
    norm(df["years_since_last_promotion"]) * 0.35 +
    rng.normal(0, 0.08, n)
)
df["promotion_eligibility"] = (promo_score > 0.4).astype(int)
df.loc[df["years_since_last_promotion"] < 1, "promotion_eligibility"] = 0  # hard constraint


# ── FIX 3: overtime_hours_per_month strictly 0 when overtime_flag=No ─────────
ot_if_yes = (
    rng.integers(5, 25, n) +
    df["job_level"].values * rng.integers(0, 3, n)
).clip(0, 40)
df["overtime_hours_per_month"] = np.where(overtime_bin, ot_if_yes, 0).astype(int)


# ── attrition_risk_score (rescaled to span full [0, 5] range) ────────────────
raw_risk = (
    norm(df["overtime_hours_per_month"])     * 1.50 +
    norm(5 - df["job_satisfaction"])         * 1.20 +
    norm(df["distance_from_home_km"])        * 0.80 +
    norm(4 - df["environment_satisfaction"]) * 0.50 +
    rng.normal(0, 0.15, n)
)
df["attrition_risk_score"] = (raw_risk / raw_risk.max() * 5).clip(0, 5).round(3)


# ── FIX 6: skill_rating (normalised inputs → full [1, 5] range) ──────────────
df["skill_rating"] = (
    norm(df["years_at_company"])         * 0.35 +
    norm(df["training_times_last_year"]) * 0.35 +
    norm(df["education_level"])          * 0.30
) * 4 + 1
df["skill_rating"] = df["skill_rating"].clip(1, 5).round(2)


# ── total_annual_compensation (anchored to monthly_income) ───────────────────
bonus_amount = df["monthly_income"] * (df["bonus_pct"] / 100)
df["total_annual_compensation"] = (
    df["monthly_income"] * 12 +
    bonus_amount * 12 +
    df["years_at_company"] * 250 +
    rng.normal(0, 800, n)
).clip(15_000, 600_000).round(0).astype(int)


# ── remote_work_pct (driven by department) ───────────────────────────────────
remote_base = {"Research & Development": 45, "Sales": 8, "Human Resources": 30}
df["remote_work_pct"] = (
    df["department"].map(remote_base).fillna(25) +
    rng.normal(0, 8, n)
).clip(0, 100).round(1)


# ── FIX 5: work_life_score replaces work_life_balance_raw in output ───────────
df["work_life_score"] = (
    norm(df["work_life_balance_raw"])            * 0.50 +
    norm(df["remote_work_pct"])                  * 0.25 +
    norm(40 - df["overtime_hours_per_month"])    * 0.25
) * 4 + 1
df["work_life_score"] = df["work_life_score"].clip(1, 5).round(2)
df.drop(columns=["work_life_balance_raw"], inplace=True)  # input consumed


# ── burnout_index ──────────────────────────────────────────────────────────────
df["burnout_index"] = (
    norm(5 - df["work_life_score"])           * 0.45 +
    norm(df["overtime_hours_per_month"])       * 0.35 +
    norm(4 - df["environment_satisfaction"])   * 0.20
) * 5
df["burnout_index"] = df["burnout_index"].clip(0, 5).round(2)


# ── manager_rating ─────────────────────────────────────────────────────────────
df["manager_rating"] = (
    norm(df["relationship_satisfaction"])          * 0.60 +
    norm(np.log1p(df["years_with_curr_manager"]))  * 0.40
) * 4 + 1
df["manager_rating"] = df["manager_rating"].clip(1, 5).round(2)


# ── peer_rating ────────────────────────────────────────────────────────────────
df["peer_rating"] = (
    norm(df["job_involvement"]) * 0.50 +
    norm(df["skill_rating"])    * 0.50
) * 4 + 1
df["peer_rating"] = df["peer_rating"].clip(1, 5).round(2)


# ── overall_360_score ──────────────────────────────────────────────────────────
df["overall_360_score"] = (
    df["manager_rating"]    * 0.35 +
    df["peer_rating"]       * 0.30 +
    df["performance_score"] * 0.35 +
    rng.normal(0, 0.12, n)
).clip(1, 5).round(2)


# ── team_size ──────────────────────────────────────────────────────────────────
dept_team_base = {"Research & Development": 7, "Sales": 11, "Human Resources": 4}
df["team_size"] = (
    df["department"].map(dept_team_base).fillna(7) +
    _seniority_ord * 2 +
    rng.integers(0, 5, n)
).clip(2, 30).astype(int)


# ── FIX 9: career_advancement_score (renamed, direction corrected) ────────────
df["career_advancement_score"] = (
    norm(df["promotion_eligibility"].astype(float))  * 0.40 +
    norm(1 / (df["years_since_last_promotion"] + 1)) * 0.35 +
    norm(df["years_at_company"])                      * 0.25 +
    rng.normal(0, 0.05, n)
) * 5
df["career_advancement_score"] = df["career_advancement_score"].clip(0, 5).round(2)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved → {OUTPUT_FILE}   ({df.shape[0]} rows × {df.shape[1]} columns)\n")

IBM_BASE = {
    "age", "attrition_raw", "business_travel", "department",
    "distance_from_home_km", "education_level", "education_field",
    "environment_satisfaction", "gender", "job_involvement", "job_level",
    "job_role", "job_satisfaction", "marital_status", "monthly_income",
    "num_companies_worked", "overtime_flag", "pct_salary_hike",
    "relationship_satisfaction", "stock_option_level", "total_working_years",
    "training_times_last_year", "years_at_company", "years_in_current_role",
    "years_since_last_promotion", "years_with_curr_manager",
}

print("── Column summary ──────────────────────────────────────────────────────")
for col in df.columns:
    tag = "(IBM base) " if col in IBM_BASE else "(engineered)"
    print(f"  {col:<42} {tag}  dtype={df[col].dtype}  nuniq={df[col].nunique()}")

print("\n── Ground Truth Dependency Graph (evaluation rubric) ───────────────────")
deps = [
    ("monthly_income",                                           "salary_band"),
    ("job_level + total_working_years",                          "seniority_level"),
    ("performance_rating_raw + job_involvement + training",      "performance_score"),
    ("performance_score + seniority_level + department",         "bonus_pct"),
    ("performance_score + years_since_last_promotion",           "promotion_eligibility"),
    ("overtime_flag + job_level",                                "overtime_hours_per_month"),
    ("overtime_hours + job_satisfaction + distance + env_sat",   "attrition_risk_score"),
    ("attrition_raw",                                            "voluntary_exit_flag"),
    ("years_at_company + training_times + education_level",      "skill_rating"),
    ("monthly_income + years_at_company + bonus_pct",            "total_annual_compensation"),
    ("department",                                               "remote_work_pct"),
    ("work_life_balance_raw + remote_work_pct + overtime_hours", "work_life_score"),
    ("work_life_score + overtime_hours + env_satisfaction",      "burnout_index"),
    ("relationship_satisfaction + years_with_curr_manager",      "manager_rating"),
    ("job_involvement + skill_rating",                           "peer_rating"),
    ("manager_rating + peer_rating + performance_score",         "overall_360_score"),
    ("department + seniority_level",                             "team_size"),
    ("promotion_eligibility + years_since_promo + years_at_co",  "career_advancement_score"),
]
for cause, effect in deps:
    print(f"  {cause:<57} -->  {effect}")
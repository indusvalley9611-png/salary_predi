# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from scipy.sparse import hstack  # 🔥 IMPORTANT

# =========================================
# 2. LOAD DATA
# =========================================
job_skills = pd.read_csv("job_skills.csv")
skills = pd.read_csv("skills.csv")
job_industries = pd.read_csv("job_industries.csv")
industries = pd.read_csv("industries.csv")
salaries = pd.read_csv("salaries.csv")

print("✅ Files Loaded")

# =========================================
# 3. STANDARDIZE COLUMNS
# =========================================
job_id_col = next((c for c in job_skills.columns if "job" in c.lower()), None)
skill_col_js = next((c for c in job_skills.columns if "skill" in c.lower()), None)

job_skills.rename(columns={job_id_col: "job_id", skill_col_js: "skill_key"}, inplace=True)

# =========================================
# 4. HANDLE SKILLS (SAFE)
# =========================================
skill_name_col = next((c for c in skills.columns if "name" in c.lower() or "skill" in c.lower()), None)

if skill_name_col:
    skills.rename(columns={skill_name_col: "skill_name"}, inplace=True)
else:
    skills["skill_name"] = ""

skill_key_col = next((c for c in skills.columns if c != "skill_name"), None)

if skill_key_col:
    skills.rename(columns={skill_key_col: "skill_key"}, inplace=True)

if "skill_key" in skills.columns and "skill_key" in job_skills.columns:
    df = job_skills.merge(skills, on="skill_key", how="left")
else:
    df = job_skills.copy()
    df["skill_name"] = df["skill_key"].astype(str)

# =========================================
# 5. INDUSTRY
# =========================================
job_industries.rename(columns={
    next(c for c in job_industries.columns if "job" in c.lower()): "job_id",
    next(c for c in job_industries.columns if "industry" in c.lower()): "industry_key"
}, inplace=True)

industries.rename(columns={
    next(c for c in industries.columns if "industry" in c.lower() or "id" in c.lower()): "industry_key",
    next(c for c in industries.columns if "name" in c.lower()): "industry_name"
}, inplace=True)

df = df.merge(job_industries, on="job_id", how="left")
df = df.merge(industries, on="industry_key", how="left")

# =========================================
# 6. SALARY
# =========================================
salaries.rename(columns={
    next(c for c in salaries.columns if "job" in c.lower()): "job_id",
    next(c for c in salaries.columns if "salary" in c.lower()): "salary"
}, inplace=True)

df = df.merge(salaries, on="job_id", how="left")

print("✅ Merge Done")

# =========================================
# 7. CLEAN DATA
# =========================================
df = df.drop_duplicates()
df = df.dropna(subset=["salary"])

df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
df = df.dropna(subset=["salary"])

# Remove extreme outliers (fast + useful)
df = df[(df["salary"] > 2000) & (df["salary"] < 500000)]

# =========================================
# 8. GROUP
# =========================================
df_final = df.groupby("job_id").agg({
    "skill_name": lambda x: " ".join(x.astype(str)),
    "industry_name": "first",
    "salary": "first"
}).reset_index()

# OPTIONAL SPEED BOOST (uncomment if large data)
# df_final = df_final.sample(10000, random_state=42)

# =========================================
# 9. FEATURE ENGINEERING
# =========================================
df_final["skill_count"] = df_final["skill_name"].apply(lambda x: len(x.split()))
df_final["unique_skills"] = df_final["skill_name"].apply(lambda x: len(set(x.split())))
df_final["skill_ratio"] = df_final["unique_skills"] / (df_final["skill_count"] + 1)

important_skills = ["python", "sql", "aws", "machine learning", "java", "excel", "data"]

for skill in important_skills:
    df_final[f"has_{skill}"] = df_final["skill_name"].str.contains(skill, case=False).astype(int)

# log target
df_final["salary_log"] = np.log1p(df_final["salary"])

# =========================================
# 10. ENCODING
# =========================================
tfidf = TfidfVectorizer(
    max_features=800,        # ⚡ reduced
    ngram_range=(1,2),
    stop_words="english"
)

X_skills = tfidf.fit_transform(df_final["skill_name"])

le = LabelEncoder()
X_industry = le.fit_transform(df_final["industry_name"])

extra_features = df_final[
    ["skill_count", "unique_skills", "skill_ratio"] +
    [f"has_{s}" for s in important_skills]
].values

# 🔥 SPARSE MATRIX (NO .toarray())
X = hstack([
    X_skills,
    X_industry.reshape(-1,1),
    extra_features
])

y = df_final["salary_log"]

# =========================================
# 11. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 12. MODELS (FAST)
# =========================================
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        n_jobs=-1   # 🔥 FAST
    )
}

results = {}

# =========================================
# 13. TRAINING
# =========================================
for name, model in models.items():

    # Scale only for Ridge
    if name == "Ridge":
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = r2

    print(f"\n{name}")
    print("MAE:", mae)
    print("R2:", r2)

# =========================================
# 14. BEST MODEL
# =========================================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")

# =========================================
# 15. SAVE MODEL
# =========================================
joblib.dump(best_model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "encoder.pkl")

print("🚀 FAST MODEL READY")

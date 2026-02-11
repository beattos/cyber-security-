import joblib

FILES = [
    "models/ada_static.pkl",
    "models/gb_static.pkl",
    "models/ada_dynamic.pkl",
    "models/gb_dynamic.pkl",
    "models/ada_dynamic_calibrated.pkl",
    "models/gb_dynamic_calibrated.pkl",
    "models/static_feature_cols.pkl",
    "models/dynamic_feature_cols.pkl",
    "models/static_imputer.pkl",
    "models/dynamic_imputer.pkl",
]

for f in FILES:
    print(f"\n=== {f} ===")
    obj = joblib.load(f)
    print("Type:", type(obj))

    # Extra details for known objects
    if hasattr(obj, "get_params"):
        print("Params:", obj.get_params())

    if hasattr(obj, "feature_importances_"):
        print("Top importances:", obj.feature_importances_[:5])

    if hasattr(obj, "statistics_"):
        print("Imputer stats (first 5):", obj.statistics_[:5])

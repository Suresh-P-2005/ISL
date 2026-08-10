from sklearn.ensemble import RandomForestClassifier

def build_random_forest(n_estimators=100, random_state=42):
    """
    Builds Random Forest classifier wrapper for static sign inference.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight='balanced',
        random_state=random_state
    )

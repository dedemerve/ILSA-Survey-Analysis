from src.analysis.ml_taxonomy import canonicalize
from src.analysis.performance_parser import parse_performance
from src.analysis.study_attributes import detect_cycles, detect_programs
from src.knowledge.predictor_kb import predictor_direction


def test_taxonomy_word_boundaries():
    # Former substring matcher mapped these to Decision Tree / SEM / PCA.
    assert canonicalize("Gradient Boosted Decision Trees").name == "Gradient Boosting"
    assert canonicalize("Semi-supervised learning").role == "learning_paradigm"
    assert canonicalize("Principal components regression").name == "Principal Components Regression"
    assert canonicalize("SVR").name == canonicalize("Support Vector Machine").name == "SVM"
    assert canonicalize("SHAP").role == "feature_or_xai"
    assert canonicalize("Distributed Random Forest").name == "Random Forest"


def test_parser_attributes_values_to_models():
    rows = parse_performance("RF rMSE=69.82, XGBoost rMSE=66.39; LightGBM PR-AUC=0.3, AUC=81%")
    got = {(r["model"], r["metric"]): r["value"] for r in rows}
    assert got[("Random Forest", "RMSE")] == 69.82
    assert got[("XGBoost", "RMSE")] == 66.39
    assert got[("LightGBM", "AUC")] == 0.81  # percent → proportion; PR-AUC ignored
    assert len(rows) == 3


def test_parser_sole_model_and_unattributed():
    assert parse_performance("Accuracy = 0.8", study_models=["XGBoost"])[0]["model"] == "XGBoost"
    assert parse_performance("Accuracy = 0.8", study_models=["XGBoost", "SVM"])[0]["model"] is None


def test_programs_and_cycles():
    text = "TIMSS 2003-2019 Grade 8 and PISA 2018 Reading"
    assert detect_programs(text) == ["PISA", "TIMSS"]
    assert {"TIMSS 2003", "TIMSS 2019", "PISA 2018"} <= set(detect_cycles(text))


def test_direction_is_clause_local():
    text = ["ESCS was positively associated with reading literacy, while math anxiety (ANXMAT) showed negative contributions."]
    assert predictor_direction("Socioeconomic status (ESCS)", "ESCS", text) == "positive"
    assert predictor_direction("Math anxiety", "ANXMAT", text) == "negative"
    assert predictor_direction("Governance index", None, ["Governance did not show a positive effect on performance."]) == "unspecified"

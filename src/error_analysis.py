"""
src/error_analysis.py
Utilities for error analysis of the best-performing model.
"""

import pandas as pd
import numpy as np


def get_error_samples(df_test, y_true, y_pred, n_samples=30):
    """
    Identify and categorize error samples.
    """
    df = df_test.copy()
    df["true_label"] = y_true
    df["pred_label"] = y_pred
    
    errors = df[df["true_label"] != df["pred_label"]].copy()
    
    # Simple heuristic categorization
    def categorize_error(row):
        text = str(row["text"]).lower()
        if len(text.split()) < 5:
            return "Văn bản quá ngắn"
        if "không" in text or "chẳng" in text or "chưa" in text:
            return "Phủ định / Mơ hồ"
        if row["true_label"] == 1:
            return "Nhãn trung tính (khó)"
        return "Khác / Domain shift"

    errors["error_category"] = errors.apply(categorize_error, axis=1)
    
    # Ensure we get a good distribution if possible
    return errors.head(n_samples)


def summarize_errors(errors_df):
    """Group errors by category for reporting."""
    summary = errors_df["error_category"].value_counts().to_dict()
    return summary


def format_error_report(errors_df, class_names):
    """Format the top 10 errors for the report as requested."""
    report = []
    for _, row in errors_df.head(10).iterrows():
        report.append({
            "Text": row["text"],
            "True": class_names[row["true_label"]],
            "Pred": class_names[row["pred_label"]],
            "Category": row["error_category"]
        })
    return report

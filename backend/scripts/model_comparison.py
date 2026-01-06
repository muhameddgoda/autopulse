#!/usr/bin/env python3
"""
AutoPulse Model Comparison
==========================
Compares different ML algorithms and hyperparameters for driver behavior classification.

Algorithms tested:
- Random Forest (default)
- XGBoost (optional, requires xgboost package)
- Gradient Boosting
- Support Vector Machine

Outputs results to a text file for review.

Usage:
    python model_comparison.py --vehicle-id <uuid> --output results.txt
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any

import httpx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ scikit-learn not installed. Run: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")


API_BASE = "http://localhost:8000"


async def fetch_training_data(vehicle_id: str, days: int = 30) -> List[Dict]:
    """Fetch training examples from the API"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Trigger data collection
        response = await client.post(
            f"{API_BASE}/api/telemetry/ml/train/{vehicle_id}?days={days}&train_behavior=false&train_anomaly=false"
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch training data: {response.text}")
        
        data = response.json()
        return data


def load_training_csv(vehicle_id: str) -> tuple:
    """Load training data from CSV file"""
    from app.ml.training.data_collector import DataCollector
    
    collector = DataCollector()
    
    # Find the most recent training data file
    data_dir = Path("app/ml/training_data")
    csv_files = list(data_dir.glob(f"training_data_{vehicle_id}*.csv"))
    
    if not csv_files:
        csv_files = list(data_dir.glob("training_data_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No training data CSV found. Run generate_training_data.py first.")
    
    # Use most recent
    csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Loading training data from: {csv_file}")
    
    examples = collector.load_training_data(str(csv_file))
    X, feature_names = collector.prepare_features_matrix(examples)
    y = collector.get_behavior_labels(examples)
    
    return X, y, feature_names, examples


def compare_models(X: np.ndarray, y: np.ndarray, feature_names: List[str], output_file: str):
    """Compare different ML models and hyperparameters"""
    
    results = []
    results.append("=" * 80)
    results.append("AUTOPULSE ML MODEL COMPARISON REPORT")
    results.append(f"Generated: {datetime.now().isoformat()}")
    results.append("=" * 80)
    results.append("")
    
    # Data overview
    results.append("DATA OVERVIEW")
    results.append("-" * 40)
    results.append(f"Total samples: {len(y)}")
    results.append(f"Features: {len(feature_names)}")
    results.append(f"Feature names: {', '.join(feature_names)}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    results.append(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        results.append(f"  • {cls}: {count} ({count/len(y)*100:.1f}%)")
    results.append("")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results.append(f"Training samples: {len(X_train)}")
    results.append(f"Test samples: {len(X_test)}")
    results.append("")
    
    # Define models to test
    models = {
        "Random Forest (default)": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "Random Forest (deep)": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=2, random_state=42
        ),
        "Random Forest (shallow)": RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "Gradient Boosting (tuned)": GradientBoostingClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42
        ),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=42),
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost (default)"] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss', random_state=42
        )
        models["XGBoost (tuned)"] = XGBClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss', random_state=42
        )
        models["XGBoost (regularized)"] = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08,
            reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric='mlogloss', random_state=42
        )
    
    # Compare models
    results.append("MODEL COMPARISON")
    results.append("=" * 80)
    
    model_scores = {}
    
    for name, model in models.items():
        results.append("")
        results.append(f"📊 {name}")
        results.append("-" * 60)
        
        try:
            # Use scaled data for SVM, unscaled for tree-based
            if "SVM" in name:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test
            
            # Train
            model.fit(X_tr, y_train)
            
            # Predict
            y_pred = model.predict(X_te)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5)
            
            results.append(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            results.append(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            results.append(f"CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
            results.append(f"\nPer-class metrics:")
            for cls in classes:
                if cls in report:
                    r = report[cls]
                    results.append(f"  {cls}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1-score']:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results.append(f"\nConfusion Matrix:")
            results.append(f"  Classes: {list(classes)}")
            for i, row in enumerate(cm):
                results.append(f"  {classes[i]}: {list(row)}")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1][:10]
                results.append(f"\nTop 10 Features:")
                for idx in sorted_idx:
                    results.append(f"  • {feature_names[idx]}: {importances[idx]:.4f}")
            
            model_scores[name] = {
                "accuracy": accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
            
        except Exception as e:
            results.append(f"❌ Error: {str(e)}")
            model_scores[name] = {"accuracy": 0, "cv_mean": 0, "cv_std": 0}
    
    # Summary
    results.append("")
    results.append("=" * 80)
    results.append("SUMMARY - RANKED BY ACCURACY")
    results.append("=" * 80)
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    results.append("")
    results.append(f"{'Rank':<5} {'Model':<35} {'Accuracy':<12} {'CV Mean':<12} {'CV Std':<10}")
    results.append("-" * 80)
    
    for i, (name, scores) in enumerate(sorted_models, 1):
        results.append(
            f"{i:<5} {name:<35} {scores['accuracy']:.4f}       "
            f"{scores['cv_mean']:.4f}       {scores['cv_std']:.4f}"
        )
    
    # Best model recommendation
    best_model = sorted_models[0]
    results.append("")
    results.append("=" * 80)
    results.append("RECOMMENDATION")
    results.append("=" * 80)
    results.append(f"Best performing model: {best_model[0]}")
    results.append(f"Test Accuracy: {best_model[1]['accuracy']*100:.1f}%")
    results.append(f"Cross-validation: {best_model[1]['cv_mean']*100:.1f}% ± {best_model[1]['cv_std']*100:.1f}%")
    results.append("")
    
    # Hyperparameter tuning for best model type
    results.append("=" * 80)
    results.append("HYPERPARAMETER TUNING (Random Forest)")
    results.append("=" * 80)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    results.append(f"\nSearching parameter grid:")
    for param, values in param_grid.items():
        results.append(f"  • {param}: {values}")
    
    try:
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        results.append(f"\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            results.append(f"  • {param}: {value}")
        
        results.append(f"\nBest CV score: {grid_search.best_score_:.4f}")
        
        # Test with best params
        best_model_tuned = grid_search.best_estimator_
        y_pred_tuned = best_model_tuned.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
        results.append(f"Test accuracy with best params: {tuned_accuracy:.4f}")
        
    except Exception as e:
        results.append(f"❌ Grid search error: {e}")
    
    results.append("")
    results.append("=" * 80)
    results.append("END OF REPORT")
    results.append("=" * 80)
    
    # Write to file
    output_text = "\n".join(results)
    
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    print(output_text)
    print(f"\n✅ Results saved to: {output_file}")
    
    return model_scores


async def main():
    parser = argparse.ArgumentParser(description="Compare ML models for AutoPulse")
    parser.add_argument("--vehicle-id", type=str, help="Vehicle UUID")
    parser.add_argument("--output", type=str, default="model_comparison_results.txt", help="Output file")
    parser.add_argument("--days", type=int, default=30, help="Days of data to use")
    
    args = parser.parse_args()
    
    if not SKLEARN_AVAILABLE:
        print("Cannot run comparison without scikit-learn")
        return
    
    # Change to backend directory
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)
    
    print("Loading training data...")
    
    try:
        X, y, feature_names, examples = load_training_csv(args.vehicle_id or "")
        print(f"Loaded {len(y)} training examples")
        
        compare_models(X, y, feature_names, args.output)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nRun the training data generator first:")
        print("  python scripts/generate_training_data.py --trips 200")


if __name__ == "__main__":
    asyncio.run(main())

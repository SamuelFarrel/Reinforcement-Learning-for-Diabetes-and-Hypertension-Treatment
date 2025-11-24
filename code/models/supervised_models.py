import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from utils.utils import (preprocess_diabetes_features, preprocess_hypertension_features, 
                 feature_selection, create_sampling_strategies)

def train_supervised_models_with_hyperparameter_tuning(X_diabetes_train, y_diabetes_train, 
                                                      X_hyper_train, y_hyper_train, seed):
    """Enhanced supervised learning with preprocessing, feature engineering and sampling"""
    print("=" * 60)
    print("TRAINING SUPERVISED LEARNING MODEL")
    print("=" * 60)

    X_diab_df = pd.DataFrame(X_diabetes_train, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    X_hyper_df = pd.DataFrame(X_hyper_train, columns=['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 
                                             'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])
    
    print("\nApplying advanced preprocessing to diabetes dataset...")
    X_diabetes_enhanced = preprocess_diabetes_features(X_diab_df)
    
    print("\nApplying advanced preprocessing to hypertension dataset...")
    X_hyper_enhanced = preprocess_hypertension_features(X_hyper_df)
    
    print("\nPerforming feature selection for diabetes...")
    X_diabetes_selected, diabetes_selected_features = feature_selection(
        X_diabetes_enhanced, y_diabetes_train, n_features=15, seed=seed
    )
    
    print("\nPerforming feature selection for hypertension...")
    X_hyper_selected, hyper_selected_features = feature_selection(
        X_hyper_enhanced, y_hyper_train, n_features=15, seed=seed
    )

    print("\nCreating sampling strategies for diabetes...")
    diabetes_strategies = create_sampling_strategies(
        X_diabetes_selected, y_diabetes_train, seed=seed
    )
    
    print("\nCreating sampling strategies for hypertension...")
    hyper_strategies = create_sampling_strategies(
        X_hyper_selected, y_hyper_train, seed=seed
    )
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5]
    }

    best_diabetes_model = None
    best_diabetes_auc = 0
    best_diabetes_metrics = None
    
    print("\nTraining diabetes models with different sampling strategies...")
    for strategy_name, (X_strategy, y_strategy) in diabetes_strategies.items():
        print(f"  Strategy: {strategy_name}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_strategy, y_strategy, test_size=0.2, random_state=seed
        )
        
        rf = RandomForestClassifier(random_state=seed)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        
        y_val_pred = best_rf.predict(X_val)
        y_val_prob = best_rf.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_val_pred)
        try:
            auc_score = roc_auc_score(y_val, y_val_prob)
        except:
            auc_score = 0
            
        print(f"    Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        if auc_score > best_diabetes_auc:
            best_diabetes_auc = auc_score
            best_diabetes_model = best_rf
            best_diabetes_metrics = {
                'accuracy': accuracy,
                'auc': auc_score,
                'confusion_matrix': confusion_matrix(y_val, y_val_pred),
                'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
            }
            print(f"    New best model! (Strategy: {strategy_name})")
    
    best_hyper_model = None
    best_hyper_auc = 0
    best_hyper_metrics = None
    
    print("\nTraining hypertension models with different sampling strategies...")
    for strategy_name, (X_strategy, y_strategy) in hyper_strategies.items():
        print(f"  Strategy: {strategy_name}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_strategy, y_strategy, test_size=0.2, random_state=seed
        )
        
        rf = RandomForestClassifier(random_state=seed)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        
        y_val_pred = best_rf.predict(X_val)
        y_val_prob = best_rf.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_val_pred)
        try:
            auc_score = roc_auc_score(y_val, y_val_prob)
        except:
            auc_score = 0
            
        print(f"    Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        if auc_score > best_hyper_auc:
            best_hyper_auc = auc_score
            best_hyper_model = best_rf
            best_hyper_metrics = {
                'accuracy': accuracy,
                'auc': auc_score,
                'confusion_matrix': confusion_matrix(y_val, y_val_pred),
                'classification_report': classification_report(y_val, y_val_pred, output_dict=True)
            }
            print(f"    New best model! (Strategy: {strategy_name})")

    print("\nRetraining best models on full datasets...")

    print("Training final diabetes model...")
    params = best_diabetes_model.get_params()
    params['random_state'] = seed
    rf_diabetes = RandomForestClassifier(**params)
    rf_diabetes.fit(X_diabetes_enhanced, y_diabetes_train)

    print("Training final hypertension model...")
    params = best_hyper_model.get_params()
    params['random_state'] = seed
    rf_hyper = RandomForestClassifier(**params)
    rf_hyper.fit(X_hyper_enhanced, y_hyper_train)
    
    print("\nSupervised learning training completed!")
    print(f"Diabetes model - Best AUC: {best_diabetes_auc:.4f}")
    print(f"Hypertension model - Best AUC: {best_hyper_auc:.4f}")
    
    return rf_diabetes, rf_hyper, {
        'diabetes': best_diabetes_metrics,
        'hypertension': best_hyper_metrics
    }
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import yaml

def load_config():
    """Load configuration from YAML file"""
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)

def create_directories():
    """Create directories for results"""
    config = load_config()
    
    RESULTS_DIR = config['results_dir']
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    subdirs = config['subdirs']
    paths = {}
    
    for key, subdir in subdirs.items():
        dir_path = os.path.join(RESULTS_DIR, subdir)
        os.makedirs(dir_path, exist_ok=True)
        paths[key] = dir_path
        
    return paths

def prepare_datasets():
    """Prepare train and test datasets"""
    config = load_config()
    
    try:
        diabetes_train_df = pd.read_csv(config['datasets']['diabetes']['train'])
        hyper_train_df = pd.read_csv(config['datasets']['hypertension']['train'])
        print("Loaded train datasets")
        
        diabetes_test_df = pd.read_csv(config['datasets']['diabetes']['test'])
        hyper_test_df = pd.read_csv(config['datasets']['hypertension']['test'])
        print("Loaded test datasets")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    diabetes_train_df = diabetes_train_df.fillna(diabetes_train_df.median())
    hyper_train_df = hyper_train_df.fillna(hyper_train_df.median())
    diabetes_test_df = diabetes_test_df.fillna(diabetes_test_df.median())
    hyper_test_df = hyper_test_df.fillna(hyper_test_df.median())
    
    diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    hyper_features = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 
                     'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    
    X_diabetes_train = diabetes_train_df[diabetes_features]
    y_diabetes_train = diabetes_train_df['Outcome']
    X_hyper_train = hyper_train_df[hyper_features]
    y_hyper_train = hyper_train_df['Risk']
    
    X_diabetes_test = diabetes_test_df[diabetes_features]
    X_hyper_test = hyper_test_df[hyper_features]

    diab_scaler = StandardScaler()
    X_diabetes_train_scaled = diab_scaler.fit_transform(X_diabetes_train)
    X_diabetes_test_scaled = diab_scaler.transform(X_diabetes_test)
    
    hyper_scaler = StandardScaler()
    X_hyper_train_scaled = hyper_scaler.fit_transform(X_hyper_train)
    X_hyper_test_scaled = hyper_scaler.transform(X_hyper_test)
    
    print(f"Train dataset shapes: Diabetes {X_diabetes_train_scaled.shape}, Hypertension {X_hyper_train_scaled.shape}")
    print(f"Test dataset shapes: Diabetes {X_diabetes_test_scaled.shape}, Hypertension {X_hyper_test_scaled.shape}")
    
    return (X_diabetes_train_scaled, y_diabetes_train, X_hyper_train_scaled, y_hyper_train,
            X_diabetes_test_scaled, X_hyper_test_scaled, diab_scaler, hyper_scaler)

def ensure_n_rows(df, n_rows):
    """Ensure dataframe has exactly n_rows"""
    if len(df) > n_rows:
        return df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    elif len(df) < n_rows:
        additional_needed = n_rows - len(df)
        additional_samples = df.sample(n=additional_needed, replace=True, random_state=42)
        return pd.concat([df, additional_samples], ignore_index=True)
    return df

def preprocess_diabetes_features(X_train):
    """Apply advanced preprocessing to diabetes features"""

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    X_train['Glucose_BMI'] = X_train['Glucose'] * X_train['BMI']
    X_train['Age_BMI'] = X_train['Age'] * X_train['BMI']
    X_train['Age_Glucose'] = X_train['Age'] * X_train['Glucose']
    
    X_train['Glucose_to_Insulin_Ratio'] = X_train['Glucose'] / (X_train['Insulin'] + 1)
    
    X_train['High_Glucose'] = (X_train['Glucose'] > 1).astype(int)
    X_train['High_BMI'] = (X_train['BMI'] > 1).astype(int)
    X_train['High_Age'] = (X_train['Age'] > 1).astype(int)
    X_train['Risk_Score'] = X_train['High_Glucose'] + X_train['High_BMI'] + X_train['High_Age']
    
    return X_train

def preprocess_hypertension_features(X_train):
    """Apply advanced preprocessing to hypertension features"""
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 
                                             'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])
    
    X_train['Age_BMI'] = X_train['age'] * X_train['BMI'] / 10
    X_train['Glucose_SysBP'] = X_train['glucose'] * X_train['sysBP'] / 10
    
    X_train['Pulse_Pressure'] = X_train['sysBP'] - X_train['diaBP']
    X_train['MAP'] = (X_train['sysBP'] + 2 * X_train['diaBP']) / 3
    
    X_train['High_BP'] = (X_train['sysBP'] > 1).astype(int)
    X_train['High_Chol'] = (X_train['totChol'] > 1).astype(int)
    X_train['High_Age'] = (X_train['age'] > 1).astype(int)
    X_train['Risk_Score'] = X_train['High_BP'] + X_train['High_Chol'] + X_train['High_Age'] + X_train['diabetes']
    
    X_train['Smoking_Intensity'] = X_train['currentSmoker'] * X_train['cigsPerDay']
    
    return X_train

def feature_selection(X, y, n_features=15, seed=42):
    """Apply feature selection to find the most important features"""
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X, y)
    
    selector = SelectFromModel(rf, max_features=n_features, threshold=-np.inf, prefit=True)
    X_selected = selector.transform(X)
    
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    return X_selected, selected_features

def simple_random_oversampling(X, y, seed=42):
    """Simple random oversampling implementation"""
    np.random.seed(seed)
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(class_counts)]
    majority_count = np.max(class_counts)
    
    X_resampled = X.copy()
    y_resampled = y.copy()
    
    for class_label in unique_classes:
        if class_label == majority_class:
            continue
            
        class_indices = np.where(y == class_label)[0]
        
        n_samples_needed = majority_count - len(class_indices)
        
        if n_samples_needed <= 0:
            continue
            
        resampled_indices = np.random.choice(class_indices, size=n_samples_needed, replace=True)
        
        X_resampled = np.vstack([X_resampled, X[resampled_indices]])
        y_resampled = np.append(y_resampled, y[resampled_indices])
    
    return X_resampled, y_resampled

def simple_random_undersampling(X, y, seed=42):
    """Simple random undersampling implementation"""
    np.random.seed(seed)
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    minority_count = np.min(class_counts)
    
    X_resampled = []
    y_resampled = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        
        if class_label != minority_class and len(class_indices) > minority_count:
            selected_indices = np.random.choice(class_indices, size=minority_count, replace=False)
        else:
            selected_indices = class_indices
            
        if len(X_resampled) == 0:
            X_resampled = X[selected_indices]
            y_resampled = y[selected_indices]
        else:
            X_resampled = np.vstack([X_resampled, X[selected_indices]])
            y_resampled = np.append(y_resampled, y[selected_indices])
    
    return X_resampled, y_resampled

def create_sampling_strategies(X, y, seed=42):
    """Create different sampling strategies to handle class imbalance using custom implementations"""

    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    
    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    strategies = {}
    
    strategies['original'] = (X, y)
    
    if imbalance_ratio >= 1.5:
        X_over, y_over = simple_random_oversampling(X, y, seed)
        strategies['oversampling'] = (X_over, y_over)
        
        X_under, y_under = simple_random_undersampling(X, y, seed)
        strategies['undersampling'] = (X_under, y_under)

        X_combined, y_combined = simple_random_oversampling(X, y, seed)
        if isinstance(X_combined, np.ndarray):
            noise = np.random.normal(0, 0.1, X_combined.shape)
            X_combined = X_combined + noise
        
        strategies['combined'] = (X_combined, y_combined)
        
        for name, (X_resampled, y_resampled) in strategies.items():
            new_class_counts = np.bincount(y_resampled)
            print(f"After {name}: {dict(zip(range(len(new_class_counts)), new_class_counts))}")
    
    return strategies
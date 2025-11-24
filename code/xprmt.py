import os
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from environments.diabetes_env import DiabetesEnv
from environments.hypertension_env import HypertensionEnv
from models.supervised_models import *
from agents.agents import *
from utils.utils import *
from utils.visualization import *

def create_environments(X_diabetes_test, X_hyper_test, rf_diabetes, rf_hyper):
    """Create all environment variants with properly processed test data"""
    
    X_diabetes_test_df = pd.DataFrame(X_diabetes_test, 
                                    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    X_hyper_test_df = pd.DataFrame(X_hyper_test,
                                  columns=['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 
                                         'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])
    
    X_diabetes_test_enhanced = preprocess_diabetes_features(X_diabetes_test_df)
    X_hyper_test_enhanced = preprocess_hypertension_features(X_hyper_test_df)
    
    diab_env_with_sl = DiabetesEnv(X_diabetes_test_enhanced.values, risk_model=rf_diabetes, use_sl_guidance=True)
    hyper_env_with_sl = HypertensionEnv(X_hyper_test_enhanced.values, risk_model=rf_hyper, use_sl_guidance=True)
    diab_env_without_sl = DiabetesEnv(X_diabetes_test_enhanced.values, risk_model=rf_diabetes, use_sl_guidance=False)
    hyper_env_without_sl = HypertensionEnv(X_hyper_test_enhanced.values, risk_model=rf_hyper, use_sl_guidance=False)
    
    return diab_env_with_sl, hyper_env_with_sl, diab_env_without_sl, hyper_env_without_sl

def main():
    """Main experiment pipeline"""
    config = load_config()
    SEEDS = config['seeds']
    dirs = create_directories()
    
    print("RL for Diabetes and Hypertension Management")
    print("=" * 60)
    
    (X_diabetes_train, y_diabetes_train, X_hyper_train, y_hyper_train,
     X_diabetes_test, X_hyper_test, diab_scaler, hyper_scaler) = prepare_datasets()
    
    all_results_with_sl = {'diabetes': [], 'hypertension': []}
    all_results_without_sl = {'diabetes': [], 'hypertension': []}
    all_reward_by_timestep_with_sl = {'diabetes': {}, 'hypertension': {}}
    all_reward_by_timestep_without_sl = {'diabetes': {}, 'hypertension': {}}
    all_health_metrics_with_sl = {'diabetes': {}, 'hypertension': {}}
    all_health_metrics_without_sl = {'diabetes': {}, 'hypertension': {}}
    
    sl_metrics_across_seeds = []
    
    total_start_time = time.time()
    
    for seed_idx, seed in enumerate(SEEDS): 
        print(f"\nExperiment {seed_idx + 1}/{len(SEEDS)} (seed: {seed})")
        print("-" * 40)
        
        print("Training supervised models with advanced techniques...")
        rf_diabetes, rf_hyper, sl_metrics = train_supervised_models_with_hyperparameter_tuning(
            X_diabetes_train, y_diabetes_train, X_hyper_train, y_hyper_train, seed)
        
        sl_metrics_across_seeds.append(sl_metrics)
        
        envs = create_environments(X_diabetes_test, X_hyper_test, rf_diabetes, rf_hyper)
        diab_env_with_sl, hyper_env_with_sl, diab_env_without_sl, hyper_env_without_sl = envs
    
        timesteps = config['timesteps']
        
        print("Training RL models...")
        
        print("  Diabetes + SL:")
        diab_models_with_sl, diab_callbacks_with_sl = train_rl_models(diab_env_with_sl, seed, timesteps)
        
        print("  Hypertension + SL:")
        hyper_models_with_sl, hyper_callbacks_with_sl = train_rl_models(hyper_env_with_sl, seed, timesteps)
        
        print("  Diabetes (no SL):")
        diab_models_without_sl, diab_callbacks_without_sl = train_rl_models(diab_env_without_sl, seed, timesteps)
        
        print("  Hypertension (no SL):")
        hyper_models_without_sl, hyper_callbacks_without_sl = train_rl_models(hyper_env_without_sl, seed, timesteps)
        
        print("Detailed evaluation...")
        
        diab_results_with_sl, diab_ep_rewards_with_sl, diab_rewards_by_timestep_with_sl, diab_health_metrics_with_sl = \
            evaluate_models_with_detailed_tracking(diab_env_with_sl, diab_models_with_sl, n_episodes=config['eval_episodes'], seed=seed)
        
        diab_results_without_sl, diab_ep_rewards_without_sl, diab_rewards_by_timestep_without_sl, diab_health_metrics_without_sl = \
            evaluate_models_with_detailed_tracking(diab_env_without_sl, diab_models_without_sl, n_episodes=config['eval_episodes'], seed=seed)
        
        hyper_results_with_sl, hyper_ep_rewards_with_sl, hyper_rewards_by_timestep_with_sl, hyper_health_metrics_with_sl = \
            evaluate_models_with_detailed_tracking(hyper_env_with_sl, hyper_models_with_sl, n_episodes=config['eval_episodes'], seed=seed)
        
        hyper_results_without_sl, hyper_ep_rewards_without_sl, hyper_rewards_by_timestep_without_sl, hyper_health_metrics_without_sl = \
            evaluate_models_with_detailed_tracking(hyper_env_without_sl, hyper_models_without_sl, n_episodes=config['eval_episodes'], seed=seed)
        
        all_results_with_sl['diabetes'].append(diab_ep_rewards_with_sl)
        all_results_without_sl['diabetes'].append(diab_ep_rewards_without_sl)
        all_results_with_sl['hypertension'].append(hyper_ep_rewards_with_sl)
        all_results_without_sl['hypertension'].append(hyper_ep_rewards_without_sl)
        
        if seed_idx == 0:
            all_reward_by_timestep_with_sl['diabetes'] = diab_rewards_by_timestep_with_sl
            all_reward_by_timestep_without_sl['diabetes'] = diab_rewards_by_timestep_without_sl
            all_reward_by_timestep_with_sl['hypertension'] = hyper_rewards_by_timestep_with_sl
            all_reward_by_timestep_without_sl['hypertension'] = hyper_rewards_by_timestep_without_sl
            
            all_health_metrics_with_sl['diabetes'] = diab_health_metrics_with_sl
            all_health_metrics_without_sl['diabetes'] = diab_health_metrics_without_sl
            all_health_metrics_with_sl['hypertension'] = hyper_health_metrics_with_sl
            all_health_metrics_without_sl['hypertension'] = hyper_health_metrics_without_sl
        
        print(f"Seed {seed} completed")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    print("\nGenerating visualizations...")
    
    plot_sl_performance(sl_metrics_across_seeds, 
                      save_path=os.path.join(dirs['sl_performance'], "sl_performance_across_seeds.png"))
    
    plot_confusion_matrices(sl_metrics_across_seeds,
                          save_path=os.path.join(dirs['sl_performance'], "sl_confusion_matrices.png"))
    
    plot_rewards_over_time(all_reward_by_timestep_with_sl['diabetes'], 
                          'Diabetes with SL Guidance',
                          save_path=os.path.join(dirs['rewards'], "diabetes_rewards_with_sl.png"))
    
    plot_rewards_over_time(all_reward_by_timestep_without_sl['diabetes'], 
                          'Diabetes without SL Guidance',
                          save_path=os.path.join(dirs['rewards'], "diabetes_rewards_without_sl.png"))
    
    plot_rewards_over_time(all_reward_by_timestep_with_sl['hypertension'], 
                          'Hypertension with SL Guidance',
                          save_path=os.path.join(dirs['rewards'], "hypertension_rewards_with_sl.png"))
    
    plot_rewards_over_time(all_reward_by_timestep_without_sl['hypertension'], 
                          'Hypertension without SL Guidance',
                          save_path=os.path.join(dirs['rewards'], "hypertension_rewards_without_sl.png"))
    
    plot_patient_health_over_time(all_health_metrics_with_sl['diabetes'], 
                                 'Diabetes', 'glucose',
                                 save_path=os.path.join(dirs['health_metrics'], "diabetes_glucose_trajectory.png"))
    
    plot_patient_health_over_time(all_health_metrics_with_sl['hypertension'], 
                                 'Hypertension', 'bp',
                                 save_path=os.path.join(dirs['health_metrics'], "hypertension_bp_trajectory.png"))
    
    plot_model_comparison_with_target_ranges(
        all_health_metrics_with_sl['diabetes'], 
        all_health_metrics_without_sl['diabetes'], 
        'Diabetes', 'glucose',
        save_path=os.path.join(dirs['comparisons'], "diabetes_sl_vs_nosl_comparison.png")
    )
    
    plot_model_comparison_with_target_ranges(
        all_health_metrics_with_sl['hypertension'], 
        all_health_metrics_without_sl['hypertension'], 
        'Hypertension', 'bp',
        save_path=os.path.join(dirs['comparisons'], "hypertension_sl_vs_nosl_comparison.png")
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED!")
    print("=" * 60)
    print("Key findings:")
    
    for condition in ['diabetes', 'hypertension']:
        print(f"\n{condition.upper()}:")
        for model in ['PPO', 'DQN', 'A2C']:
            with_sl_avg = np.mean([seed_result[model] for seed_result in all_results_with_sl[condition]])
            without_sl_avg = np.mean([seed_result[model] for seed_result in all_results_without_sl[condition]])
            improvement = with_sl_avg - without_sl_avg
            print(f"  {model}: {improvement:+.2f} improvement with SL guidance")
    
    print("\nAll visualizations saved to:")
    print(f"  - SL model performance: {dirs['sl_performance']}")
    print(f"  - Reward plots: {dirs['rewards']}")
    print(f"  - Health trajectories: {dirs['health_metrics']}")
    print(f"  - Model comparisons: {dirs['comparisons']}")

if __name__ == "__main__":
    main()

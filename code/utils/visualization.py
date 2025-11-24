import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_rewards_over_time(reward_by_timestep, title, save_path=None):
    """Plot average reward accumulation over timesteps with confidence intervals"""
    plt.figure(figsize=(14, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (model_name, rewards_by_episode) in enumerate(reward_by_timestep.items()):
        max_length = max(len(episode_rewards) for episode_rewards in rewards_by_episode)
        
        padded_rewards = []
        for episode_rewards in rewards_by_episode:
            if len(episode_rewards) < max_length:
                padding = [episode_rewards[-1]] * (max_length - len(episode_rewards))
                padded_rewards.append(episode_rewards + padding)
            else:
                padded_rewards.append(episode_rewards)
        
        rewards_array = np.array(padded_rewards)
        
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        n_episodes = len(rewards_by_episode)
        ci_95 = 1.96 * std_rewards / np.sqrt(n_episodes)
        
        timesteps = range(1, max_length + 1)
        plt.plot(timesteps, mean_rewards, label=model_name, color=colors[i], linewidth=2)
        plt.fill_between(timesteps, mean_rewards - ci_95, mean_rewards + ci_95, alpha=0.2, color=colors[i])
    
    plt.title(f'Average Reward Over Timesteps - {title}', fontsize=16)
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_patient_health_over_time(health_metrics, condition_name, metric_type='glucose', save_path=None):
    """Plot patient health metrics over time with target ranges clearly highlighted"""
    plt.figure(figsize=(14, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    if metric_type == 'glucose':
        target_range = (90, 130)
        trajectories_key = 'glucose_trajectories'
        ylabel = 'Glucose Level (mg/dL)'
        title = 'Blood Glucose Level Over Time'
        danger_low = 70
        danger_high = 180
    else:
        target_range = (110, 130)
        trajectories_key = 'bp_trajectories'
        ylabel = 'Systolic Blood Pressure (mmHg)'
        title = 'Blood Pressure Over Time'
        danger_low = 90
        danger_high = 140
    
    plt.axhspan(danger_high, 300, alpha=0.2, color='red', label='Danger High Zone')
    plt.axhspan(target_range[1], danger_high, alpha=0.2, color='orange', label='Warning Zone')
    plt.axhspan(target_range[0], target_range[1], alpha=0.3, color='green', label='Target Range')
    plt.axhspan(danger_low, target_range[0], alpha=0.2, color='orange', label='Warning Zone')
    plt.axhspan(0, danger_low, alpha=0.2, color='red', label='Danger Low Zone')
    
    for i, (model_name, metrics) in enumerate(health_metrics.items()):
        trajectories = metrics[trajectories_key]
        
        if not trajectories:
            continue
            
        max_len = max(len(traj) for traj in trajectories)
        padded_trajectories = []
        
        for traj in trajectories:
            if len(traj) < max_len:
                padding = [traj[-1]] * (max_len - len(traj))
                padded_trajectories.append(traj + padding)
            else:
                padded_trajectories.append(traj)
        
        trajectories_array = np.array(padded_trajectories)
        mean_trajectory = np.mean(trajectories_array, axis=0)
        std_trajectory = np.std(trajectories_array, axis=0)
        
        n_trajectories = len(trajectories)
        ci_95 = 1.96 * std_trajectory / np.sqrt(n_trajectories)
        
        steps = range(len(mean_trajectory))
        
        plt.plot(steps, mean_trajectory, label=f'{model_name}', color=colors[i], linewidth=2.5)
        
        plt.fill_between(steps, mean_trajectory - ci_95, mean_trajectory + ci_95, 
                        alpha=0.3, color=colors[i])
    
    avg_time_in_range = {}
    for model_name, metrics in health_metrics.items():
        trajectories = metrics[trajectories_key]
        if trajectories:
            time_in_range_pct = []
            for traj in trajectories:
                in_range = sum(1 for val in traj if target_range[0] <= val <= target_range[1])
                time_in_range_pct.append(100.0 * in_range / len(traj))
            avg_time_in_range[model_name] = np.mean(time_in_range_pct)
    
    annotation_text = "Time in Target Range:\n" + "\n".join(
        [f"{model}: {pct:.1f}%" for model, pct in avg_time_in_range.items()]
    )
    plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.title(f'{condition_name} - {title}', fontsize=16)
    plt.xlabel('Episode Step', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_comparison_with_target_ranges(with_sl_health, without_sl_health, condition_name, metric_type='glucose', save_path=None):
    """Compare models with and without SL guidance with respect to target ranges"""
    plt.figure(figsize=(16, 8))
    
    if metric_type == 'glucose':
        target_range = (90, 130)
        trajectories_key = 'glucose_trajectories'
        ylabel = 'Glucose Level (mg/dL)'
        title = 'Blood Glucose Control Comparison'
    else:
        target_range = (110, 130)
        trajectories_key = 'bp_trajectories'
        ylabel = 'Systolic Blood Pressure (mmHg)'
        title = 'Blood Pressure Control Comparison'
    
    models = list(with_sl_health.keys())
    n_models = len(models)
    
    for i, model_name in enumerate(models):
        plt.subplot(1, n_models, i+1)
        
        if metric_type == 'glucose':
            plt.axhspan(180, 300, alpha=0.2, color='red', label='Hyperglycemia')
            plt.axhspan(130, 180, alpha=0.2, color='orange')
            plt.axhspan(90, 130, alpha=0.3, color='green', label='Target Range')
            plt.axhspan(70, 90, alpha=0.2, color='orange')
            plt.axhspan(0, 70, alpha=0.2, color='red', label='Hypoglycemia')
        else:
            plt.axhspan(140, 250, alpha=0.2, color='red', label='Hypertension')
            plt.axhspan(130, 140, alpha=0.2, color='orange')
            plt.axhspan(110, 130, alpha=0.3, color='green', label='Target Range')
            plt.axhspan(90, 110, alpha=0.2, color='orange')
            plt.axhspan(0, 90, alpha=0.2, color='red', label='Hypotension')
        
        with_sl_trajectories = with_sl_health[model_name][trajectories_key]
        max_len = max(len(traj) for traj in with_sl_trajectories)
        with_sl_padded = []
        
        for traj in with_sl_trajectories:
            if len(traj) < max_len:
                padding = [traj[-1]] * (max_len - len(traj))
                with_sl_padded.append(traj + padding)
            else:
                with_sl_padded.append(traj)
        
        with_sl_array = np.array(with_sl_padded)
        with_sl_mean = np.mean(with_sl_array, axis=0)
        with_sl_std = np.std(with_sl_array, axis=0)
        with_sl_ci = 1.96 * with_sl_std / np.sqrt(len(with_sl_trajectories))
        
        without_sl_trajectories = without_sl_health[model_name][trajectories_key]
        without_sl_padded = []
        
        for traj in without_sl_trajectories:
            if len(traj) < max_len:
                padding = [traj[-1]] * (max_len - len(traj))
                without_sl_padded.append(traj + padding)
            else:
                without_sl_padded.append(traj)
        
        without_sl_array = np.array(without_sl_padded)
        without_sl_mean = np.mean(without_sl_array, axis=0)
        without_sl_std = np.std(without_sl_array, axis=0)
        without_sl_ci = 1.96 * without_sl_std / np.sqrt(len(without_sl_trajectories))
        
        steps = range(len(with_sl_mean))
        
        plt.plot(steps, with_sl_mean, label='With SL', color='blue', linewidth=2)
        plt.fill_between(steps, with_sl_mean - with_sl_ci, with_sl_mean + with_sl_ci, alpha=0.2, color='blue')
        
        plt.plot(steps, without_sl_mean, label='Without SL', color='orange', linewidth=2)
        plt.fill_between(steps, without_sl_mean - without_sl_ci, without_sl_mean + without_sl_ci, alpha=0.2, color='orange')
        
        with_sl_tir = []
        for traj in with_sl_trajectories:
            in_range = sum(1 for val in traj if target_range[0] <= val <= target_range[1])
            with_sl_tir.append(100.0 * in_range / len(traj))
        
        without_sl_tir = []
        for traj in without_sl_trajectories:
            in_range = sum(1 for val in traj if target_range[0] <= val <= target_range[1])
            without_sl_tir.append(100.0 * in_range / len(traj))
        
        plt.annotate(f"Time in Range:\nWith SL: {np.mean(with_sl_tir):.1f}%\nWithout SL: {np.mean(without_sl_tir):.1f}%", 
                    xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.title(f'{model_name}', fontsize=14)
        plt.xlabel('Episode Step', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{condition_name} - {title}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_sl_performance(sl_metrics_across_seeds, save_path=None):
    """Plot supervised learning model performance across seeds"""
    plt.figure(figsize=(14, 10))
    plt.suptitle('Supervised Learning Model Performance', fontsize=16)
    
    metrics = ['accuracy']
    conditions = ['diabetes', 'hypertension']
    colors = {'diabetes': '#FF6B6B', 'hypertension': '#4ECDC4'}
    
    if 'auc' in sl_metrics_across_seeds[0]['diabetes']:
        metrics.append('auc')
    
    n_metrics = len(metrics)
    for m_idx, metric in enumerate(metrics):
        plt.subplot(n_metrics, 1, m_idx + 1)
        
        for condition in conditions:
            metric_values = [seed_metric[condition][metric] for seed_metric in sl_metrics_across_seeds]
            seeds = range(1, len(metric_values) + 1)
            
            plt.plot(seeds, metric_values, marker='o', label=condition.title(), color=colors[condition])
            
            mean_value = np.mean(metric_values)
            plt.axhline(y=mean_value, color=colors[condition], linestyle='--', 
                      alpha=0.7, label=f"{condition.title()} Mean: {mean_value:.3f}")
        
        plt.title(f"{metric.upper()} across seeds")
        plt.xlabel("Seed Index")
        plt.ylabel(metric.upper())
        plt.xticks(seeds)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrices(sl_metrics_across_seeds, save_path=None):
    """Plot confusion matrices for the SL models"""
    avg_cm_diabetes = np.mean([metrics['diabetes']['confusion_matrix'] for metrics in sl_metrics_across_seeds], axis=0)
    avg_cm_hypertension = np.mean([metrics['hypertension']['confusion_matrix'] for metrics in sl_metrics_across_seeds], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(avg_cm_diabetes, annot=True, fmt='.1f', cmap='Blues', ax=ax1)
    ax1.set_title('Diabetes Risk Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_xticklabels(['Low Risk', 'High Risk'])
    ax1.set_yticklabels(['Low Risk', 'High Risk'])
    
    sns.heatmap(avg_cm_hypertension, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title('Hypertension Risk Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_xticklabels(['Low Risk', 'High Risk'])
    ax2.set_yticklabels(['Low Risk', 'High Risk'])
    
    plt.suptitle('SL Model Confusion Matrices (Average Across Seeds)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

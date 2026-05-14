import optuna

# Load the existing study from the database
study = optuna.load_study(
    study_name="mamba-dna-hpo_add", 
    storage="sqlite:///mamba_hpo_add.db"
)

print(f"Best Validation AP: {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

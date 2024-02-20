from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler

def apply_tabnet(train, target, cv=5):
    # Splitting data into features (X) and target variable (y)
    y = target
    X = train

    # Standardize the Features if needed
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize TabNet classifier
    tabnet_model = TabNetClassifier()

    # Add early stopping to TabNet model
    early_stopping = True  # Set to True to enable early stopping
    if early_stopping:
        tabnet_model.fit(
            X_train=X_scaled,
            y_train=y,
            eval_set=[(X_scaled, y)],  # Use the whole dataset for evaluation
            eval_metric=['auc'],  # Use AUC for evaluation
            max_epochs=100,  # Maximum number of training epochs
            patience=20,  # Number of epochs with no improvement after which training will be stopped
            batch_size=1024,  # Batch size for training
            virtual_batch_size=128,  # Virtual batch size for batch normalization
            num_workers=0,  # Number of workers for data loading
            drop_last=False,  # Whether to drop the last incomplete batch
            pin_memory=True,  # Whether to pin memory in DataLoader
        )
        best_tabnet_model = tabnet_model
        best_params = None  # Early stopping does not require hyperparameters
        best_score = tabnet_model.best_cost  # Use the best cost as the score
    else:
        # If early stopping is disabled, return None for best model and score
        best_tabnet_model = None
        best_params = None
        best_score = None

    return best_tabnet_model, best_params, best_score

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tqdm import tqdm



def apply_catboost(df_slice, target, cv=5):

    # Define the parameter grid for CatBoost
    param_grid = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.05, 0.01, 0.1],
        'depth': [4, 6, 8]
    }
    # Splitting data into features (X) and target variable (y)
    y = target
    X = df_slice

    # Standardize the Features if needed
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize CatBoost classifier
    catboost_model = CatBoostClassifier(verbose=False)

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(catboost_model, param_grid=param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(X_scaled, y)

    # Get the best model from the grid search
    best_catboost_model = grid_search.best_estimator_


    return best_catboost_model, grid_search.best_params_, grid_search.best_score_




def apply_lightgbm(train, target, cv=5):
    # Define the parameter grid for LightGBM
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.01, 0.1],
        'num_leaves': [31, 63, 127]
    }
    
    # Splitting data into features (X) and target variable (y)
    y = target
    X = train
    
    # Standardize the Features if needed
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize LightGBM classifier
    lgbm_model = LGBMClassifier()
    
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(lgbm_model, param_grid=param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(X_scaled, y)
    
    # Get the best model from the grid search
    best_lgbm_model = grid_search.best_estimator_
    
    return best_lgbm_model, grid_search.best_params_, grid_search.best_score_



def apply_xgboost(train, target, cv=5):

    # Define the parameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.01, 0.1],
        'max_depth': [4, 6, 8]
    }

    # Splitting data into features (X) and target variable (y)
    y = target
    X = train

    # Standardize the Features if needed
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize XGBoost classifier
    xgb_model = XGBClassifier()

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(xgb_model, param_grid=param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(X_scaled, y)

    # Get the best model from the grid search
    best_xgb_model = grid_search.best_estimator_

    return best_xgb_model, grid_search.best_params_, grid_search.best_score_

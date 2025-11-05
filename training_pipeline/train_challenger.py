import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pathlib
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name = "Hyperparameter Tuning - Random Forest Regressor")
def hp_tuning_rfr(X_train, X_val, y_train, y_val, dv):
    
    mlflow.sklearn.autolog()

    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")
    
    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")

    def objective_rfr(trial: optuna.trial.Trial):
        params = {
            'max_depth': trial.suggest_int("max_depth", 4, 100),
            'n_estimators': trial.suggest_int("n_estimators", 50, 100),
            'criterion': 'squared_error',
        }

        with mlflow.start_run(nested = True):
            mlflow.set_tag('model_family', "Random_Forest_Regressor")
            mlflow.log_params(params)

            rf_model = RandomForestRegressor(**params)
            rf_model.fit(X_train, y_train)

            y_pred = rf_model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            mlflow.log_metric('rmse', rmse)

            signature = infer_signature(X_val, y_pred)

            mlflow.sklearn.log_model(
                rf_model,
                name='model',
                input_example=X_val[:5],
                signature=signature
            )
        
        return rmse
    

    sampler = TPESampler(seed=42)
    rfr_study = optuna.create_study(direction='minimize', sampler=sampler)

    with mlflow.start_run(run_name='Random Forest Regressor Optimized (Optuna)', nested=True):
        rfr_study.optimize(objective_rfr, n_trials=3)
    
    best_params_rfr = rfr_study.best_params

    best_params_rfr['max_depth'] = int(best_params_rfr["max_depth"])
    best_params_rfr['seed'] = 42
    best_params_rfr['criterion'] = 'squared_error'

    return best_params_rfr

@task(name = "Hyperparameter Tuning - SVR")
def hp_tuning_svr(X_train, X_val, y_train, y_val, dv):
    
    mlflow.sklearn.autolog()

    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")
    
    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")

    def objective_svr(trial: optuna.trial.Trial):
        params = {
            'kernel': trial.suggest_categorical("kernel", ['linear','poly','rbf','sigmoid']),
            'epsilon': trial.suggest_float("epsilon", 0, 1),
            'C': trial.suggest_float("C", 1, 2)
        }

        with mlflow.start_run(nested = True):
            mlflow.set_tag('model_family', "Support_Vector_Regressor")
            mlflow.log_params(params)

            svr_model = SVR(**params)
            svr_model.fit(X_train, y_train)

            y_pred = svr_model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            mlflow.log_metric('rmse', rmse)

            signature = infer_signature(X_val, y_pred)

            mlflow.sklearn.log_model(
                svr_model,
                name='model',
                input_example=X_val[:5],
                signature=signature
            )
        
        return rmse
    

    sampler = TPESampler(seed=42)
    svr_study = optuna.create_study(direction='minimize', sampler=sampler)

    with mlflow.start_run(run_name='Random Forest Regressor Optimized (Optuna)', nested=True):
        svr_study.optimize(objective_svr, n_trials=3)
    
    best_params_svr = svr_study.best_params

    best_params_svr['max_depth'] = int(best_params_svr["max_depth"])
    best_params_svr['seed'] = 42
    best_params_svr['criterion'] = 'squared_error'

    return best_params_svr

@task(name = "Train Models")
def train_best_models(X_train, X_val, y_train, y_val, dv, best_params_rfr, best_params_svr) -> None:

    with mlflow.start_run(run_name="Random Forest Regressor Model"):

        mlflow.log_params(best_params_rfr)
        mlflow.set_tags({
            'project': "NYC Taxi Time Prediction Project",
            'optimizer_engine': 'Optuna',
            'model_family': 'Random_Forest_Regressor',
            'feature_set_version': 1
        })


        rfr = RandomForestRegressor(**best_params_rfr)

        rfr.fit(X_train, y_train)

        y_pred_rfr = rfr.predict(X_val)

        rmse_rfr = root_mean_squared_error(y_val, y_pred_rfr)

        mlflow.log_metric('rmse', rmse_rfr)

        pathlib.Path('preprocessor').mkdir(exist_ok=True)
        with open('preprocessor/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact('preprocessor/preprocessor.b', artifact_path='preprocessor')

        feature_names_rfr = dv.get_feature_names_out()
        input_example_rfr = pd.DataFrame(X_val[:5].toarray(), columns=feature_names_rfr)

        signature_rfr = infer_signature(input_example_rfr, y_val[:5])

        mlflow.sklearn.log_model(
            rfr,
            name='model',
            input_example=input_example_rfr,
            signature=signature_rfr
        )
    
    with mlflow.start_run(run_name="SVR"):

        mlflow.log_params(best_params_svr)
        mlflow.set_tags({
            'project': "NYC Taxi Time Prediction Project",
            'optimizer_engine': 'Optuna',
            'model_family': 'Support_Vector_Regressor',
            'feature_set_version': 1
        })


        svr = SVR(**best_params_svr)

        svr.fit(X_train, y_train)

        y_pred_svr = svr.predict(X_val)

        rmse_svr = root_mean_squared_error(y_val, y_pred_svr)

        mlflow.log_metric('rmse', rmse_svr)

        pathlib.Path('preprocessor').mkdir(exist_ok=True)
        with open('preprocessor/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact('preprocessor/preprocessor.b', artifact_path='preprocessor')

        feature_names_svr = dv.get_feature_names_out()
        input_example_svr = pd.DataFrame(X_val[:5].toarray(), columns=feature_names_svr)

        signature_svr = infer_signature(input_example_svr, y_val[:5])

        mlflow.sklearn.log_model(
            svr,
            name='model',
            input_example=input_example_svr,
            signature=signature_svr
        )
    
    return None

@task(name = "Register Model")
def register_model(EXPERIMENT_NAME):

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=['metrics.rmse ASC'],
        output_format='list'
    )

    if len(runs) > 0:
        champion_run = runs[0]
        champion_id = champion_run.info.run_id
        challenger_run = runs[1]
        challenger_id = challenger_run.info.run_id
        print(f'Challenger Run ID: {challenger_id}')
    else:
        print("⚠️ No se encontraron runs con métrica RMSE.")
    
    model_name = 'workspace.default.nyc-taxi-model-prefect'
    result = mlflow.register_model(
        model_uri=f'runs:/{challenger_id}/model',
        name=model_name
    )

    client = MlflowClient()

    model_version = result.version
    new_alias = "Challenger"

    client.set_registered_model_alias(
        name=model_name,
        alias=new_alias,
        version=model_version
    )

    return champion_id, challenger_id

@task(name = 'Compare')
def compare_models(champion_id, challenger_id, df_train) -> None:
    champ_uri = f'runs:/{champion_id}/model'
    chall_uri = f'runs:/{challenger_id}/model'
    
    champ_model = mlflow.pyfunc.load_model(champ_uri)
    chall_model = mlflow.pyfunc.load_model(chall_uri)

    df_reval = read_data('../data/green_tripdata_2025-03.parquet')

    X_train, X_reval, y_train, y_reval, dv = add_features(df_train, df_reval)

    reval_dataset = mlflow.data.from_numpy(X_reval.data, targets = y_reval, name="green_tripdata_2025-03")

    y_champ_preds = champ_model.predict(X_reval)
    y_chall_preds = chall_model.predict(X_reval)

    rmse_champ = root_mean_squared_error(y_reval, y_champ_preds)
    rmse_chall = root_mean_squared_error(y_reval, y_chall_preds)

    if rmse_champ < rmse_chall:
        print('El modelo champion sigue siendo mejor!')
    else:
        new_client = MlflowClient()

        model_name = 'workspace.default.nyc-taxi-model-prefect'
        result_2 = mlflow.register_model(
            
            model_uri=f'runs:/{challenger_id}/model',
            name=model_name
        )

        model_version = result_2.version

        new_client.set_registered_model_alias(
            name=model_name,
            alias="Champion",
            version=model_version
        )
    
    return None

@flow(name="New Main Flow")
def main_flow(year: int, month_train:str, month_val:str) -> None:

    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"

    load_dotenv(override=True)  # Carga las variables del archivo .env
    EXPERIMENT_NAME = "/Users/alfonso.maldonado@iteso.mx/nyc-taxi-experiment-prefect"

    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    best_params_rfr = hp_tuning_rfr(X_train, X_val, y_train, y_val, dv)
    best_params_svr = hp_tuning_svr(X_train, X_val, y_train, y_val, dv)

    train_best_models(X_train, X_val, y_train, y_val, dv, best_params_rfr, best_params_svr)

    champion_id, challegner_id = register_model(EXPERIMENT_NAME)

    compare_models(champion_id, challegner_id, df_train)

if __name__ == '__main__':
    main_flow(year=2025, month_train='01', month_val='02')

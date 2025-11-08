import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings 

warnings.filterwarnings("ignore")


def arguments():
    parser = argparse.ArgumentParser(description="Train a Random Forest Classifier on the provided dataset.")

    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the Random Forest.')
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of the trees.')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum samples required to be at a leaf node.')

    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')

    return parser.parse_args()


def loading_data(test_size, random_state):
    
    df = pd.read_csv('../data/winequality-white.csv', sep=';')
    
    print(f"   Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    X = df.drop('quality', axis=1)
    y = df['quality']

    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    print(f"   Features: {len(features)}")

    return X_train, X_test, y_train, y_test, features


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Importancia de Features - Random Forest', fontsize=14, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importancia', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz guardada: {save_path}")


def train_model(X_train, y_train, params):

    print("\nEntrenando Random Forest.")
    
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print("Modelo entrenado exitosamente")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):

    print("\nEvaluando modelo.")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
        'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
        'test_recall': recall_score(y_test, y_pred_test, average='weighted')
    }
    
    print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"   Test F1-Score:  {metrics['test_f1']:.4f}")
    print(f"   Test Precision: {metrics['test_precision']:.4f}")
    print(f"   Test Recall:    {metrics['test_recall']:.4f}")
    
    overfit_gap = metrics['train_accuracy'] - metrics['test_accuracy']
    if overfit_gap > 0.1:
        print(f"Posible overfitting (gap: {overfit_gap:.4f})")
    
    return metrics, y_pred_test


def main():

    print("="*70)
    print("ENTRENAMIENTO - PREDICCIÓN DE CALIDAD DE VINOS")
    print("="*70)
    
    # 1. Parsear argumentos
    args = arguments()
    
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'test_size': args.test_size,
        'random_state': args.random_state
    }
    
    print(f"\nHiperparámetros:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    

    # Guardar MLflow en la raíz del proyecto (un nivel arriba de project/)
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)  # Subir un nivel
    mlruns_path = os.path.join(project_root, "mlruns")
    
    print(f"\nDirectorio actual: {current_dir}")
    print(f"MLflow tracking en: {mlruns_path}")
    
    # Configurar tracking URI
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("wine_quality_prediction")
    
    experiment = mlflow.get_experiment_by_name("wine_quality_prediction")
    print(f"Experimento: wine_quality_prediction (ID: {experiment.experiment_id})")

    
    # 3. Iniciar run de MLflow
    with mlflow.start_run():

        X_train, X_test, y_train, y_test, feature_names = loading_data(
            test_size=params['test_size'],
            random_state=params['random_state']
        )



        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        

        model = train_model(X_train, y_train, params)

        metrics, y_pred_test = evaluate_model(model, X_train, X_test, y_train, y_test)
        

        mlflow.log_metrics(metrics)

        print("\nCreando visualizaciones.")
        plot_feature_importance(model, feature_names)
        plot_confusion_matrix(y_test, y_pred_test)
        

        mlflow.log_artifact('feature_importance.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        report = classification_report(y_test, y_pred_test)
        with open('classification_report.txt', 'w') as f:
            f.write("REPORTE DE CLASIFICACIÓN\n")
            f.write("="*50 + "\n\n")
            f.write(report)
        mlflow.log_artifact('classification_report.txt')
        

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="WineQualityModel"
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nRun completado exitosamente!")
        print(f"Run ID: {run_id}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        
    print("\n" + "="*70)
    print("ENTRENAMIENTO FINALIZADO")
    print("="*70)



if __name__ == "__main__":
    main()
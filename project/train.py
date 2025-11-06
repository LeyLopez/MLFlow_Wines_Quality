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
import joblib
import warnings 
warnings.filterwarnings("ignore")



def arguments():
    parser = argparse.ArgumentParser(description="Train a Random Forest Classifier on the provided dataset.")

    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the Random Forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the trees.')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum samples required to be at a leaf node.')

    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')

    return parser.parse_args()



def loading_data(test_size, random_state):
    df = pd.read_csv('data/dataset.csv', sep=';')
    X = df.drop('target', axis=1)
    y = df['target']

    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test, features



#Importancia de las características
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
    
    print("Modelo entrenado")
    
    return model



def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
        'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
        'test_recall': recall_score(y_test, y_pred_test, average='weighted')
    }
    
    # Mostrar métricas
    print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"   Test F1-Score:  {metrics['test_f1']:.4f}")
    print(f"   Test Precision: {metrics['test_precision']:.4f}")
    print(f"   Test Recall:    {metrics['test_recall']:.4f}")
    
    # Detectar overfitting
    overfit_gap = metrics['train_accuracy'] - metrics['test_accuracy']
    if overfit_gap > 0.1:
        print(f"Posible overfitting (gap: {overfit_gap:.4f})")
    
    return metrics, y_pred_test




def main():

    print("="*70)
    print("Model Training - Random Forest Classifier for Wine Quality Prediction")
    print("="*70)
    
    # 1. Parsear argumentos
    args = arguments()
    
    # Crear diccionario de parámetros
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
    
    # 2. Configurar MLflow
    mlflow.set_experiment("wine_quality_prediction")
    mlflow.set_tracking_uri("file:./mlruns")
    
    # 3. Iniciar run de MLflow
    with mlflow.start_run():
        
        # 4. Cargar datos
        X_train, X_test, y_train, y_test, feature_names = loading_data(
            test_size=params['test_size'],
            random_state=params['random_state']
        )
        
        # 5. Registrar parámetros en MLflow
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # 6. Entrenar modelo
        model = train_model(X_train, y_train, params)
        
        # 7. Evaluar modelo
        metrics, y_pred_test = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 8. Registrar métricas en MLflow
        mlflow.log_metrics(metrics)
        
        # 9. Crear visualizaciones
        plot_feature_importance(model, feature_names)
        plot_confusion_matrix(y_test, y_pred_test)
        
        # 10. Registrar artifacts en MLflow
        mlflow.log_artifact('feature_importance.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        # Guardar reporte de clasificación
        report = classification_report(y_test, y_pred_test)
        with open('classification_report.txt', 'w') as f:
            f.write("REPORTE DE CLASIFICACIÓN\n")
            f.write("="*50 + "\n\n")
            f.write(report)
        mlflow.log_artifact('classification_report.txt')
        
        # 11. Registrar modelo
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="WineQualityModel"
        )
        
        # 12. Guardar información del run
        run_id = mlflow.active_run().info.run_id
        print(f"   Run ID: {run_id}")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        
    print("\n" + "="*70)
    print("ENTRENAMIENTO FINALIZADO")
    print("="*70)
    print(f"\nPara ver resultados: mlflow ui")


if __name__ == "__main__":
    main()



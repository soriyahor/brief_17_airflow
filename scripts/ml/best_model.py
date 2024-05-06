import os
import mlflow
import tensorflow as tf
from audio_classifier.dataset import RecordingsDataset
from audio_classifier.config import AudioCNNYAMLConfig
from audio_classifier.models import AudioCNNModel

def best_model(
    dataset_path: str,
    n_epochs: int,
    data_batch_size: int,
    n_classes: int,
    correlation_id: str,
    model_yaml_config: str):
    
    # Définir l'URI de suivi MLFlow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_REMOTE_TRACKING_URI"))
    
    # Charger le modèle précédemment entraîné
    model_uri = f"runs:/{correlation_id}/model"
    model = mlflow.keras.load_model(model_uri)

    # Charger le jeu de données de validation
    validation_set = RecordingsDataset(os.path.join(dataset_path, "validation.tfrecords"))
    validation_set = validation_set.get_dataset_as_batches(batch_size=data_batch_size)

    # Évaluer le modèle sur l'ensemble de validation
    metrics = model.evaluate(validation_set, verbose=0)
    
    # Afficher les métriques
    print("Validation Loss:", metrics[0])
    print("Validation Accuracy:", metrics[1])

    # Enregistrer les métriques dans MLFlow
    mlflow.log_metrics({"val_loss": metrics[0], "val_accuracy": metrics[1]})

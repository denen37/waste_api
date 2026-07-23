from django.apps import AppConfig
import os
import json
import tensorflow as tf

class ClassifierConfig(AppConfig):
    name = 'classifier'
    model = None
    class_names = None

    def ready(self):
        model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'waste_classifier.keras')
        class_names_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'class_names.json')

        ClassifierConfig.model = tf.keras.models.load_model(model_path)

        with open(class_names_path, 'r') as f:
            ClassifierConfig.class_names = json.load(f)

        print("Model loaded successfully.")
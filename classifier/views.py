import numpy as np
from PIL import Image
from django.apps import apps
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

IMG_SIZE = (224, 224)  # match whatever size you trained on

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # match your training normalization
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@api_view(['POST'])
def predict_waste(request):
    if 'image' not in request.FILES:
        return Response(
            {"error": "No image file provided. Use key 'image'."},
            status=status.HTTP_400_BAD_REQUEST
        )

    image_file = request.FILES['image']

    try:
        img_array = preprocess_image(image_file)
    except Exception as e:
        return Response(
            {"error": f"Invalid image file: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST
        )

    classifier_config = apps.get_app_config('classifier')
    model = classifier_config.model
    class_names = classifier_config.class_names

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    # Return all class probabilities too, useful for debugging/UI
    all_probs = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    return Response({
        "predicted_class": class_names[predicted_index],
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs
    })

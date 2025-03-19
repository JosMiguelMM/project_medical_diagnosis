from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .classifier import SymptomClassifier
from .serializers import PredictionSerializer  # Importa el serializador
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json

@api_view(['POST'])
def predict_symptom(request):
    """
    API endpoint para realizar predicciones de diagnóstico.
    """
    #Usa el serializador
    serializer = PredictionSerializer(data=request.data)
    if serializer.is_valid():
        try:
            classifier = SymptomClassifier()
            #No es necesario cargar datos si ya se cargo el modelo, se hace en el metodo predict.
            prediction, probability = classifier.predict(serializer.validated_data)
            return Response({'diagnosis': prediction, 'probability': probability}, status=status.HTTP_200_OK)
        except FileNotFoundError as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({'error': 'Error al realizar la predicción: ' + str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@csrf_exempt #Solo para propositos de desarrollo
def train_model_view(request):
  """
    Vista para entrenar modelo. Recibe datos directamente.
  """
  if request.method == 'POST':
    try:
      #Intenta obtener los datos como JSON
      try:
        data = json.loads(request.body.decode('utf-8'))
      except json.JSONDecodeError:
          return JsonResponse({'error': 'Datos JSON inválidos'}, status=400)

      #Verifica si es una lista
      if not isinstance(data, list):
        return JsonResponse({'error': 'Se esperaba una lista de diccionarios'}, status=400)

      df = pd.DataFrame(data)

      #Verifica que el DF no esté vacío
      if df.empty:
        return JsonResponse({'error': "No hay datos"}, status=400)


      #Entrena el modelo
      classifier = SymptomClassifier()
      #Usa un try/except para manejar cualquier error.
      try:
        #No necesitamos pasar la ruta al archivo, pasamos directamente el DataFrame.
        classifier.train(data_path=df, target_column='diagnostico') #Se entrena con el dataframe en memoria.
        # ELIMINA ESTA LÍNEA:
        # classifier.model.fit(df.drop(columns=['diagnostico']), df['diagnostico'])

      except KeyError:
          return JsonResponse({'error': "Asegúrate de que la columna 'diagnostico' esté presente"}, status=400)
      except Exception as e:
        return JsonResponse({'error': f'Error durante el entrenamiento: {str(e)}'}, status=500)


      return JsonResponse({'message': 'Modelo entrenado exitosamente'}, status=200)
    except Exception as e:
      return JsonResponse({'error': str(e)}, status=400)
  return JsonResponse({'error': "Metodo no permitido"}, status=405)

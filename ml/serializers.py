from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    """
    Serializador para la entrada de la API de predicción.
    """
    # Define los campos que la API espera recibir
    fiebre = serializers.IntegerField(required=False)
    tos = serializers.IntegerField(required=False)
    dolor_garganta = serializers.IntegerField(required=False)
    estornudos = serializers.IntegerField(required=False)
    dificultad_respirar = serializers.IntegerField(required=False)
    # Agrega aquí más campos según los síntomas que uses
    # ...

    def create(self, validated_data):
        #Este metodo no se usara, pero hay que definirlo
         return validated_data

    def update(self, instance, validated_data):
       #Este metodo no se usara, pero hay que definirlo
        return instance
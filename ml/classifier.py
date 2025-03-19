import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .pipelines.preprocessing import create_preprocessing_pipeline # Importa el pipeline
import joblib  # Para guardar y cargar el modelo
from sklearn.metrics import accuracy_score

class SymptomClassifier:
    def __init__(self, model_path='ml/models/random_forest_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None #Preprocesador

    def train(self, data_path, target_column='diagnostico', test_size=0.2, random_state=42):
        """
        Entrena el modelo de clasificación.

        Args:
            data_path (str): Ruta al archivo CSV con los datos de entrenamiento.
            target_column (str): Nombre de la columna con las etiquetas (diagnósticos).
            test_size (float): Proporción de datos para el conjunto de prueba.
            random_state (int): Semilla aleatoria para la reproducibilidad.
        """
        # Carga los datos
        data = pd.read_csv(data_path)

        # Separa características (X) y etiquetas (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # --- Identificar tipos de columnas ---
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

        # Crea el pipeline de preprocesamiento
        self.preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features, X) # Pasa X al pipeline.

        # Divide los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


        #Aplica el preprocesamiento
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test) #Solo transform

        # Inicializa y entrena el modelo (Random Forest)
        self.model = RandomForestClassifier(random_state=random_state)
        self.model.fit(X_train_processed, y_train)

        # --- Evaluación básica ---
        y_pred = self.model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy en el conjunto de prueba: {accuracy}")

        # Guarda el modelo y el preprocesador
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.preprocessor, self.model_path + '_preprocessor.pkl') # Guarda el preprocesador



    def load_model(self):
        """Carga el modelo entrenado desde el archivo."""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.model_path + '_preprocessor.pkl') #Carga el preprocesador
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path} ni su preprocesador.")


    def predict(self, data):
        """
        Realiza predicciones sobre nuevos datos.

        Args:
            data (pd.DataFrame o dict):  Los datos de entrada.  Debe tener las mismas
                                        columnas que los datos de entrenamiento
                                        (excepto la columna de diagnóstico).

        Returns:
            str: La clase predicha.
            float: la probabilidad de la clase predicha.
        """
        if self.model is None:
            self.load_model()


        # Asegura que 'data' sea un DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])  # Crea un DataFrame de una sola fila

        # Preprocesa los datos de entrada
        data_processed = self.preprocessor.transform(data)
        # Realiza la predicción
        prediction = self.model.predict(data_processed)[0]
        probability = self.model.predict_proba(data_processed)[0].max() #Probabilidad

        return prediction, probability
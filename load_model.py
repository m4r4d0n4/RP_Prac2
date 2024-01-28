from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os

# Cargar el modelo entrenado
model = load_model('/usr/src/app/inception.keras')  # Ajusta la ruta si es necesario

# Directorio que contiene las imágenes para inferencia
inference_dir = '/usr/src/app/test'  # Ajusta la ruta si es necesario

# Obtener la lista ordenada de nombres de archivos en el directorio de inferencia
image_files = sorted([f for f in os.listdir(inference_dir) if os.path.isfile(os.path.join(inference_dir, f))])

# Mapeo de etiquetas numéricas a nombres de clase
class_mapping = {0: 'bosque', 1: 'setas', 2: 'hierba', 3: 'hojas', 4: 'ensalada'}

# Iterar sobre cada imagen y aplicar el modelo
output_file_path = '/usr/src/app/test/Competicion2.txt'

with open(output_file_path, 'w') as file:
    #Las fotos estan ordenadas por orden alfabetico de menor a mayor

    for image_file in image_files:
        # Cargar la imagen y preprocesarla
        img_path = os.path.join(inference_dir, image_file)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizar porque nuestro preprocesado de datos lo hace

        # Hacer predicción en la imagen
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Obtener el nombre de la clase a partir del mapeo
        class_name = class_mapping[predicted_class]

        # Escribir los resultados en el archivo
        file.write(f"{class_name}\n")

print(f'Resultados de inferencia escritos en {output_file_path}')

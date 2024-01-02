### Función desarrollada para realizar aumentado de datos sobre un conjunto de imagenes ###

from PIL import Image
import Augmentor
import os

# Función para filtrar las imagenes por dimensión
def filtrar_imagenes(ruta_carpeta):
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(ruta_carpeta)

    for archivo in archivos:
        # Obtener la ruta completa del archivo
        ruta_archivo = os.path.join(ruta_carpeta, archivo)

        try:
            # Abrir la imagen y obtener sus dimensiones
            with Image.open(ruta_archivo) as img:
                ancho, alto = img.size

                # Verificar si las dimensiones son 150x150
                if ancho != 150 or alto != 150:
                    # Redimensionar la imagen a 150x150
                    img = img.resize((150, 150))

                # Verificar si la imagen tiene 3 canales
                if len(img.getbands()) != 3:
                    # Convertir la imagen a modo RGB
                    img = img.convert('RGB')
                    print(ruta_archivo)

                # Guardar la imagen modificada
                img.save(ruta_archivo)

        except Exception as e:
            # Manejar errores al abrir o guardar la imagen
            print(f"Error al procesar {ruta_archivo}: {e}")


def data_augmentation(input_folder, output_folder, num_augmented_images):
    # Crear un objeto Pipeline de Augmentor
    pipeline = Augmentor.Pipeline(input_folder, output_folder)

    # Definir operaciones de data_augmentation
    pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom_random(probability=0.5, percentage_area=0.8)
    pipeline.flip_top_bottom(probability=0.5)
    pipeline.crop_random(probability=0.7, percentage_area=0.8)
    pipeline.random_brightness(probability=0.7, min_factor=0.8, max_factor=1.2)

    # Ejecutar la aumentación de datos
    pipeline.sample(num_augmented_images)

if __name__ == "__main__":

    # Ruta de la carpeta de imágenes para revisar dimension
    ruta_carpeta_imagenes = "C:\\Users\\anita\\Jupyter\\Practica 2 RP\\train\\forest - copia"
    
    # Filtrar las imágenes por dimension
    filtrar_imagenes(ruta_carpeta_imagenes)
    
    # Ruta de las imágenes originales
    input_folder = "C:\\Users\\anita\\Jupyter\\Practica 2 RP\\prueba"

    # Ruta para guardar las imágenes aumentadas
    output_folder = "C:\\Users\\anita\\Jupyter\\Practica 2 RP\\salida"

    # Número de imágenes que queremos generar
    num_augmented_images = 100

    # Verificar que existe la carpeta de salida sino se crea
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Aplicar data_augmentation para las imagenes originales
    data_augmentation(input_folder, output_folder, num_augmented_images)

    print("Aumentación de datos completada.")

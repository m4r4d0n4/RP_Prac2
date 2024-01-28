from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from PIL import Image
import os
import tensorflow as tf
def check_images(folder_path):
    extensions = []
    for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            print('** Path: {}  **'.format(file_path))
            
            try:
                im = Image.open(file_path)
                rgb_im = im.convert('RGB')
            except:
                print(f'Error al convertir la imagen {file_path} a RGB')
                
                # Eliminar el archivo si hay un error en la conversión
                os.remove(file_path)
                print(f'Archivo {file_path} eliminado.')
            if filee.split('.')[1] not in extensions:
                extensions.append(filee.split('.')[1])
def check_test(test_path:str):
    for filee in os.listdir(test_path):
            file_path = os.path.join(test_path, filee)
            print('** Path: {}  **'.format(file_path))
            
            try:
                im = Image.open(file_path)
                rgb_im = im.convert('RGB')
            except:
                print(f'Error al convertir la imagen {file_path} a RGB')
                
                # Eliminar el archivo si hay un error en la conversión
                os.remove(file_path)
                print(f'Archivo {file_path} eliminado.')

def is_valid_image(filename):
    try:
        img = Image.open(filename)
        img.verify()
        return True
    except (IOError, SyntaxError) as e:
        if "truncated" in str(e):
            print(f'La imagen {filename} está truncada y será ignorada.')
            return False
        else:
            print(f'Error en la imagen {filename}: {e}')
            return False
        
def read_train(train_dir:str):

    # Parámetros
    batch_size = 32
    img_size = (150, 150)

    # Utilizamos ImageDataGenerator para cargar y preprocesar las imágenes
    datagen = ImageDataGenerator(rescale=1./255,    
    rotation_range=20,      # Rango de rotación aleatoria en grados
    width_shift_range=0.2,  # Rango de desplazamiento horizontal aleatorio
    height_shift_range=0.2, # Rango de desplazamiento vertical aleatorio
    shear_range=0.2,        # Rango de deformación
    zoom_range=0.2,         # Rango de zoom aleatorio
    horizontal_flip=True,   # Volteo horizontal aleatorio
    fill_mode='nearest',     # Estrategia de relleno para píxeles fuera de los límites de la imagen
    validation_split=0.2)
    # Filtrar archivos no válidos
    valid_images = [f for f in os.listdir(train_dir) if is_valid_image(os.path.join(train_dir, f))]
    # Crear generadores de datos de entrenamiento y validación
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=valid_images
    )

    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',   
        classes=valid_images
    )
    return train_generator, validation_generator
'''
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)'''
check_images("./train")
# Crear el modelo secuencial
model = Sequential()

# Capa convolucional 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

# Capa convolucional 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Capa convolucional 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Capa convolucional 4
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Aplanar la salida para conectarla a una capa densa
model.add(Flatten())

# Capa densa 1
model.add(Dense(512, activation='relu'))

# Capa de salida
model.add(Dense(5, activation='softmax'))
# Paso 4: Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Paso 5: Mostrar un resumen del modelo
model.summary()


train_generator, validation_generator = read_train("./train")
# Número de clases
num_classes = len(train_generator.class_indices)
batch_size = train_generator.batch_size
# Codificar las etiquetas en formato one-hot
train_labels = to_categorical(train_generator.classes, num_classes=num_classes)
validation_labels = to_categorical(validation_generator.classes, num_classes=num_classes)

# Entrenar el modelo
epochs = 2000  # Puedes ajustar este número según sea necesario
plot_model(model,to_file="plot_cnn.png", show_shapes=True)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Guardar el modelo en formato HDF5
model.save('model_cnn.keras')

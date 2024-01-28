from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from PIL import Image
from keras.applications import InceptionV3
from keras.applications import ResNet50
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os



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

#Para GPU RTX hay que poner esto
#Descomentar si se usa la gpu
#devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0], True)

check_images("./train")
#Cargar el modelo preentrenado (InceptionV3 en este caso)
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

#Congelar las capas convolucionales de la VGG16
for layer in base_model.layers:
    layer.trainable = False

#Crear el modelo secuencial con la VGG16 como base
model = Sequential()

#Añadir la base VGG16
model.add(base_model)

# Añadir una capa convolucional adicional antes de Flatten
'''model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))'''

# Añadir un cabezal personalizado para la clasificación
#model.add(Flatten())

#model.add(Dense(256, activation='relu'))
model.add(GlobalAveragePooling2D())  # Utiliza Global Average Pooling para reducir el número de parámetros

model.add(Dense(4096, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5 clases para la clasificación

#Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Mostrar un resumen del modelo
model.summary()
plot_model(model,to_file="plot_vgg.png", show_shapes=True)

train_generator, validation_generator = read_train("./train")
# Número de clases
num_classes = len(train_generator.class_indices)
batch_size = train_generator.batch_size
# Codificar las etiquetas en formato one-hot
train_labels = to_categorical(train_generator.classes, num_classes=num_classes)
validation_labels = to_categorical(validation_generator.classes, num_classes=num_classes)

#Entrenar el modelo
epochs = 100  # Puedes ajustar este número según sea necesario

# Se muestra la arquitectura de la red
plot_model(model,to_file="plot_vgg.png", show_shapes=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Guardar el modelo
model.save('/usr/src/app/train/model.keras')

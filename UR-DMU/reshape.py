import os
import glob
from PIL import Image

# Directorio que contiene las carpetas con los frames
base_dir = r"C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/UCF_Crime_Frames"

# Función para procesar las imágenes
def process_images():
    # Recorrer todas las carpetas dentro de la carpeta base
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if os.path.isdir(folder_path):
            # Obtener todos los archivos de imágenes en la carpeta
            image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

            # Iterar sobre las imágenes, guardando solo 1 de cada 15
            for i, image_file in enumerate(image_files):
                if i % 15 == 0:  # Mantener solo 1 de cada 15
                    continue
                else:
                    # Borrar el resto de las imágenes
                    os.remove(image_file)
        print(f"Processed folder {folder}")

# Ejecutar la función
if __name__ == "__main__":
    process_images()

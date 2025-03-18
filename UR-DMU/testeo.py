import os

# Ruta al archivo .list
list_file = 'list/UCF_Train.list'

# Ruta a la carpeta principal de 'train'
train_folder = 'UCF_Crime_Frames/train'

# Leer la lista de archivos desde el .list
with open(list_file, 'r') as f:
    file_list = f.read().splitlines()

# Archivos esperados para el set de entrenamiento
expected_folders  = set()

# Procesar cada línea en la lista
for file_path in file_list:
    # Extraer el nombre de la subcarpeta sin ruta y sin extensión .npy
    file_name = file_path.split('train/')[-1].split('_x264')[0]
    expected_folders.add(file_name)

# Subcarpetas encontradas en la carpeta 'train'
found_folders = set()

# Recorrer las subcarpetas dentro de cada categoría (Abuse, Arrest, etc.)
for category_folder in os.listdir(train_folder):
    category_path = os.path.join(train_folder, category_folder)
    
    # Solo queremos procesar las carpetas, no archivos sueltos
    if os.path.isdir(category_path):
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            found_folders.add(subfolder)
            
            # Verificar si el nombre de la subcarpeta está en la lista esperada
            # if subfolder not in expected_folders:
            #     # Eliminar todos los archivos dentro de la subcarpeta
            #     for file in os.listdir(subfolder_path):
            #         full_file_path = os.path.join(subfolder_path, file)
            #         os.remove(full_file_path)
            #     print(f'Archivo eliminado: {subfolder}')
                
            #     # Eliminar la subcarpeta vacía
            #     os.rmdir(subfolder_path)
            #     print(f'Subcarpeta eliminada: {subfolder_path}')

# Comparar subcarpetas esperadas con subcarpetas encontradas
missing_folders = expected_folders - found_folders

# Reportar subcarpetas que faltan
if missing_folders:
    print('Subcarpetas faltantes que están en la lista pero no en las carpetas:')
    for folder in missing_folders:
        print(f'- {folder}')
else:
    print('No faltan subcarpetas. Todas las subcarpetas de la lista están presentes.')

print('Proceso completado.')

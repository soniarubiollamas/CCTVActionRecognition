from tqdm import tqdm
import numpy as np
import os

def ten2one(source,dst):
    features=os.listdir(source)
    # Filtrar solo los archivos que terminan en '.npy'
    npy_files = [feature for feature in features if feature.endswith('.npy')]
    
    for feature in tqdm(npy_files):
        data=np.load(os.path.join(source,feature))
        for i in range(10):
            np.save("{}/{}_{}.npy".format(dst,feature.split(".npy")[0],i),data[i])

ten2one("C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/feature_extract2/UCF_ten/30fps/test","C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/feature_extract2/UCF_one/30fps/test")


###########################################################################################################################################################
# import os
# import shutil

# def reorganize_videos(base_dir):
#     # Iterate through the base directory (Abuse, Arrest, etc.)
#     for folder in os.listdir(base_dir):
#         folder_path = os.path.join(base_dir, folder)
        
#         if os.path.isdir(folder_path):
#             # Iterate through all files in the subfolder (images)
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith(".png"):
#                     # Extract the video name (e.g., 'Abuse001') from the file name
#                     video_name = file_name.split("_x264")[0]
#                     video_folder = os.path.join(folder_path, video_name)
                    
#                     # Create the subfolder for the video if it doesn't exist
#                     if not os.path.exists(video_folder):
#                         os.makedirs(video_folder)
                    
#                     # Move the image file to the corresponding video folder
#                     src_file = os.path.join(folder_path, file_name)
#                     dst_file = os.path.join(video_folder, file_name)
#                     shutil.move(src_file, dst_file)

# # Provide the base directory where the folders (Abuse, Arrest, etc.) are located
# base_directory = "C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/UCF_Crime_Frames/train"
# reorganize_videos(base_directory)

###########################################################################################################################################################

# import os

# def subcarpetas_no_en_lista(directorio_principal, archivo_list):
#     # Cargar los nombres del archivo .list
#     with open(archivo_list, 'r') as f:
#         nombres_en_lista = set(line.strip() for line in f.readlines())
#         lista = []
#         for nombres in nombres_en_lista:
#             nombre = nombres.split("/")[-1].split("_x264")[0]
#             if nombre in lista:
#                 continue
#             else:
#                 lista.append(nombre)


#     subcarpetas_no_en_list = []

#     # Recorre cada carpeta dentro del directorio principal
#     for carpeta in os.listdir(directorio_principal):
#         carpeta_path = os.path.join(directorio_principal, carpeta)

#         # Verifica si es un directorio
#         if os.path.isdir(carpeta_path):
#             # Recorre las subcarpetas dentro de cada carpeta
#             for subcarpeta in os.listdir(carpeta_path):
#                 subcarpeta_path = os.path.join(carpeta_path, subcarpeta)
                
#                 # Verifica si la subcarpeta es un directorio
#                 if os.path.isdir(subcarpeta_path):
#                     # Si la subcarpeta no está en el archivo .list, se añade a la lista
#                     if subcarpeta not in lista:
#                         subcarpetas_no_en_list.append(subcarpeta)

#     return subcarpetas_no_en_list

# # Ruta de tu carpeta principal y del archivo .list
# directorio_principal = "C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/UCF_Crime_Frames/train"
# archivo_list = "C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/list/UCF_Train.list"

# # Ejecuta la función y muestra las subcarpetas que no están en el archivo .list
# subcarpetas_faltantes = subcarpetas_no_en_lista(directorio_principal, archivo_list)
# print("Subcarpetas que no están en el archivo .list:")
# for subcarpeta in subcarpetas_faltantes:
#     print(subcarpeta)



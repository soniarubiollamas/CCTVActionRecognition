# Ruta del archivo de texto que contiene los nombres de archivos a eliminar
files_to_remove_file_path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/files_to_remove.txt'
files_to_keep_path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/mini_test.txt'

# Ruta del archivo .list original
input_file_path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/test.list'
# Ruta del archivo .list filtrado
output_file_path = 'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/list/ucf/minitest.list'

# Leer los nombres de archivos desde el archivo de texto
remove = False
keep = True

if remove:

    files_to_remove = []
    with open(files_to_remove_file_path, 'r') as f:
        for line in f:
            file = line.split("_x264")[0]
            files_to_remove.append(file)


    # Crear un conjunto para búsqueda rápida, considerando solo la parte base
    files_to_remove_base = {file.split('__')[0] for file in files_to_remove}


    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Comprobar si la línea contiene un archivo a eliminar
            if not any(base_file in line for base_file in files_to_remove_base):
                outfile.write(line)
    print(f"Filtrado completado. Se han guardado las líneas restantes en {output_file_path}.")


if keep:
    files_to_keep = []
    

    with open(files_to_keep_path, 'r') as f:
        for line in f:
            file = line.split("\n")[0]
            files_to_keep.append(file)

    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Comprobar si la línea contiene un archivo a mantener
            if any(base_file in line for base_file in files_to_keep):
                outfile.write(line)


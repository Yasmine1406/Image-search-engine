import os
import shutil

# Specify the source folder containing the images
source_folder = 'D:\Tbourbi\ImageSearchEngine\archive\image\zebra'

# Specify the destination folder where renamed images will be saved
destination_folder = 'D:\ProjetCBIR\Data'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get a list of all files in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Loop through the files and rename them
for i, filename in enumerate(image_files, start=(11652)):

    file_extension = os.path.splitext(filename)[1]
    
    new_filename = f'image{i}{file_extension}'
    
    shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, new_filename))

print("Images renamed and moved successfully!")

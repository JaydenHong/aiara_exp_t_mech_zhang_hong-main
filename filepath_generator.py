import os

# File path Processing

# Base folder name
def filepath(base_folder_name = 'saved_data'):
    # Find existing folders that match the base folder name
    existing_folders = [folder for folder in os.listdir('.') if folder.startswith(base_folder_name)]

    # Find the highest numbered folder among the existing matching folders
    folder_numbers = [int(folder[len(base_folder_name):]) for folder in existing_folders
                      if folder[len(base_folder_name):].isdigit()]
    max_folder_number = max(folder_numbers) + 1 if folder_numbers else 0
    FILE_PATH = f'{base_folder_name}{max_folder_number:02}'
    return FILE_PATH, max_folder_number

if __name__ == "__main__":
    print(filepath())
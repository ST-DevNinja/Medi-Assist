import os

def print_directory_structure(root_dir, indent=""):
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}/")
            print_directory_structure(path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

print_directory_structure("MEDI-ASSIST")  # replace with your actual folder if needed

import os
from PIL import Image
import pillow_heif


class HeicConverterPng:
    @staticmethod
    def convert_heic_to_png_and_replace(input_path: str) -> bool:
        """Converts HEIC file to PNG and replaces it"""
        try:
            if not os.path.exists(input_path):
                print(f"❌ File not found: {input_path}")
                return False
            if not input_path.lower().endswith(".heic"):
                print(f"❌ Not a HEIC file: {input_path}")
                return False
            heif_image = pillow_heif.open_heif(input_path)
            image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
            output_path = os.path.splitext(input_path)[0] + ".png"
            image.save(output_path, "PNG")
            os.remove(input_path)
            print(f"✔ Converted & Replaced: {input_path} → {output_path}")
            return True
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            return False

    @staticmethod
    def convert_all_heic_in_folder(folder_path: str) -> list:
        """Converts all HEIC files in the folder to PNG and returns failed conversions"""
        failed_files = []
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".heic"):
                    input_file = os.path.join(folder_path, filename)
                    success = HeicConverterPng.convert_heic_to_png_and_replace(
                        input_file
                    )
                    if not success:
                        failed_files.append(filename)
            return failed_files
        except Exception as e:
            print(f"Error converting files in {folder_path}: {e}")
            return None

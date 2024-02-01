import os
from PIL import Image

def resize_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Resize images in the input directory and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                try:
                    # Open the image file
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path)
                    
                    # Resize the image
                    resized_image = image.resize(target_size)
                    
                    # Save the resized image
                    output_path = os.path.join(output_dir, file)
                    resized_image.save(output_path)
                    
                    print(f"Resized: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
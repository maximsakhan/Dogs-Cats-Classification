import os
from cnnClassifier.utils.image_utils import resize_images
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

STAGE_NAME = "Image Resizing Stage"

class ImageResizingTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_ingestion_config = self.config.get_data_ingestion_config()

    def main(self):
        try:
            logger.info("Image Resizing Stage started")

            # Specify input and output directories for cats and dogs
            cats_input_dir = os.path.join(self.data_ingestion_config.root_dir, 'cats')
            dogs_input_dir = os.path.join(self.data_ingestion_config.root_dir, 'dogs')
            cats_output_dir = os.path.join(self.data_ingestion_config.root_dir, 'cats_resized')
            dogs_output_dir = os.path.join(self.data_ingestion_config.root_dir, 'dogs_resized')

            # Create output directories if they don't exist
            os.makedirs(cats_output_dir, exist_ok=True)
            os.makedirs(dogs_output_dir, exist_ok=True)

            # Resize images for cats and dogs
            resize_images(cats_input_dir, cats_output_dir)
            resize_images(dogs_input_dir, dogs_output_dir)

            logger.info("Image Resizing Stage completed\n")
        except Exception as e:
            logger.exception(f"Error in Image Resizing Stage: {str(e)}")
            raise e


if __name__ == '__main__':
    try:
        resizing_pipeline = ImageResizingTrainingPipeline()
        resizing_pipeline.run()
    except Exception as e:
        logger.exception(f"Error in Image Resizing Pipeline: {str(e)}")
        raise e

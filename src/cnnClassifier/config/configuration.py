from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                DataTransformationConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)
import os
import shutil
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            image_size=self.params.IMAGE_SIZE,
            mean=self.params.MEAN,
            std=self.params.STD
        )

        return data_transformation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config
    
    @staticmethod
    def merge_directories(source_directories, destination_directory):
        """
        Merges the contents of multiple source directories into separate subdirectories within the destination directory.
        
        Args:
        - source_directories (list of str): List of source directory paths.
        - destination_directory (str): Path to the destination directory.
        """
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)
        
        # Iterate through source directories
        for source_dir in source_directories:
            # Get the base name of the source directory
            source_base_name = os.path.basename(source_dir)
            # Construct the destination directory path for the source directory
            destination_source_dir = os.path.join(destination_directory, source_base_name)
            # Create the destination source directory if it doesn't exist
            os.makedirs(destination_source_dir, exist_ok=True)
            
            # Iterate through files in the source directory
            for filename in os.listdir(source_dir):
                # Construct the source and destination file paths
                source_file_path = os.path.join(source_dir, filename)
                destination_file_path = os.path.join(destination_source_dir, filename)
                # Copy the file to the destination source directory
                shutil.copy(source_file_path, destination_file_path)

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        cats_resized_dir = os.path.join(self.config.data_ingestion.unzip_dir, "cats_resized")
        dogs_resized_dir = os.path.join(self.config.data_ingestion.unzip_dir, "dogs_resized")        
        training_data_dir = os.path.join(self.config.data_ingestion.unzip_dir, "training_data")

        create_directories([
            Path(training.root_dir),
            Path(cats_resized_dir),
            Path(dogs_resized_dir),
            Path(training_data_dir)
        ])

        self.merge_directories([cats_resized_dir, dogs_resized_dir], training_data_dir)

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_patience=params.PATIENCE
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/model.h5"),
            training_data=Path("artifacts/data_ingestion/training_data"),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
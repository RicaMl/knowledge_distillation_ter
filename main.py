from src.utils.captionning_manager import  *
from src.utils.data_manager import *
from src.utils.new_captionning_manager import *
from pathlib import Path


def main():
    # generate_uuid_csv_valid()
    # generate_csv_uuids_info_for_val_images()
    #generate_csv_uuids_info_for_train_images()
    #generate_csv_uuids_info_for_test_images()

    #generate_captions_via_llm()
    #generate_captions_via_vlm()

    #genarate_captionning_and_csvs_train()
    #genarate_captionning_and_csvs_test()
    #prepare_test_csv_for_clip(
    #    csv_path='captions_ecology_test.csv',
    #    images_dir=Path('./src/data/Fathomnet-by-Phylum-2/test/images'),
    #    output_path='./src/data/csvfiles/test/captions_ecology_test_clip.csv'
    #)
    #prepare_test_csv_for_clip(
    #    csv_path='captions_morphology_test.csv',
    #    images_dir='./src/data/Fathomnet-by-Phylum-2/test/images',
    #    output_path='./src/data/csvfiles/test/captions_morphology_test_clip.csv'
    #)

    #prepare_train_csv_augmented(
    #    captions_csv='captions_ecology_train.csv',
    #    train_images_dir=Path('./src/data/Fathomnet-by-Phylum-2/valid/images'),
    #    output_path='./src/data/csvfiles/val/captions_ecology_val_clip.csv'
    #)

    prepare_train_csv_augmented(
        captions_csv='captions_morpho_val.csv',
        train_images_dir=Path('./src/data/Fathomnet-by-Phylum-2/images/original'),
        output_path='./src/data/csvfiles/val/captions_morphology_val_clip_original.csv'
    )

    #prepare_train_csv_augmented(
    #    captions_csv='captions_ecology_train.csv',
    #    train_images_dir=Path('./src/data/Fathomnet-by-Phylum-2/train/images'),
    #    output_path='./src/data/csvfiles/train/captions_ecology_train_clip.csv'
    #)

    prepare_train_csv_augmented(
        captions_csv='captions_morpho_train.csv',
        train_images_dir=Path('./src/data/Fathomnet-by-Phylum-2/images/original'),
        output_path='./src/data/csvfiles/train/captions_morphology_train_clip_original.csv'
    )

    #prepare_train_csv_augmented(
    #   captions_csv='./src/data/csvfiles/llm_captions_train.csv',
    #   train_images_dir=Path('./src/data/Fathomnet-by-Phylum-2/valid/images'),
    #   output_path='./src/data/csvfiles/val/captions_morphology_val_llm_clip.csv'
    #)

    #prepare_train_csv_augmented(
    #   captions_csv='./src/data/csvfiles/llm_captions_train.csv',
    #   train_images_dir=Path('./src/data/Fathomnet-by-Phylum-2/train/images'),
    #   output_path='./src/data/csvfiles/train/captions_morphology_train_llm_clip.csv'
    #)

    #genarate_captionning_and_csvs_train2()
    #generate_llm()
    #genarate_captionning_and_csvs_test2()


    print("Hello World")


if __name__ == "__main__":
    main()
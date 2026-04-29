import csv
from pathlib import Path
import requests
from fathomnet.api import images as imagesFathomnet, taxa
from PIL import Image
import pandas as pd
import io


def generate_uuid_csv(directory_path, output_csv_name):
    path = Path(directory_path)
    if not path.exists():
        print(f"Directory '{directory_path}' doesn't exist")
        return
    #uuid extraction
    uuids = [f.stem[:36] for f in path.iterdir() if f.is_file()]
    uuids = set(uuids)
    verified_uuids = []
    print(f"Start Verified uuid from fathomnet")
    with open(output_csv_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writerow(['uuid'])
        for uuid in uuids:
            writer.writerow([uuid])

    print(f"{output_csv_name} created")

def get_taxamony(concept):
    providers = taxa.list_taxa_providers()
    provider = providers[0] if providers else 'fathomnet'
    res = {
        "phylum": "N/A", "class": "N/A", "order": "N/A",
        "family": "N/A", "genus": "N/A", "rank": "unknown",
    }
    #clean name is for concept who comes with cf. or sp. because of incertitude
    clean_name = concept.replace(" cf. ", " ").replace(" sp. ", " ").strip()
    try:
        results = taxa.find_taxa(provider, clean_name)
        if results:
            match = next((t for t in results if t.name.lower() in clean_name.lower()), results[0])
            if match and match.rank:
                init_rank = str(match.rank).lower().strip()
                res["rank"] = init_rank
                if init_rank in res:
                    res[init_rank] = match.name if match.name else "N/A"
        current = clean_name
        while True:
            parent = taxa.find_parent(provider, current)
            if not parent or parent.name == current:
                break
            p_rank = str(parent.rank).lower().strip() if parent.rank else ""
            if p_rank in res:
                res[p_rank] = str(parent.name) if parent.name else "N/A"
            current = parent.name
            if p_rank == 'phylum' or parent.name == 'Animalia':
                break
    except:
        pass
    return res







def generate_uuid_csv_test():
    try:
        test_path = "src/data/Fathomnet-by-Phylum-2/test/images"
        print("Start generating uuid csv for test images uuids")
        generate_uuid_csv(test_path, "src/data/csvfiles/test_uuids.csv")
        print("End generating uuid csv for test images uuids")
    except Exception as e:
        print(e)

def generate_uuid_csv_train():
    try:
        train_path = "src/data/Fathomnet-by-Phylum-2/train/images"
        print("Start generating uuid csv for train images uuids")
        generate_uuid_csv(train_path, "src/data/csvfiles/train_uuids.csv")
        print("End generating uuid csv for train images uuids")
    except Exception as e:
        print(e)


def filter_fathomnet_csv():
    file_path = Path.cwd() / "src/data/csvfiles/train_uuids_info.csv"
    output_path = Path.cwd() / "src/data/csvfiles/train_uuids_info_clean.csv"
    to_remove = [
        "manipulator", "polypropylene line", "equipment",
        "drawer", "biology box", "suction sampler", "talus"
    ]

    df = pd.read_csv(file_path)
    clean_df = df[~df['concept'].str.lower().isin([x.lower() for x in to_remove])]
    clean_df.to_csv(output_path, index=False)

    print(f"Ending cleaning. deleted rows : {len(df) - len(clean_df)}")


# Utilisation
# filter_fathomnet_csv('ton_dataset.csv', 'dataset_clean.csv')

def generate_captions_with_llava():
    pass

def generate_csv_uuids_info_for_train_images_with_exlude_concepts():
    train_uuids_csv_path = Path.cwd() / "src/data/csvfiles/train_uuids.csv"
    output_csv_path = Path.cwd() / "src/data/csvfiles/train_uuids_info.csv"
    original_image_directory_path = Path.cwd() / "src/data/Fathomnet-by-Phylum-2/train/original_images"

    headers = [
        'uuid', 'depth', 'temperature', 'total_objects',
        'concept', 'concept_rank', 'is_certain',
        'phylum', 'class', 'order', 'family', 'genus',
        'x', 'y', 'width', 'height'
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    try:
        with open(train_uuids_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            uuids = [row[0].strip() for row in reader if row]
    except Exception as e:
        print(f"Erreur lecture CSV : {e}"); exit()

    print(f"({len(uuids)} images)...")

    for uuid in uuids:
        try:
            image_record = imagesFathomnet.find_by_uuid(uuid)

            # Download Original image if not exist
            img_path = original_image_directory_path / f"{uuid}_original.jpg"
            if not img_path.exists():
                resp = requests.get(str(image_record.url), stream=True, timeout=10)

                if resp.status_code == 200:
                    with Image.open(resp.raw) as img:
                        img.convert('RGB').save(img_path, "JPEG", quality=95)
                else:
                    continue

            depth = getattr(image_record, 'depthMeters', 'N/A')
            temp = getattr(image_record, 'temperatureCelsius', 'N/A')
            num_boxes = len(image_record.boundingBoxes)

            excluded_concepts = ["manipulator", "polypropylene line", "equipment", "drawer", "biology box",
                                 "suction sampler", "talus"]

            with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                for box in image_record.boundingBoxes:
                    concept = box.concept

                    # Filter for excluse
                    if concept.lower() in excluded_concepts:
                        continue

                    is_certain = 0 if (" cf. " in concept or " sp. " in concept) else 1
                    taxo = get_taxamony(concept)

                    writer.writerow([
                        uuid, depth, temp, num_boxes,
                        concept, taxo['rank'], is_certain,
                        taxo['phylum'], taxo['class'], taxo['order'], taxo['family'], taxo['genus'],
                        box.x, box.y, box.width, box.height
                    ])

            print(f"OK : {uuid} (Total objects in the image: {num_boxes})")

        except Exception as e:
            print(f"Saut : {uuid} | Erreur: {e}")

    print(f"**** END ****")


def generate_csv_uuids_info_for_train_images():
    train_uuids_csv_path = Path.cwd() / "src/data/csvfiles/train_uuids.csv"
    output_csv_path = Path.cwd() / "src/data/csvfiles/train_uuids_info.csv"
    original_image_directory_path = Path.cwd() / "src/data/Fathomnet-by-Phylum-2/train/original_images"

    headers = [
        'uuid', 'depth', 'temperature', 'total_objects',
        'concept', 'concept_rank', 'is_certain',
        'phylum', 'class', 'order', 'family', 'genus',
        'x', 'y', 'width', 'height'
    ]

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    try:
        with open(train_uuids_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            uuids = [row[0].strip() for row in reader if row]
    except Exception as e:
        print(f"Erreur lecture CSV : {e}"); exit()

    print(f"({len(uuids)} images)...")

    for uuid in uuids:
        try:
            image_record = imagesFathomnet.find_by_uuid(uuid)

            # Download Original image if not exist
            img_path = original_image_directory_path / f"{uuid}_original.jpg"
            if not img_path.exists():
                resp = requests.get(str(image_record.url), stream=True, timeout=10)

                if resp.status_code == 200:
                    with Image.open(resp.raw) as img:
                        img.convert('RGB').save(img_path, "JPEG", quality=95)
                else:
                    continue

            depth = getattr(image_record, 'depthMeters', 'N/A')
            temp = getattr(image_record, 'temperatureCelsius', 'N/A')
            num_boxes = len(image_record.boundingBoxes)

            for box in image_record.boundingBoxes:
                concept = box.concept
                is_certain = 0 if (" cf. " in concept or " sp. " in concept) else 1
                taxo = get_taxamony(concept)

                with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        uuid, depth, temp, num_boxes,
                        concept, taxo['rank'], is_certain,
                        taxo['phylum'], taxo['class'], taxo['order'], taxo['family'], taxo['genus'],
                        box.x, box.y, box.width, box.height
                    ])

            print(f"OK : {uuid} | {taxo['rank']} (Total: {num_boxes})")

        except Exception as e:
            print(f"Saut : {uuid} | Erreur: {e}")

    print(f"**** END ****")










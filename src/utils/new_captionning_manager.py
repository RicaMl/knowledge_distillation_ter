import base64
import os
from groq import Groq
import pandas as pd
from collections import Counter
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
clientOpenai  = OpenAI(api_key=api_key)
api_key = os.getenv("GROQ_API_KEY")
client= Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')



def caption_morphology_env(image_path, depth_m, main_concepts, counts_str):
    """
    main_concepts : list of concepts (scientific name return by api) (ex: ['Zoantharia', 'Galatheoidea'])
    """
    concepts_str = " and ".join(main_concepts[:2])
    depth_str = f"{depth_m}m" if depth_m else ""
    examples = """
    Example 1:
    Inhabitants: Bathyraja spinosissima
    Abundance: Bathyraja spinosissima x1
    Context: Depth 1832m.
    Output: Pale deep-sea skate Bathyraja spinosissima resting on volcanic rocky substrate at 1832m.
    
    Example 2:
    Inhabitants: Pannychia moseleyi ; Pterasteridae
    Abundance: Pannychia moseleyi x2 ; Pterasteridae x1
    Context: Depth 1440m.
    Output: Two translucent Pannychia moseleyi beside Pterasteridae cushion star on fine sediment with basalt fragments at 1440m.
    """

    prompt = f"""
    {examples}

    You are a marine biologist describing deep-sea imagery for machine learning.

    Focus on:
    - morphology (shape, texture, color)
    - spatial arrangement of organisms
    - substrate type (rock, sediment, basalt, etc.)
    - depth context when relevant

    Rules:
    - Naturally include scientific names if provided (do not list them separately)
    - Do NOT write metadata-style phrases (e.g., "Inhabitants are...")
    - Integrate abundance naturally only if it improves readability
    - One single sentence only
    - Maximum 30 words
    - Output only the caption (no explanation, no labels)

    Scene:
    Inhabitants: {concepts_str}
    Abundance: {counts_str}
    Context: Depth {depth_str}.
    """
    base64_image = encode_image(image_path)
    response = clientOpenai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Marine biologist. You describe visual traits concisely. Output only caption."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=50,
        temperature=0.2
    )
    return response.choices[0].message.content.strip().strip('"')

def genarate_captionning_and_csvs_train2():
    images_dir= Path("./src/data/Fathomnet-by-Phylum-2/images/train")
    csv_input_path = Path("src/data/csvfiles/train_uuids_info.csv")
    df = pd.read_csv(csv_input_path)
    cols = ['uuid', 'depth', 'temp', 'total_count', 'concept', 'rank', 'is_correct',
            'phylum', 'class', 'order', 'family', 'genus', 'x', 'y', 'w', 'h']
    df.columns = cols

    results_morpho = []

    # Grouped by uuid for process an image once
    grouped = df.groupby('uuid')

    for uuid, group in grouped:
        image_path = os.path.join(images_dir, f"{uuid}_original.jpg")

        if not os.path.exists(image_path):
            print(f"Skipping {uuid}: Image not found")
            continue

        # Metadata extraction
        depth = group['depth'].iloc[0]
        temp = group['temp'].iloc[0]

        # List of concepts who are in one image
        all_concepts = group['concept'].tolist()
        raw_concepts = group['concept'].tolist()
        bio_concepts = [c for c in raw_concepts if c.lower() not in ['n/a', 'unknown']]

        main_concepts = list(set(bio_concepts))
        counts_bio = dict(Counter(bio_concepts))
        counts_str = " ; ".join([f"{k} x{v}" for k, v in counts_bio.items()])
        # Count the number of concept for the ecology captioning
        concepts_counts = dict(Counter(all_concepts))

        print(f"Processing image: {uuid}...")

        try:
            cap_morpho = caption_morphology_env(image_path, depth, main_concepts, counts_str)
            print(cap_morpho)
            time.sleep(2)
            # Save result
            results_morpho.append({'uuid': uuid, 'caption': cap_morpho})

        except Exception as e:
            print(f"Error on {uuid}: {e}")

    # Save on separate csv
    pd.DataFrame(results_morpho).to_csv('captions_morphology_train.csv', index=False)
    print("Ending, captions_morphology_train.csv created.")

def genarate_captionning_and_csvs_test2():
    images_dir = Path("./src/data/Fathomnet-by-Phylum-2/test/images")
    csv_input_path = Path("src/data/csvfiles/test_uuids_info.csv")

    df = pd.read_csv(csv_input_path)
    cols = ['uuid', 'depth', 'temp', 'total_count', 'concept', 'rank', 'is_correct',
            'phylum', 'class', 'order', 'family', 'genus', 'x', 'y', 'w', 'h']
    df.columns = cols

    results_morpho = []

    grouped = df.groupby('uuid')

    print(f"Starting process for {len(grouped)} images...")

    for uuid, group in grouped:
        matching_files = list(images_dir.glob(f"{uuid}*.jpg"))

        if not matching_files:
            print(f"Skipping {uuid}: Image not found")
            continue

        image_path = matching_files[0]
        depth = group['depth'].iloc[0]

        raw_concepts = group['concept'].tolist()
        bio_concepts = [c for c in raw_concepts if c.lower() not in ['n/a', 'unknown']]

        main_concepts = list(set(bio_concepts))
        concepts_counts = dict(Counter(raw_concepts))
        counts_bio = dict(Counter(bio_concepts))
        counts_str = " ; ".join([f"{k} x{v}" for k, v in counts_bio.items()])

        print(f"Processing: {uuid}")

        try:
            cap_morpho = caption_morphology_env(image_path, depth, main_concepts, counts_str)
            time.sleep(2)
            print(cap_morpho)

            results_morpho.append({'uuid': uuid, 'caption': cap_morpho})

        except Exception as e:
            print(f"Error on {uuid}: {e}")

    if results_morpho:
        pd.DataFrame(results_morpho).to_csv('captions_morphology_test2.csv', index=False)
        print("\nProcess complete!")
        print("Created: captions_morphology_test2.csv ")
    else:
        print("No captions were generated.")


def retry_failed_captions():
    images_dir = Path("./src/data/Fathomnet-by-Phylum-2/images/train")
    csv_input_path = Path("src/data/csvfiles/train_uuids_info.csv")
    df = pd.read_csv(csv_input_path)

    # Liste des UUIDs extraits de tes erreurs (copie-colle bien tout ici)
    failed_uuids = [
        "2d3ef085-a4da-4b23-ad77-f136e6f95e72", "45f8e621-4357-4a62-b3de-c6bb997d8b88",
        "46563a77-107a-4795-8aa8-10584abbb2e4", "4666969d-a6a6-4e12-848b-79e0d39fec67",
        "4691de51-2cf2-4b2f-8d1f-0b694005565a", "47c9f848-c697-4d15-b899-4a64a987cf53",
        "55dd270f-6fc2-434d-9745-c8bad851eb67", "5626fa1a-5a2e-4394-89e6-30e74805d778",
        "566da252-c5b0-4857-900f-2626b9782cd8", "56d75631-464f-46b7-8d47-2269a557dbf3",
        "5e711c56-639a-4222-9cb1-2fa34907ba36", "656f1397-2962-48f3-b2f0-6d2c9930d788",
        "671fa775-5a7f-419a-868a-9b16fdd5d237", "67aad265-8a88-4275-8e8c-fac22a989d28",
        "68ce135f-af03-4081-bce4-11afc3764776", "74557982-8aaa-4ecb-a8f0-0348a26dc4bf",
        "7537b6ae-24f6-4440-9c6b-fcb95832fd0d", "78026949-b615-4ada-9036-111b4be70b46",
        "78ab5828-326d-4d43-9ce6-a62d44a9bc98", "7a6a340d-6d6b-4578-9b1b-62b7dc7d6385",
        "807ead0e-c26f-4fe9-b800-95fe3f62711f", "85f39cfe-0173-4645-8994-e1389c54c88b",
        "89f52ffb-4ff2-460d-9534-64df47fdfc65", "8c0419d9-b33e-4bce-8735-f64b422292c4",
        "8eb9fb01-42a2-4a48-a0d3-c9ca72fbbe3d", "9cf8ee76-148c-48bc-8ba8-7f3f632624e0",
        "a28a684e-a2da-4139-a5ec-d51a3dcab2ce", "a520109b-ae85-4020-8c94-9e93a6adb291",
        "a8b47b61-389d-480a-840e-e10b6737316b", "b4fe74b2-b7f1-437d-8db1-a14a15497a2d",
        "bb8f49a5-21ad-4354-93a4-806005e61586", "bbc74f9d-4bcb-4bd8-9e5f-7cfcc85fedbd",
        "bf4848b2-6f3c-4c68-9db4-aecebd640d97", "c55c4a6f-30ef-4bd8-abb0-6d038b83503e",
        "c8433d2b-a9ff-491e-abfe-50e26a89f547", "c96a4cd6-4b8d-4d6f-b84d-03b13b0ecc95",
        "cae70be6-7fd2-4703-940f-e1f5b4d43f74", "cbec2932-30e5-45eb-b383-ee2f52fe38d3",
        "ccf44649-8091-41e8-961b-4c02d052e889", "cee4205b-64a6-44a3-925d-3a168b2dcf85",
        "d2f6da09-6c7f-4fc9-8205-7d88d8dced01", "d5cbfffd-8a95-4761-a8a6-efdb74f5d2c6",
        "d7764b50-4715-470f-b43e-8d6810a3b57f", "d8c29f5b-898d-4660-90d9-b6544d5da283",
        "dad27d25-2a59-4b02-8b41-e074fd998206", "dc5a97a8-dd19-4331-a08f-7cc3623a743b",
        "dc7d41c6-0abb-485b-bd2a-70ab74fa3684", "dd3280cc-2814-473b-995f-81208b690843",
        "e289635c-44d9-4313-8309-5b1ce19f80ae", "f3b9348a-8e3f-4205-b358-07f3c588dc66",
        "f3dbe151-d540-4a9d-a7fc-b565116b337c", "f535b1e8-38d7-4879-a999-20729f01c6f4",
        "f5a31f29-cd3e-45e9-9816-6f694ed964ef", "f5b67889-4ca9-4f97-b9ec-a12eeb33955b",
        "f5f3ac9a-6f5e-4534-9d9f-3d3c064753e6", "f727209f-3b4c-49e9-ac26-f744a4cbe2d8",
        "f9608489-acc2-46a5-96bf-47640d68cbce", "fc4f0300-8602-4b52-bf31-e400ff4aecf7",
        "fd008e42-e8b8-4b55-9d23-4e23b9d9e2ca", "fd5f6f77-b8aa-4cd2-ab9c-164cc314ef60"
    ]

    results_morpho = []

    # Filtrer le DataFrame pour ne garder que les manquants
    df_retry = df[df['uuid'].isin(failed_uuids)]
    grouped = df_retry.groupby('uuid')

    print(f"Relance de {len(grouped)} images ayant échoué...")

    for uuid, group in grouped:
        image_path = os.path.join(images_dir, f"{uuid}_original.jpg")

        # ... (Garder ton extraction de métadonnées identique) ...
        depth = group['depth'].iloc[0]
        bio_concepts = [c for c in group['concept'].tolist() if str(c).lower() not in ['n/a', 'unknown']]
        main_concepts = list(set(bio_concepts))
        counts_bio = dict(Counter(bio_concepts))
        counts_str = " ; ".join([f"{k} x{v}" for k, v in counts_bio.items()])

        print(f"Retry image: {uuid}...")

        success = False
        wait_time = 5  # On commence à 5s pour laisser le TPM respirer

        while not success:
            try:
                cap_morpho = caption_morphology_env(image_path, depth, main_concepts, counts_str)
                print(f"Success: {cap_morpho}")
                results_morpho.append({'uuid': uuid, 'caption': cap_morpho})
                success = True
                time.sleep(1)  # Petit délai entre deux réussites
            except Exception as e:
                if "429" in str(e):
                    print(f"Limite atteinte pour {uuid}. Pause de {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time *= 1.5  # Augmentation progressive
                else:
                    print(f"Erreur fatale sur {uuid}: {e}")
                    break

    # Sauvegarde dans un fichier spécifique pour ne pas écraser ton travail précédent
        # Création du DataFrame avec les nouveaux résultats
        new_df = pd.DataFrame(results_morpho)

        # Chemin de ton fichier principal
        main_csv_path = 'captions_morphology_train.csv'

        # Vérification si le fichier existe pour décider d'écrire l'entête ou non
        file_exists = os.path.isfile(main_csv_path)

        # Ajout à la fin du fichier
        new_df.to_csv(main_csv_path,
                      mode='a',
                      index=False,
                      header=not file_exists,
                      encoding='utf-8')

        print(f"Terminé. {len(results_morpho)} nouvelles lignes ajoutées à {main_csv_path}.")

def caption_ecology(image_path, depth_m, concepts_counts):
    depth_str = f"{depth_m}m" if depth_m!= "N/A" else "deep sea"
    counts_str = ", ".join([f"{c} {sp}" for sp, c in concepts_counts.items()])
    prompt = f"""Ecology-focused caption. {counts_str} at depth {depth_str}. Describe any observable interaction (e.g., foraging, clustering, spatial arrangement) or community type. Be factual, avoid 'likely' or 'probably'. One short sentence (max 25 words)."""

    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Marine ecologist. Output only caption. No speculation."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=50,
        temperature=0.2
    )
    return response.choices[0].message.content.strip().strip('"')



def genarate_captionning_and_csvs_trainn():
    images_dir= Path("./src/data/Fathomnet-by-Phylum-2/train/original_images")
    csv_input_path = Path("src/data/csvfiles/train_uuids_info.csv")
    df = pd.read_csv(csv_input_path)
    cols = ['uuid', 'depth', 'temp', 'total_count', 'concept', 'rank', 'is_correct',
            'phylum', 'class', 'order', 'family', 'genus', 'x', 'y', 'w', 'h']
    df.columns = cols

    results_multi = []
    results_morpho = []
    results_ecology = []

    # Grouped by uuid for process an image once
    grouped = df.groupby('uuid')

    for uuid, group in grouped:
        image_path = os.path.join(images_dir, f"{uuid}_original.jpg")

        if not os.path.exists(image_path):
            print(f"Skipping {uuid}: Image not found")
            continue

        # Metadata extraction
        depth = group['depth'].iloc[0]
        temp = group['temp'].iloc[0]

        # List of concepts who are in one image
        all_concepts = group['concept'].tolist()
        raw_concepts = group['concept'].tolist()
        bio_concepts = [c for c in raw_concepts if c.lower() not in ['n/a', 'unknown']]

        main_concepts = list(set(bio_concepts))
        counts_bio = dict(Counter(bio_concepts))

        # Count the number of concept for the ecology captioning
        concepts_counts = dict(Counter(all_concepts))

        print(f"Processing image: {uuid}...")

        try:
            cap_morpho = caption_morphology_env(image_path, depth, main_concepts)
            time.sleep(2)
            cap_ecology = caption_ecology(image_path, depth, concepts_counts)
            time.sleep(2)

            # Save result
            results_morpho.append({'uuid': uuid, 'caption': cap_morpho})
            results_ecology.append({'uuid': uuid, 'caption': cap_ecology})

        except Exception as e:
            print(f"Error on {uuid}: {e}")

    # Save on separate csv
    pd.DataFrame(results_morpho).to_csv('captions_morphology_train.csv', index=False)
    pd.DataFrame(results_ecology).to_csv('captions_ecology_train.csv', index=False)
    print("Ending, captions_morphology_train.csv and captions_ecology_train.csv created.")

def genarate_captionning_and_csvs_test():
    images_dir = Path("./src/data/Fathomnet-by-Phylum-2/test/images")
    csv_input_path = Path("src/data/csvfiles/test_uuids_info.csv")

    df = pd.read_csv(csv_input_path)
    cols = ['uuid', 'depth', 'temp', 'total_count', 'concept', 'rank', 'is_correct',
            'phylum', 'class', 'order', 'family', 'genus', 'x', 'y', 'w', 'h']
    df.columns = cols

    results_morpho = []
    results_ecology = []

    grouped = df.groupby('uuid')

    print(f"Starting process for {len(grouped)} images...")

    for uuid, group in grouped:
        matching_files = list(images_dir.glob(f"{uuid}*.jpg"))

        if not matching_files:
            print(f"Skipping {uuid}: Image not found")
            continue

        image_path = matching_files[0]
        depth = group['depth'].iloc[0]

        raw_concepts = group['concept'].tolist()
        bio_concepts = [c for c in raw_concepts if c.lower() not in ['n/a', 'unknown']]

        main_concepts = list(set(bio_concepts))
        concepts_counts = dict(Counter(raw_concepts))

        print(f"Processing: {uuid}")

        try:
            cap_morpho = caption_morphology_env(image_path, depth, main_concepts)
            time.sleep(2)
            cap_ecology = caption_ecology(image_path, depth, concepts_counts)
            time.sleep(2)

            results_morpho.append({'uuid': uuid, 'caption': cap_morpho})
            results_ecology.append({'uuid': uuid, 'caption': cap_ecology})

        except Exception as e:
            print(f"Error on {uuid}: {e}")

    if results_morpho:
        pd.DataFrame(results_morpho).to_csv('captions_morphology_val.csv', index=False)
        pd.DataFrame(results_ecology).to_csv('captions_ecology_val.csv', index=False)
        print("\nProcess complete!")
        print("Created: captions_morphology_val.csv and captions_ecology_val.csv")
    else:
        print("No captions were generated.")


import pandas as pd
from pathlib import Path


def prepare_test_csv_for_clip(csv_path, images_dir, output_path):
    """
    Replace the uuid column by the image path.
    """
    df = pd.read_csv(csv_path)
    img_dir = Path(images_dir)

    print(f"Process {len(df)} rows...")

    #Create a dictionary like {uuid: nom_complet}
    #We scan the folder once
    file_mapping = {f.name[:36]: f.name for f in img_dir.glob("*.jpg")}

    def get_full_filename(uuid):
        # get full nma ebecause the name of the image is not juste uuid.jpg
        full_name = file_mapping.get(uuid)
        if full_name:
            return str(img_dir / full_name)
        else:
            print(f"Not found image for l'UUID {uuid}")
            return None

    # update the column
    df['uuid'] = df['uuid'].apply(get_full_filename)

    # delete row missed
    df = df.dropna(subset=['uuid'])

    df = df.rename(columns={'uuid': 'filepath'})
    cols = ['caption', 'filepath']
    df = df[cols]

    # Save
    df.to_csv(output_path, index=False)
    print(f"End : {output_path}")
    print(f"Structure finale : \n{df.head(2)}")


def prepare_train_csv_augmented(captions_csv, train_images_dir, output_path):
    """

    """

    ref_df = pd.read_csv(captions_csv)

    caption_lookup = dict(zip(ref_df['uuid'], ref_df['caption']))

    train_dir = Path(train_images_dir)
    data_rows = []

    print(f"Scanning directory: {train_images_dir}...")

    all_files = list(train_dir.glob("*.jpg"))
    print(f"Found {len(all_files)} images")
    for file_path in all_files:
        filename = file_path.name
        uuid_prefix = filename[:36]

        if uuid_prefix in caption_lookup:
            data_rows.append({
                'filepath': file_path,
                'caption': caption_lookup[uuid_prefix]
            })
        else:
            print(f"Not found image for l'UUID {uuid_prefix}")
            pass


    train_final_df = pd.DataFrame(data_rows)

    train_final_df = train_final_df[['caption', 'filepath']]

    train_final_df.to_csv(output_path, index=False)

    print(f"End : {output_path}")
    print(f"Total d'images trouvées (incluant augmentations) : {len(train_final_df)}")


def get_refined_caption(vlm_raw, species_list):
    """
    Le LLM fusionne tout en une seule phrase fluide et naturelle.
    La taxonomie est utilisée pour corriger, pas pour être listée.
    """
    # On prépare un contexte riche mais discret pour le LLM
    taxa_context = ""
    for s in species_list:
        taxa_context += f"- {s['count']} {s['name']} (Class: {s['class']}, Family: {s['family']})\n"

        prompt = f"""[SYSTEM] You are a marine biology expert.
    [CONTEXT]
    Raw VLM Observation: "{vlm_raw}"
    Scientific Data:
    {taxa_context}
    
    [TASK]
    Create a single, fluid, professional scientific sentence (MAX 30 WORDS).
    1. Integrate the count naturally (e.g., 'Three sea stars...').
    2. Use the taxonomy Class/Family ONLY to correct morphology errors (e.g., ensure an Asteroidea is described as star-shaped).
    3. DO NOT write "Taxonomic hierarchy" or "Class:". 
    4. Start directly with the description of the scene explaind by the vlm observation.
    """

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0.1,
            max_tokens=60
        )
        return completion.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        return vlm_raw # Sécurité

def generate_llm():
    # 1. Chargement des données
    # uuid_info : contient la biologie
    info_cols = ['uuid', 'depth', 'temp', 'count', 'concept', 'rank', 'occ',
                 'phylum', 'class', 'order', 'family', 'genus', 'x', 'y', 'w', 'h']
    df_info = pd.read_csv('src/data/csvfiles/train_uuids_info.csv', names=info_cols, header=None)

    # caption.csv : contient la phrase brute du VLM
    df_vlm = pd.read_csv('captions_morphology_train.csv')

    # 2. Groupement par UUID pour gérer le multi-espèces
    grouped_info = df_info.groupby('uuid')

    final_results = []

    print(f"Début de la reformulation pour {len(grouped_info)} images...")

    for uuid, group in grouped_info:
        # Trouver la phrase VLM correspondante
        vlm_row = df_vlm[df_vlm['uuid'] == uuid]
        if vlm_row.empty:
            continue

        raw_phrase = vlm_row.iloc[0]['caption']

        # Préparer la liste des espèces pour cet UUID
        species_info = []
        for _, row in group.iterrows():
            species_info.append({
                'name': row['concept'],
                'count': row['count'],
                'class': row['class'],
                'family': row['family']
            })

        # Appel au LLM pour reformuler la phrase du VLM avec la taxonomie
        refined = get_refined_caption(raw_phrase, species_info)

        # Optionnel : Ajouter la profondeur à la fin pour CLIP
        #depth = group['depth'].iloc[0]
        #final_caption = f"{refined} Recorded at {round(float(depth))}m."

        final_results.append({'uuid': uuid, 'caption': refined })

        # Feedback console & limite de rate-limit (Groq est généreux mais prudence)
        if len(final_results) % 10 == 0:
            print(f"Traités : {len(final_results)} / {len(grouped_info)}")
        time.sleep(1)

        # 3. Sauvegarde
    pd.DataFrame(final_results).to_csv('src/data/csvfiles/final_expert_captions.csv', index=False)
    print("Terminé ! Le fichier expert est prêt.")


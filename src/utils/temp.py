import os
import pandas as pd
import ollama
from PIL import Image
from io import BytesIO
from pathlib import Path

def main():
    # --- CONFIGURATION ---
    CSV_INPUT = Path('src/data/csvfiles/train_uuids_info.csv')
    IMAGE_DIR = Path("./src/data/Fathomnet-by-Phylum-2/images/train")
    CSV_OUTPUT = 'dataset_final_VMI_style_imagenet.csv'
    MODEL_NAME = 'llava:7b-v1.6'

    RESOLUTION_THRESHOLD = 6000

    if not CSV_INPUT.exists():
        print(f"❌ Erreur : {CSV_INPUT} introuvable.")
        return

    df = pd.read_csv(CSV_INPUT)
    results = []

    print(f"🚀 Génération de légendes style ImageNet (Court & Précis)...")

    limit = 20

    for i, row in df.iterrows():
        if i >= limit: break

        uuid = row['uuid']
        concept = row['concept']
        img_path = IMAGE_DIR / f"{uuid}_original.jpg"

        if not img_path.exists(): continue

        try:
            with Image.open(img_path) as img:
                crop = img.crop((row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height'])).convert("RGB")
                buf = BytesIO()
                crop.save(buf, format='PNG')
                img_bytes = buf.getvalue()

            surface = row['width'] * row['height']

            if surface < RESOLUTION_THRESHOLD:
                # Style minimaliste pour les petites images
                vlm_description = f"small deep-sea organism, part of {row['class']} class"
            else:
                # PROMPT STYLE IMAGENET
                prompt_expert = (
                    f"Task: You are a marine biologist. Provide a one-sentence dictionary definition for '{concept}' based only on visible features.Focus on morphology. Do not compare to non-marine objects. "
                    f"Example style: 'large aggressive shark with striped body' or 'small golden freshwater fish'. "
                    f"RULES: No full sentences, no 'I can see', no headers, no 'Geometry'. Just the visual traits. "
                    f"Maximum 20 words."
                )

                response = ollama.generate(
                    model=MODEL_NAME,
                    prompt=prompt_expert,
                    images=[img_bytes],
                    options={"temperature": 0}
                )

                # Nettoyage agressif du texte
                vlm_description = response['response'].strip().lower()
                # Supprime les préfixes courants si l'IA bavarde quand même
                for prefix in ["the image shows", "this is", "a specimen of", "description:", "geometry:"]:
                    vlm_description = vlm_description.replace(prefix, "").strip()
                vlm_description = vlm_description.split('.')[0]  # On garde juste la première phrase

            # --- CONSTRUCTION DU CSV FINAL ---
            # Format: Concept | Nom Scientifique | Description courte
            scientific_name = row.get('acceptedNameUsage', concept)
            final_caption = f"{concept}, {scientific_name} | {vlm_description}"

            results.append({
                'uuid': uuid,
                'concept': concept,
                'caption': final_caption,
                'depth': row['depth']
            })

            print(f"✅ [{i + 1}/{limit}] {concept}: {vlm_description}")

        except Exception as e:
            print(f"❌ Erreur ligne {i} : {e}")

    df_final = pd.DataFrame(results)
    df_final.to_csv(CSV_OUTPUT, index=False)
    print(f"\n✨ Dataset prêt au format ImageNet : {CSV_OUTPUT}")


if __name__ == '__main__':
    main()
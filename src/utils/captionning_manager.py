import pandas as pd
from groq import Groq
import time
import os
from dotenv import load_dotenv
from pathlib import Path
import ollama
from PIL import Image
from io import BytesIO

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"

def get_llm_description(row):
    """
    Generates a descriptive sentence by using the available taxamony hierarchy to guid the LLM.
    """
    species_name = row.get('concept', 'Unknown')
    species_name  = species_name .replace(" cf. ", " ").replace(" sp. ", " ").strip()
    #Get all available taxa who are not empty
    taxa_info = []
    for rank in ["phylum", "class", "order", "family", "genus"]:
        val = row.get(rank)
        if val and str(val).lower() not in ["n/a", "nan", "none", ""]:
            taxa_info.append(f"{rank}: {val}")

    taxa_str = ", ".join(taxa_info)
    # Prompt super guided
    prompt = (
        f"Context: Marine organism '{species_name}'. Taxonomy: {taxa_str}. "
        f"Task: Describe the morphology of '{species_name}' in one short, fluid sentence."
        "Focus on shape, color, and texture. Be biologically accurate based on its taxonomy."
        "Do not use lists. Start directly with adjectives."
    )

    try:
        #Ask groq
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=60
        )
        return completion.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Error for{species_name}: {e}")
        return ""


def build_hierarchical_caption(row, vlm_description):
    """
    Assembly for the caption
    """
    concept = row.get('concept', 'Unknown')
    rank = str(row.get('concept_rank', 'unknown')).lower().strip()

    # TAXA
    taxa_labels = {
        "phylum": row.get('phylum'),
        "class": row.get('class'),
        "order": row.get('order'),
        "family": row.get('family'),
        "genus": row.get('genus')
    }

    # order from general to specific
    order_of_ranks = ["phylum", "class", "order", "family", "genus"]
    max_idx = 5 if rank == "species" else (order_of_ranks.index(rank) if rank in order_of_ranks else 0)
    selected_taxa = [f"{order_of_ranks[i]} {taxa_labels[order_of_ranks[i]]}"
                     for i in range(max_idx)
                     if str(taxa_labels[order_of_ranks[i]]) not in ["N/A", "nan", "None", ""]]

    # ENVIRONNEMENT
    env_info = []
    depth = row.get('depth')
    temp = row.get('temperature')
    if depth not in ["N/A", "nan", None, ""]:
        env_info.append(f"depth {round(float(depth), 1)}m")
    if temp not in ["N/A", "nan", None, ""]:
        env_info.append(f"temp {round(float(temp), 2)}°C")

    # Assembly style like imageNet
    # Header: Nom Commun (si dispo), Nom Scientifique
    header = f"{concept},"

    # Body: Description via llm + Taxo + Env
    body_parts = []
    if vlm_description: body_parts.append(vlm_description)
    if selected_taxa: body_parts.append(f"taxonomic hierarchy: {', '.join(selected_taxa)}")
    if env_info: body_parts.append(f"environment: {', '.join(env_info)}")

    return f"{header}\t{'. '.join(body_parts)}."


def generate_captions_via_llm():
    df = pd.read_csv('src/data/csvfiles/train_uuids_info.csv')
    # get unique species for them to have the same description
    unique_species = df['concept'].unique()
    print(len(unique_species))
    descriptions_map = {}

    print(f"Starting to generate descriptions for {len(unique_species)} with llm...")
    for species in unique_species:
        # We just retrieve the first line of the first row who have this specie as concept
        row_data = df[df['concept'] == species].iloc[0]

        # get description for this specie
        descriptions_map[species] = get_llm_description(row_data)

        print(f"{species} traited.")
        time.sleep(2) #for sending query with small awaiting beetween them

    # Apply the final caption to all ros swho have these species as concept
    results = []
    for _, row in df[df['concept'].isin(unique_species)].iterrows():
        vlm_desc = descriptions_map.get(row['concept'], "")
        final_caption = build_hierarchical_caption(row, vlm_desc)
        results.append({'uuid': row['uuid'], 'caption': final_caption})

    # Save
    pd.DataFrame(results).to_csv('src/data/csvfiles/final_train_captions.csv', index=False)
    print("End ! See 'final_train_captions.csv in csvfiles'")


def generate_captions_via_vlm():
    CSV_INPUT = Path('src/data/csvfiles/train_uuids_info.csv')
    IMAGE_DIR = Path("./src/data/Fathomnet-by-Phylum-2/images/train")
    CSV_OUTPUT = 'src/data/csvfiles/final_train_captions_vlm.csv'
    MODEL_NAME = 'moondream'

    RESOLUTION_THRESHOLD = 6000

    if not CSV_INPUT.exists():
        print(f"Error: {CSV_INPUT} not found.")
        return

    df = pd.read_csv(CSV_INPUT)
    results = []

    print(f"Sarting the captionning generation with vlm")
    for i, row in df.iterrows():
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
                # for small images
                vlm_description = f""
            else:
                taxa_info = []
                for rank in ["phylum", "class", "order", "family", "genus"]:
                    val = row.get(rank)
                    if val and str(val).lower() not in ["n/a", "nan", "none", ""]:
                        taxa_info.append(f"{rank}: {val}")

                taxa_str = ", ".join(taxa_info)
                prompt_expert = (
                     f"Context: Marine organism '{concept}'. Taxonomy: {taxa_str}. "
                    f"Task: Describe the morphology of '{concept}' in one short, fluid sentence."
                    "Focus on shape, color, and texture. Be biologically accurate based on its taxonomy."
                    "Do not use lists. Start directly with adjectives."
                )
                response = ollama.generate(
                    model=MODEL_NAME,
                    prompt=prompt_expert,
                    images=[img_bytes],
                    options={"temperature": 0}
                )

                vlm_description = response['response'].strip().lower()
                for prefix in ["the image shows", "this is", "a specimen of", "description:", "geometry:"]:
                    vlm_description = vlm_description.replace(prefix, "").strip()
                vlm_description = vlm_description.split('.')[0]  # On garde juste la première phrase

            #csv construction
            final_caption = build_hierarchical_caption(row, vlm_description)
            results.append({'uuid': row['uuid'], 'caption': final_caption})


            print(f"{concept} traited.")

        except Exception as e:
            print(f"Error in line {i} : {e}")

    df_final = pd.DataFrame(results)
    df_final.to_csv(CSV_OUTPUT, index=False)
    print(f"\nEnd ! See {CSV_OUTPUT}")
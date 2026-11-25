
import gc
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part_d_rag"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
DATA_ROOT = ROOT / "starter" / "data" / "uhat" / "data"

active_model = None

mae = None
classifier = None
vae = None
ddpm = None
diffusion = None
class_names = None
retriever = None
generator = None


def release_models():
    global active_model, mae, classifier, vae
    global ddpm, diffusion, retriever, generator

    mae = None
    classifier = None
    vae = None
    ddpm = None
    diffusion = None
    retriever = None
    generator = None
    active_model = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_mae():
    global active_model, mae, classifier

    if active_model == "mae":
        return

    release_models()

    from part_a_mae.mae import MaskedAutoencoder
    import torch.nn as nn

    from starter.eurosat import CLASS_NAMES

    mae = MaskedAutoencoder().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINTS / "best_mae.pth",
        map_location=DEVICE,
        weights_only=False
    )

    mae.load_state_dict(checkpoint["model_state_dict"])
    mae.eval()

    classifier = nn.Linear(192, 10).to(DEVICE)

    classifier.load_state_dict(
        torch.load(
            CHECKPOINTS / "mae_classifier.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    classifier.eval()

    active_model = "mae"


def load_vae():
    global active_model, vae

    if active_model == "vae":
        return

    release_models()

    from part_b_vae.vae import VAE

    vae = VAE(latent_dim=16).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINTS / "vae_best.pth",
        map_location=DEVICE,
        weights_only=False
    )

    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    active_model = "vae"


def load_ddpm():
    global active_model, ddpm, diffusion, class_names

    if active_model == "ddpm":
        return

    release_models()

    from part_c_diffusion.model import DiffusionUNet
    from part_c_diffusion.diffusion import Diffusion

    checkpoint = torch.load(
        CHECKPOINTS / "ddpm_best.pth",
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    ddpm = DiffusionUNet().to(DEVICE)
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    ddpm.eval()

    diffusion = Diffusion(
        timesteps=1000,
        device=str(DEVICE)
    )

    active_model = "ddpm"


def load_rag():
    global active_model, retriever, generator

    if active_model == "rag":
        return

    release_models()

    from search import BioASQRetriever
    from generator import BioASQGenerator

    retriever = BioASQRetriever()
    generator = BioASQGenerator()

    active_model = "rag"


def patchify(images, patch_size=8):
    n, c, h, w = images.shape

    x = images.reshape(
        n, c,
        h // patch_size, patch_size,
        w // patch_size, patch_size
    )

    x = x.permute(0, 2, 4, 3, 5, 1)

    return x.reshape(
        n,
        (h // patch_size) * (w // patch_size),
        patch_size * patch_size * c
    )


def unpatchify(patches, patch_size=8, channels=3):
    n, length, _ = patches.shape
    h = w = int(length ** 0.5)

    x = patches.reshape(
        n, h, w,
        patch_size, patch_size, channels
    )

    x = x.permute(0, 5, 1, 3, 2, 4)

    return x.reshape(
        n, channels,
        h * patch_size,
        w * patch_size
    )


def tensor_to_rgb(tensor):
    array = tensor.detach().cpu().clamp(0, 1)
    array = array.permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)

    return Image.fromarray(array)


def tensor_to_gray(tensor):
    array = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    array = (array * 255).astype(np.uint8)

    return Image.fromarray(array, mode="L")


def mae_predict(image):
    if image is None:
        raise gr.Error("Please upload a satellite image.")

    load_mae()

    image = image.convert("RGB").resize((64, 64))

    array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        patches = patchify(tensor)

        predictions, mask = mae(patches)

        masked = patches.clone()
        masked[mask.bool()] = 0.5

        reconstructed = patches.clone()
        reconstructed[mask.bool()] = predictions[mask.bool()]

        masked_image = unpatchify(masked)
        reconstructed_image = unpatchify(reconstructed)

        features = mae.encode_all(patches)
        logits = classifier(features)

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    from starter.eurosat import CLASS_NAMES

    label = CLASS_NAMES[predicted.item()]
    confidence_value = confidence.item() * 100

    result = (
        f"**Predicted class:** {label}\n\n"
        f"**Confidence:** {confidence_value:.2f}%"
    )

    return (
        tensor_to_rgb(tensor[0]),
        tensor_to_rgb(masked_image[0]),
        tensor_to_rgb(reconstructed_image[0]),
        result
    )


def preprocess_character(image):
    image = image.convert("L").resize((28, 28))

    array = np.asarray(image, dtype=np.float32) / 255.0

    if array.mean() > 0.5:
        array = 1.0 - array

    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def vae_reconstruct(image):
    if image is None:
        raise gr.Error("Please upload a character image.")

    load_vae()

    tensor = preprocess_character(image).to(DEVICE)

    with torch.no_grad():
        reconstruction, _, _ = vae(tensor)
        reconstruction = torch.sigmoid(reconstruction)

    return (
        tensor_to_gray(tensor[0]),
        tensor_to_gray(reconstruction[0])
    )


def vae_generate(number):
    load_vae()

    with torch.no_grad():
        samples = vae.sample(int(number), DEVICE)

    return [
        tensor_to_gray(sample)
        for sample in samples
    ]


def ddpm_generate(selected_class, number):
    if not selected_class:
        raise gr.Error("Please select a character class.")

    load_ddpm()

    class_id = class_names.index(selected_class)

    labels = torch.full(
        (int(number),),
        class_id,
        device=DEVICE,
        dtype=torch.long
    )

    with torch.no_grad():
        samples = diffusion.sample(
            ddpm,
            labels,
            image_size=28
        )

    samples = ((samples + 1) / 2).clamp(0, 1)

    return [
        tensor_to_gray(sample)
        for sample in samples
    ]


def rag_answer(question):
    if not question or not question.strip():
        raise gr.Error("Please enter a biomedical question.")

    load_rag()

    retrieved = retriever.search(
        question.strip(),
        top_k=5
    )

    answer = generator.generate(
        question.strip(),
        retrieved
    )

    references = ""

    for i, passage in enumerate(retrieved, 1):
        references += (
            f"\n### [{i}] Abstract ID: {passage['abstract_id']}\n"
            f"**Similarity:** {passage['score']:.4f}\n\n"
            f"{passage['text']}\n\n"
            "---\n"
        )

    final_answer = (
        answer.strip()
        + "\n\n---\n"
        + "**For educational use only. Not medical advice.**"
    )

    return final_answer, references


with gr.Blocks(
    title="Generative AI - Four Models, One App",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown(
        """
        # Generative AI: Four Models, One App

        Explore four generative AI approaches:
        **Masked Autoencoders, Variational Autoencoders,
        Diffusion Models, and Retrieval-Augmented Generation.**
        """
    )

    with gr.Tab("MAE - Satellite Image"):
        gr.Markdown(
            "Upload a satellite image to see its masked reconstruction "
            "and classification result."
        )

        with gr.Row():
            mae_input = gr.Image(
                label="Upload Satellite Image",
                type="pil"
            )

            mae_original = gr.Image(label="Original Image")

        mae_button = gr.Button(
            "Reconstruct and Classify",
            variant="primary"
        )

        with gr.Row():
            mae_masked = gr.Image(label="75% Masked Image")
            mae_reconstructed = gr.Image(label="Reconstructed Image")

        mae_result = gr.Markdown()

        mae_button.click(
            mae_predict,
            inputs=mae_input,
            outputs=[
                mae_original,
                mae_masked,
                mae_reconstructed,
                mae_result
            ]
        )

    with gr.Tab("VAE - Character Generation"):
        gr.Markdown(
            "Reconstruct a handwritten character or generate "
            "new characters from random latent vectors."
        )

        with gr.Row():
            vae_input = gr.Image(
                label="Upload Character",
                type="pil"
            )

            vae_original = gr.Image(label="Original Character")
            vae_reconstructed = gr.Image(label="Reconstructed Character")

        vae_reconstruct_button = gr.Button(
            "Reconstruct Character",
            variant="primary"
        )

        vae_reconstruct_button.click(
            vae_reconstruct,
            inputs=vae_input,
            outputs=[vae_original, vae_reconstructed]
        )

        gr.Markdown("### Generate New Characters")

        vae_number = gr.Slider(
            minimum=1,
            maximum=16,
            value=8,
            step=1,
            label="Number of Characters"
        )

        vae_generate_button = gr.Button("Generate Characters")

        vae_gallery = gr.Gallery(
            label="VAE Generated Characters",
            columns=8,
            height="auto"
        )

        vae_generate_button.click(
            vae_generate,
            inputs=vae_number,
            outputs=vae_gallery
        )

    with gr.Tab("DDPM - Conditional Generation"):
        gr.Markdown(
            "Select one of the 40 UHaT Urdu character classes "
            "and generate new handwritten samples."
        )

        ddpm_class = gr.Dropdown(
            choices=sorted(
                [
                    "af", "khaa", "laam"
                ]
            ),
            label="Character Class",
            allow_custom_value=True
        )

        ddpm_number = gr.Slider(
            minimum=1,
            maximum=8,
            value=4,
            step=1,
            label="Number of Samples"
        )

        ddpm_button = gr.Button(
            "Generate with DDPM",
            variant="primary"
        )

        ddpm_gallery = gr.Gallery(
            label="Generated Characters",
            columns=4,
            height="auto"
        )

        def get_ddpm_classes():
            load_ddpm()
            return gr.Dropdown(
                choices=class_names,
                value=class_names[0]
            )

        ddpm_button.click(
            ddpm_generate,
            inputs=[ddpm_class, ddpm_number],
            outputs=ddpm_gallery
        )

        app.load(
            get_ddpm_classes,
            outputs=ddpm_class
        )

    with gr.Tab("RAG - Biomedical Question Answering"):
        gr.Markdown(
            """
            Ask biomedical questions and receive answers grounded
            in retrieved BioASQ abstracts.

            **Note:** Answers are for educational use only.
            """
        )

        rag_question = gr.Textbox(
            label="Biomedical Question",
            placeholder="e.g. Is capmatinib effective for glioblastoma?",
            lines=3
        )

        rag_button = gr.Button(
            "Ask Question",
            variant="primary"
        )

        rag_answer_output = gr.Markdown(
            label="Generated Answer"
        )

        rag_references = gr.Markdown(
            label="Retrieved Passages"
        )

        rag_button.click(
            rag_answer,
            inputs=rag_question,
            outputs=[rag_answer_output, rag_references]
        )

    gr.Markdown(
        """
        ---
        **Project:** Generative AI - Assignment 04

        **Models:** MAE | VAE | DDPM | RAG
        """
    )


if __name__ == "__main__":
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )

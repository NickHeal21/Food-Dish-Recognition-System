import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B2_Weights
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Food Recognition System",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Food Dish Recognition System")

st.write(
    "Upload a food image, and the AI model will predict "
    "the food category."
)

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "best_efficientnet_b2_food101.pth"


@st.cache_resource
def load_model():
    weights = EfficientNet_B2_Weights.DEFAULT

    model = models.efficientnet_b2(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        101
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    ##model.load_state_dict(checkpoint) previous one
    model.to(DEVICE)
    model.eval()

    return model, weights.transforms()


model, preprocess = load_model()

# --------------------------------------------------
# Food-101 class names
# --------------------------------------------------

class_names = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles"
]

# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a food photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded food image",
        use_container_width=True
    )

    if st.button("Recognize Food"):

        with st.spinner("Analyzing image..."):

            input_tensor = preprocess(image).unsqueeze(0)
            input_tensor = input_tensor.to(DEVICE)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )[0]

            top5_probabilities, top5_indices = (
                torch.topk(probabilities, 5)
            )

        predicted_class = class_names[
            top5_indices[0].item()
        ]

        predicted_confidence = (
            top5_probabilities[0].item() * 100
        )

        st.success(
            f"Predicted food: {predicted_class.replace('_', ' ').title()}"
        )

        st.metric(
            "Top Prediction Confidence",
            f"{predicted_confidence:.2f}%"
        )

        st.subheader("Top 5 Predictions")

        for probability, index in zip(
            top5_probabilities,
            top5_indices
        ):
            food_name = class_names[index.item()]
            confidence = probability.item() * 100

            st.write(
                f"**{food_name.replace('_', ' ').title()}** "
                f"— {confidence:.2f}%"
            )

            st.progress(float(probability.item()))

st.divider()

st.caption(
    "Model: EfficientNet-B2 | Dataset: Food-101 | "
    "Number of classes: 101"
)
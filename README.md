# Food Dish Recognition System

A deep-learning-based food image classification system that recognizes food dishes across **101 categories** using transfer learning on the **Food-101 dataset**.

The project evaluates ImageNet-pretrained convolutional neural network architectures and provides a Streamlit application for predicting food categories from uploaded images.

## Project Highlights

- Food image classification across 101 Food-101 categories.
- Transfer learning with pretrained **EfficientNet-B0**, **ResNet50**, and **EfficientNet-B2**.
- Implemented using **Python, PyTorch, and Torchvision**.
- Evaluation using Top-1 accuracy, Top-5 accuracy, macro precision, macro recall, and macro F1-score.
- Streamlit interface for uploading an image and viewing predicted food categories with confidence scores.

## Dataset

This project uses the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

| Dataset detail | Value |
|---|---:|
| Number of food categories | 101 |
| Total images | 101,000 |
| Official training images | 75,750 |
| Official test images | 25,250 |
| Images per category | 1,000 |

For the full-data experiments, the official training split was divided into:

- **Training:** 68,175 images
- **Validation:** 7,575 images

The official test split contains 25,250 images and was used for final evaluation.

## Models and Experiments

Five experiment configurations were recorded:

| Experiment | Architecture | Training images | Best validation accuracy | Test Top-1 | Test Top-5 |
|---|---|---:|---:|---:|---:|
| 1 | EfficientNet-B0 | 5,050 | Not recorded | 45.63% | 72.31% |
| 2 | EfficientNet-B0 | 15,150 | 48.91% | 52.57% | 78.03% |
| 3 | ResNet50 | 8,000 | 61.90% | Not recorded | Not recorded |
| 4 | ResNet50 | 68,175 | 66.30% | Not recorded | Not recorded |
| 5 / Run 1 | EfficientNet-B2 | 68,175 | 69.58% | 74.70% | 93.35% |
| 5 / Run 2 | EfficientNet-B2 | 68,175 | See training log | **94.76%** | **98.65%** |

Run 2 was based on the existing EfficientNet-B2 checkpoint and fine-tuned further. The Run 2 evaluation reported the following results:

| Metric | Run 2 result |
|---|---:|
| Top-1 test accuracy | **94.76%** |
| Top-5 test accuracy | **98.65%** |
| Macro precision | 85.34% |
| Macro recall | 84.70% |
| Macro F1-score | 84.42% |
| Test images | 25,250 |
| Evaluation time | 4.41 minutes |

**Metric verification note:** The reported Top-1 accuracy and macro recall should be checked against the saved predictions and metric-calculation code. On a balanced test set, standard macro recall across all classes should equal overall Top-1 accuracy. The values above are reproduced as supplied in the Run 2 evaluation output and are not silently adjusted here.

## Technology Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Computer vision:** Torchvision
- **Dataset:** Food-101
- **Model architectures:** EfficientNet-B0, ResNet50, EfficientNet-B2
- **Demo application:** Streamlit
- **Training environment:** Google Colab with GPU acceleration

## Application

The Streamlit application allows a user to:

1. Upload a food image.
2. Run the trained model on the image.
3. View the predicted food category or ranked candidate categories.
4. View the model's confidence scores for its predictions.

The model is designed for the 101 categories included in Food-101. It may not correctly classify food outside those categories. Confidence scores should be treated as model scores, not guaranteed probabilities of correctness.

## Setup

The exact dependency versions and entry-point filenames depend on the project files. The following is a general setup outline.

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages used by the project and add their tested versions to a requirements file.

### 4. Prepare the dataset and model checkpoint

- Download Food-101 from its official source.
- Extract the dataset and configure the dataset path used by the training code.
- Place the trained model checkpoint at the path expected by the application.
- Keep large dataset files and model checkpoints out of Git unless the repository specifically requires them.

### 5. Run the Streamlit application

If the application entry point is `app.py`, run:

```bash
streamlit run app.py
```

Replace `app.py` with the actual application filename if it differs.

## Repository Structure

A suggested structure for organizing the project is:

```text
Food-Dish-Recognition/
├── app.py
├── models/
│   └── best_model.pth
├── notebooks/
│   └── food101_training.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

This is an organizational suggestion; the actual repository may use a different structure.

## Evaluation Metrics

- **Top-1 accuracy:** Percentage of test images for which the highest-ranked predicted class is correct.
- **Top-5 accuracy:** Percentage of test images for which the correct class appears among the five highest-ranked predictions.
- **Macro precision:** Precision calculated separately for each class and averaged equally across all classes.
- **Macro recall:** Recall calculated separately for each class and averaged equally across all classes.
- **Macro F1-score:** F1-score calculated for each class and averaged equally across all classes.

## Limitations

- The classifier is limited to the 101 categories in Food-101.
- Visually similar dishes may be confused.
- Performance on images from real-world environments may differ from performance on the Food-101 test set.
- The application is a demonstration and has not been established as a production-ready dietary or nutritional assessment system.
- Food recognition alone does not provide a reliable estimate of portion size, calories, or nutritional content.

## Future Improvements

- Verify and report all metrics from a single saved set of test predictions.
- Evaluate performance per food category and inspect the confusion matrix.
- Test on additional real-world food images.
- Improve generalization through controlled fine-tuning and data augmentation experiments.
- Measure application inference latency and usability.

## Acknowledgements

The project uses the Food-101 benchmark dataset introduced by Bossard, Guillaumin, and Van Gool.

**Dataset reference:**  
L. Bossard, M. Guillaumin, and L. Van Gool, “Food-101 – Mining Discriminative Components with Random Forests,” *European Conference on Computer Vision (ECCV)*, 2014.

---

*This README documents the project and the experiment results supplied for it. Confirm paths, dependency versions, and metric calculations against the final source code before release.*
"""

out = Path("/mnt/data/README.md")
pypandoc.convert_text(readme, "md", format="md", outputfile=str(out), extra_args=["--standalone"])
print(f"Created: {out}")

Analyzed
Python

from pathlib import Path
import pypandoc

readme = """# Food Dish Recognition System

A deep-learning-based food image classification system that recognizes food dishes across **101 categories** using transfer learning on the **Food-101 dataset**.

The project evaluates ImageNet-pretrained convolutional neural network architectures and provides a Streamlit application for predicting food categories from uploaded images.

## Project Highlights

- Food image classification across 101 Food-101 categories.
- Transfer learning with pretrained **EfficientNet-B0**, **ResNet50**, and **EfficientNet-B2**.
- Implemented using **Python, PyTorch, and Torchvision**.
- Evaluation using Top-1 accuracy, Top-5 accuracy, macro precision, macro recall, and macro F1-score.
- Streamlit interface for uploading an image and viewing predicted food categories with confidence scores.

## Dataset

This project uses the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

| Dataset detail | Value |
|---|---:|
| Number of food categories | 101 |
| Total images | 101,000 |
| Official training images | 75,750 |
| Official test images | 25,250 |
| Images per category | 1,000 |

For the full-data experiments, the official training split was divided into:

- **Training:** 68,175 images
- **Validation:** 7,575 images

The official test split contains 25,250 images and was used for final evaluation.

## Models and Experiments

Five experiment configurations were recorded:

| Experiment | Architecture | Training images | Best validation accuracy | Test Top-1 | Test Top-5 |
|---|---|---:|---:|---:|---:|
| 1 | EfficientNet-B0 | 5,050 | Not recorded | 45.63% | 72.31% |
| 2 | EfficientNet-B0 | 15,150 | 48.91% | 52.57% | 78.03% |
| 3 | ResNet50 | 8,000 | 61.90% | Not recorded | Not recorded |
| 4 | ResNet50 | 68,175 | 66.30% | Not recorded | Not recorded |
| 5 / Run 1 | EfficientNet-B2 | 68,175 | 69.58% | 74.70% | 93.35% |
| 5 / Run 2 | EfficientNet-B2 | 68,175 | See training log | **94.76%** | **98.65%** |

Run 2 was based on the existing EfficientNet-B2 checkpoint and fine-tuned further. The Run 2 evaluation reported:

| Metric | Run 2 result |
|---|---:|
| Top-1 test accuracy | **94.76%** |
| Top-5 test accuracy | **98.65%** |
| Macro precision | 85.34% |
| Macro recall | 84.70% |
| Macro F1-score | 84.42% |
| Test images | 25,250 |
| Evaluation time | 4.41 minutes |

**Metric verification note:** The reported Top-1 accuracy and macro recall should be checked against the saved predictions and metric-calculation code. On a balanced test set, standard macro recall across all classes should equal overall Top-1 accuracy. The values above are reproduced as supplied in the Run 2 evaluation output and are not silently adjusted here.

## Technology Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Computer vision:** Torchvision
- **Dataset:** Food-101
- **Model architectures:** EfficientNet-B0, ResNet50, EfficientNet-B2
- **Demo application:** Streamlit
- **Training environment:** Google Colab with GPU acceleration

## Application

The Streamlit application allows a user to:

1. Upload a food image.
2. Run the trained model on the image.
3. View the predicted food category or ranked candidate categories.
4. View the model's confidence scores for its predictions.

The model is designed for the 101 categories included in Food-101. It may not correctly classify food outside those categories. Confidence scores should be treated as model scores, not guaranteed probabilities of correctness.

## Setup

The exact dependency versions and entry-point filenames depend on the project files. The following is a general setup outline.

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\\Scripts\\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages used by the project and add their tested versions to a requirements file.

### 4. Prepare the dataset and model checkpoint

- Download Food-101 from its official source.
- Extract the dataset and configure the dataset path used by the training code.
- Place the trained model checkpoint at the path expected by the application.
- Keep large dataset files and model checkpoints out of Git unless the repository specifically requires them.

### 5. Run the Streamlit application

If the application entry point is `app.py`, run:

```bash
streamlit run app.py
```

Replace `app.py` with the actual application filename if it differs.

## Suggested Repository Structure

```text
Food-Dish-Recognition/
├── app.py
├── models/
│   └── best_model.pth
├── notebooks/
│   └── food101_training.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

This is an organizational suggestion; the actual repository may use a different structure.

## Evaluation Metrics

- **Top-1 accuracy:** Percentage of test images for which the highest-ranked predicted class is correct.
- **Top-5 accuracy:** Percentage of test images for which the correct class appears among the five highest-ranked predictions.
- **Macro precision:** Precision calculated separately for each class and averaged equally across all classes.
- **Macro recall:** Recall calculated separately for each class and averaged equally across all classes.
- **Macro F1-score:** F1-score calculated for each class and averaged equally across all classes.

## Limitations

- The classifier is limited to the 101 categories in Food-101.
- Visually similar dishes may be confused.
- Performance on images from real-world environments may differ from performance on the Food-101 test set.
- The application is a demonstration and has not been established as a production-ready dietary or nutritional assessment system.
- Food recognition alone does not provide a reliable estimate of portion size, calories, or nutritional content.

## Future Improvements

- Verify and report all metrics from a single saved set of test predictions.
- Evaluate performance per food category and inspect the confusion matrix.
- Test on additional real-world food images.
- Improve generalization through controlled fine-tuning and data augmentation experiments.
- Measure application inference latency and usability.

## Acknowledgements

The project uses the Food-101 benchmark dataset introduced by Bossard, Guillaumin, and Van Gool.

**Dataset reference:**  
L. Bossard, M. Guillaumin, and L. Van Gool, “Food-101 – Mining Discriminative Components with Random Forests,” *European Conference on Computer Vision (ECCV)*, 2014.

---

*This README documents the project and the experiment results supplied for it. Confirm paths, dependency versions, and metric calculations against the final source code before release.*

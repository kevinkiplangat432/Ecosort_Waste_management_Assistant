# EcoSort Waste Management Assistant
**Kaggle notebook:** [EcoSort Waste Management Assistant](https://www.kaggle.com/code/kevinkiplangat432/ecosort-waste-management-assistant)

>For the best viewing experience, including all cell outputs, plots, and confusion matrices, >view this notebook on Kaggle rather than through GitHub's notebook renderer, which can sometimes fail to render large notebooks with heavy output cleanly.

Module 8 Summative Lab, an integrated waste classification and disposal guidance system for Metro City, combining a CNN, a text classifier, and a retrieval augmented generation (RAG) system into a single assistant.

## Overview

EcoSort is a prototype AI system for Metro City's waste management department. A resident can provide either a photo of a waste item or a short text description, and the assistant identifies the material category and returns disposal instructions grounded in the city's actual recycling policy documents.

The system has three components, working together behind one entry point.

1. A CNN, built on MobileNetV2 transfer learning, classifies waste items from images into 9 material categories
2. A text classifier, TF IDF features with Logistic Regression, classifies waste items from short text descriptions into the same 9 categories
3. A RAG system retrieves and generates disposal instructions grounded in Metro City's policy documents, based on the classified category

## Datasets

**RealWaste** (UCI ML Repository, id 908), 4,752 images across 9 waste material classes, Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation. Class distribution is uneven, Plastic and Metal are the largest classes, Textile Trash the smallest.

**waste_descriptions.csv**, 5,000 short text descriptions of waste items labeled with the same 9 categories, close to evenly balanced.

**waste_policy_documents.json**, 14 Metro City recycling policy documents. 9 are single category, dedicated guidance for one material type, 5 bundle several categories together and repeat text from the single category documents.

## Notebook structure

The notebook follows the CRISP-DM methodology.

1. Business Understanding, problem framing and success criteria
2. Data Understanding, exploration of all three raw data sources
3. Data Preparation, image pipeline with class weighting, text pipeline with TF IDF, RAG document chunking and embedding
4. Exploratory Data Analysis, a look at the data as it reaches the models
5. Modelling, CNN architecture, MobileNetV2 transfer learning
6. Train the Model
7. Learning Curves
8. Evaluation, including a fine tuning step that unfreezes part of the base model
9. Text Classifier Modelling and Training
10. Text Classifier Evaluation
11. RAG System, retrieval and generation of disposal instructions
12. Integrated Assistant, single entry point combining all three components
13. Deployment, a Gradio interface for interactive testing
14. Final Reflections

## Results

**CNN**, 81.88 percent test accuracy after fine tuning, up from an 80.21 percent frozen baseline. Vegetation and Metal are the strongest classes, Miscellaneous Trash and Cardboard the weakest, Miscellaneous Trash's recall stayed at 0.67 even after fine tuning.

**Text classifier**, effectively 100 percent test accuracy. This reflects how directly category defining vocabulary appears in the description text rather than a modelling result on the same footing as the CNN's image classification task.

**RAG**, retrieval is reliable, every tested query returns the correct category's policy document, resolving an earlier issue where unrelated categories were retrieved. Generation is not yet reliable, the current setup, flan t5 base with beam search, tends to return document section headers rather than actual extracted instructions. This is documented as an open issue in Section 11 and Final Reflections rather than treated as solved.

**Integrated assistant**, correctly classifies and routes both image and text input to a shared output format, confirmed with a text input test. Image input testing was not separately exercised in the current run.

## Setup

### Running on Kaggle

Attach two datasets to the kernel.

1. `RealWaste`, the extracted image folder
2. `ecosort-data`, containing `waste_descriptions.csv` and `waste_policy_documents.json`

Enable GPU and internet access in kernel settings, GPU is needed for CNN training, internet is needed for `pip install` calls and downloading the pretrained models.

```bash
kaggle kernels push -p .
kaggle kernels status <username>/ecosort-waste-management-assistant
```

To pull back the executed notebook with outputs, download it from the kernel's page in the Kaggle web interface, File menu, Download Notebook, rather than `kaggle kernels output`, which only retrieves generated files, not the notebook with embedded cell outputs.

### Running locally

Place the extracted `RealWaste` folder, `waste_descriptions.csv`, and `waste_policy_documents.json` in the same directory as the notebook. A GPU is strongly recommended for the CNN training section, expect significantly longer training time on CPU.

## Dependencies

Installed within the notebook, `sentence-transformers`, `transformers`, `gradio`. Core dependencies, `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `Pillow`.

## Known limitations

RAG generation quality needs further work, the retrieval mechanism correctly identifies the right policy document for a given category, but the generation step does not yet reliably turn that context into specific, actionable instructions. A further prompt engineering pass, few shot examples, constrained generation forcing direct extraction, or a different generation model, would likely address this.

Miscellaneous Trash is the CNN's weakest performing class, likely reflecting how broad and visually inconsistent that catch all category is compared to more visually distinct classes.

Image input for the integrated assistant was not separately tested against a sample file in the current run, only text input was exercised end to end.

# Smiles

**State-of-the-art Multimodal Language-Vision Exploration Suite**

This repository provides reproducible guides and interactive notebooks for demonstrating four cutting-edge vision-language models. Each folder contains a Google Colab notebook, detailed instructions, and code examples to explore model capabilities.

## Models Included

- **BLIP-2**: Bootstrapped Language-Image Pre-training with frozen vision encoders and large language models.
- **Open-Flamingo**: Few-shot learning for vision-language tasks using visual and textual prompts.
- **LLaVA**: Large Language and Vision Assistant demo with interactive Gradio interface.
- **MedCLIP**: Contrastive learning from unpaired medical images and texts for diagnostic applications.

## Repository Structure
```text
Smiles/
├── Blip2/                   # BLIP-2 Colab guide, examples, tests
│   ├── README.md            # Instructions to reproduce BLIP-2 results
│   ├── blip2_colab.ipynb    # Colab notebook
│   ├── blip2_advanced_examples.py
│   └── blip2_test_cases.py
├── Flamingo/                # Open-Flamingo Colab instructions
│   ├── README.md
│   └── flamingo_colab.ipynb
├── LLaVA/                   # LLaVA demonstration guide and notebook
│   ├── README.md
│   └── llava_demo.ipynb
├── MedClip/                 # MedCLIP Colab notebook and documentation
│   ├── README.md
│   └── MedClip Colab.ipynb
└── Technical Presentation_Smiles.pdf  # Overview and key results
```

## Getting Started

Choose one of the following options:

1. **Google Colab**
   - Open the notebook in each model folder.
   - Enable GPU runtime (Runtime > Change runtime type > GPU).
   - Run cells sequentially to reproduce results.

2. **Local Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/denis-samatov/Smiles.git
     cd Smiles
     ```
   - Install Python 3.8+.
   - Follow dependency instructions in each folder's README.md.

## Usage

1. Navigate to a folder for the model you wish to explore.
2. Read the provided README.md for detailed setup and execution steps.
3. Run the Colab notebook or Python scripts to generate captions, answer questions, and visualize outputs.
4. Explore advanced examples and benchmark tests where available.

## Technical Presentation

- A comprehensive overview of the SMILES framework, architecture, and benchmark results is available in [Technical Presentation_Smiles.pdf].

# MeanFlow 🌊

![MeanFlow](https://img.shields.io/badge/MeanFlow-PyTorch-blue.svg)
![Release](https://img.shields.io/badge/Release-v1.0-orange.svg)

Welcome to the **MeanFlow** repository! This project offers a PyTorch implementation of the paper **"Mean Flows for One-step Generative Modeling"** by Geng et al. This repository aims to provide researchers and developers with a practical tool for understanding and utilizing mean flows in generative modeling.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Generative modeling has seen rapid advancements, particularly with the rise of diffusion models and flow-based methods. The **MeanFlow** implementation focuses on improving the efficiency and quality of generative models by leveraging the concept of mean flows. This approach allows for one-step generation, which simplifies the process and enhances performance.

## Features

- **Easy to Use**: The implementation is straightforward and user-friendly.
- **Flexible Architecture**: Customize the model architecture as per your requirements.
- **High Performance**: Achieve state-of-the-art results with optimized algorithms.
- **Comprehensive Documentation**: Detailed guides and examples are available.
- **Active Community**: Join discussions and contribute to the project.

## Installation

To install **MeanFlow**, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Mahmoudaboutaleb/MeanFlow.git
   cd MeanFlow
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. If you want to download the latest release, visit [Releases](https://github.com/Mahmoudaboutaleb/MeanFlow/releases). Download the latest version and follow the instructions provided.

## Usage

To use **MeanFlow**, follow these simple steps:

1. Import the necessary modules in your Python script:
   ```python
   import torch
   from mean_flow import MeanFlowModel
   ```

2. Initialize the model:
   ```python
   model = MeanFlowModel()
   ```

3. Train the model with your dataset:
   ```python
   model.train(your_dataset)
   ```

4. Generate samples:
   ```python
   samples = model.generate(num_samples=100)
   ```

5. For detailed examples, refer to the [examples](examples/) directory.

## Model Architecture

The **MeanFlow** model is based on a flow-based architecture that allows for efficient sampling and density estimation. The key components include:

- **Flow Layers**: These layers transform simple distributions into complex ones.
- **Mean Calculation**: The model calculates the mean flow to enhance generation quality.
- **Diffusion Process**: Integrates diffusion models to refine the output.

![Model Architecture](https://example.com/model_architecture.png)

## Training

To train the model effectively:

1. Prepare your dataset. Ensure it is in the correct format.
2. Set hyperparameters in the `config.py` file.
3. Run the training script:
   ```bash
   python train.py --config config.py
   ```

4. Monitor the training process. You can visualize the loss and generated samples.

For more detailed training instructions, check the [Training Guide](docs/training_guide.md).

## Evaluation

Evaluating the model is crucial for understanding its performance. To evaluate the trained model:

1. Load the trained model:
   ```python
   model.load('path_to_trained_model.pth')
   ```

2. Use evaluation metrics such as Inception Score (IS) or Fréchet Inception Distance (FID) to assess the quality of generated samples.

3. Generate and visualize samples:
   ```python
   samples = model.generate(num_samples=100)
   visualize(samples)
   ```

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```

3. Make your changes and commit them:
   ```bash
   git commit -m "Add your feature"
   ```

4. Push to your branch:
   ```bash
   git push origin feature/YourFeature
   ```

5. Create a pull request.

Please ensure your code follows the project's coding standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Geng et al. for their groundbreaking work on mean flows.
- The PyTorch community for their continuous support and resources.
- All contributors who have helped improve this project.

For the latest updates and releases, visit [Releases](https://github.com/Mahmoudaboutaleb/MeanFlow/releases). Download the latest version and follow the provided instructions to get started with **MeanFlow**.
# LUCID++: Enhanced Deep-Learning Solution for DDoS Attack Detection

## Project Metadata
### Authors
- **Team:** Syed M Minhaj Naqvi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Distributed Denial of Service (DDoS) attacks are one of the most disruptive threats to network security, often overwhelming critical infrastructure and degrading service availability. traditional Intrusion Detection Systems (IDS) rely heavily on signature-based or rule-based approaches which may fail against large-scale attack patterns.

Recent research demonstrates that deep learning can provide effective mitigation of such attacks, enabling automated feature extraction and robust detection of malicious traffic flows. One such approach is LUCID (Lightweight CNN for DDoS Detection), which applies convolutional neural networks to flow-level traffic classification. LUCID achieves strong results while remaining lightweight enough for practical deployment.

Despite this progress, challenges remain in ensuring robustness, minimizing false positives, and improving generalization across diverse datasets. This project, LUCID++, proposes enhancements to address these challenges while keeping efficiency in mind.

## Problem Statement
While LUCID has demonstrated the feasibility of deep learning for lightweight DDoS detection, limitations remain in areas such as dataset diversity, performance consistency across different environments, and fine-tuning for real-world deployment.

We will focus on how we can extend and refine the LUCID framework to improve detection accuracy, reduce false positives, and potentially even strengthen cross-dataset generalization while maintaining low computational cost suitable for SOC environments.

## Application Area and Project Domain
The project is situated at the intersection of cybersecurity and deep learning, with a focus on network intrusion detection systems (NIDS). The intended application is real-time traffic monitoring within Security Operations Centers (SOCs).

## What is the paper trying to do, and what are you planning to do?
The original LUCID work focuses on applying a convolutional neural network to network flow features for lightweight and accurate DDoS detection. It shows that a deep learning–based approach can outperform traditional detection methods across benchmark intrusion detection datasets.

**LUCID++** aims to:

- Model Optimization: Investigate methods to improve performance metrics (accuracy, F1-score, macro-F1, AUC) through refined training and evaluation strategies.

- Data Handling Enhancements: Explore better preprocessing and class-imbalance handling to improve detection of diverse and minority attack patterns.


If time permits, we will also explore:

- Cross-Dataset Evaluation: Test the approach on multiple standard IDS datasets to assess robustness and generalization.

- Post-processing Techniques: Consider lightweight strategies to reduce false positives and stabilize detection outputs.

- Deployment Considerations: Ensure the solution remains efficient and practical for SOC environments.


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection](https://ieeexplore.ieee.org/document/8984222)

### Reference Dataset
- [UNB CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)


## Project Technicalities

### Terminologies
- **DDoS Attack:**  A malicious attempt to disrupt a network or service by overwhelming it with a flood of internet traffic.
- **Lightweight Model** A neural network with a very small number of parameters, designed to run on resource-constrained devices (e.g., edge gateways).
- **Flow-based Detection:** The technique of grouping packets by their source/destination (bi-directional flow) before analysis.
- **Flow Fragment**  A sample of traffic containing the first n packets (e.g., 10 packets) of a flow within a specific t second time window (e.g., 10 seconds).
- **1D Convolution:** A convolutional operation that slides in only one dimension. In this project, it is implemented with a Conv2D layer where the kernel width matches the feature width, forcing it to convolve only along the time (packet) axis.
- **Batch Normalization:** A layer that stabilizes and accelerates training by normalizing the output of the previous layer.
- **AdamW:** An optimizer that improves upon Adam by implementing weight decay more effectively, which helps prevent overfitting.
- **GridSearchCV:** A Scikit-learn utility used to perform an exhaustive search over a specified parameter grid to find the best-performing model hyperparameters.

### Problem Statements
- **Validate LUCID on a Modern Dataset:** The original LUCID paper demonstrated high performance on datasets from 2012, 2017, and 2018. This project's first goal was to establish a performance baseline for this architecture on the newer, more complex CIC-DDoS2019 dataset.
- **Improve Baseline Performance:** The second goal was to enhance the original LUCID architecture with modern deep learning techniques to improve its detection accuracy and balance.
- **Address the Performance-Overhead Trade-off:** While many deep learning models are effective, their computational cost makes them impractical for real-time edge deployment. We aim to improve LUCID's accuracy while maintaining its lightweight philosophy.
- **Reduce False Positives:** In a real-world setting, a high False Positive Rate (FPR) can overwhelm security analysts with "alert fatigue." A key goal was to find a model that balances a high detection rate (TPR) with a very low FPR.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.

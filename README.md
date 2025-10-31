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

### Project Goals and Motivation
- **Validate LUCID on a Modern Dataset:** The original LUCID paper demonstrated high performance on datasets from 2012, 2017, and 2018. This project's first goal was to establish a performance baseline for this architecture on the newer, more complex CIC-DDoS2019 dataset.
- **Improve Baseline Performance:** The second goal was to enhance the original LUCID architecture with modern deep learning techniques to improve its detection accuracy and balance.
- **Address the Performance-Overhead Trade-off:** While many deep learning models are effective, their computational cost makes them impractical for real-time edge deployment. We aim to improve LUCID's accuracy while maintaining its lightweight philosophy.
- **Reduce False Positives:** In a real-world setting, a high False Positive Rate (FPR) can overwhelm security analysts with "alert fatigue." A key goal was to find a model that balances a high detection rate (TPR) with a very low FPR.

### Our Proposed Solution: LUCID++
To achieve these goals, we introduce LUCID++, an enhanced version of the original architecture. The key enhancements are:

- **AdamW Optimizer:** We replace the standard Adam optimizer with AdamW and include weight_decay in our hyperparameter search, improving model generalization.
- **Batch Normalization:** We add a BatchNormalization layer directly after the convolutional layer. This stabilizes training and allows the model to converge to a more robust solution.
- **Hidden Dense Layer:** We add a small, fully-connected (Dense) layer after the pooling/flatten stage. This gives the model additional capacity to learn complex, non-linear combinations of the features extracted by the CNN, leading to more sophisticated classification.

This repository provides the code to preprocess the data, run the GridSearchCV to find the best model, and evaluate its performance.

### Key Components
- **lucid_cnn_original.py** and **lucid_cnn_enhanced.py** are the two main Python scripts. They contains the logic for training (--train), hyperparameter tuning (GridSearchCV), and evaluation (--predict).
- **lucid_dataset_parser.py** is the preprocessing script. This must be run first to convert raw .pcap files into the .hdf5 datasets used for training.
- **util_functions.py** is a utility file containing helper functions for data loading (load_dataset) and preprocessing (normalize_and_padding).
- **requirements.txt** is a list of all required Python libraries.

## Model Workflow
The project is divided into two main stages: Preprocessing and Training.

1. **Preprocessing Workflow:**
   - **Input:** Raw .pcap traffic files from the CIC-DDoS2019 dataset.
   - **Parsing:** lucid_dataset_parser.py (using PyShark) reads the .pcap files.
   - **Flow Grouping:** Packets are grouped into bi-directional flows and segmented into 10-second time windows.
   - **Fragmentation:** Each flow is sampled into fragments of the first 10 packets.
   - **Feature Extraction:** 11 features (e.g., packet length, IP flags, TCP flags) are extracted for each packet.
   - **Normalization & Saving:** The fragments are normalized, padded to a uniform (10, 11) shape, and saved as three .hdf5 files: train, val, and test.

2. **Training Workflow:**
   - **Input:** The preprocessed .hdf5 files.
   - **Data Loading:** The lucid_cnn_*.py scripts loads the data and performs the final float32 type casting and label reshaping.
   - **Hyperparameter Search:** GridSearchCV is initialized with the LUCID++ architecture and a grid of 1920 parameter combinations.
   - **Training & Validation:** The script trains and validates 1,920 models (384 combinations x 5 folds) to find the single best set of hyperparameters.
   - **Final Fit:** GridSearchCV automatically re-trains a final "champion" model on the combined training and validation data using the best parameters.
   - **Output:** The script saves the best model as a .h5 file and prints the final classification report and metrics.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/LUCID-Enhanced-Deep-Learning-Solution-for-DDoS-Attack-Detection.git
    cd LUCID-Enhanced-Deep-Learning-Solution-for-DDoS-Attack-Detection
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run Preprocessing:**
    Run the following commands step-by-step:
    ```bash
    python lucid_dataset_parser.py --dataset_type DOS2019 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id DOS2019 --time_window 10
    python lucid_dataset_parser.py --preprocess_folder ./sample-dataset/
    ```

4. **Run Training & Hyperparameter Search:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python lucid_cnn_*.py --train ./sample-dataset/ --epochs 1000 --cv 5
    #The "*" is wildcard. For baseline case, use "lucid_cnn_original.py" and for enhanced case use "lucid_cnn_enhanced.py" 
    ```
    At the end, the best model will be saved to the ./output/ directory and the final parameters will be printed to the console.

5. **Evaluate the Model:**
    To generate the ROC curve and classification report from the saved model, you can use a separate evaluation script.
    ```bash
    python  lucid_cnn_*.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-DOS2019-LUCID.h5
    #Again, "*" is wildcard. For baseline case, use "lucid_cnn_original.py" and for enhanced case use "lucid_cnn_enhanced.py" 
    ```
    

## Acknowledgments
- **Original Authors:** This project is a direct extension of the work by Doriguzzi-Corin, et al. [1].
- **Dataset Provider:** The Canadian Institute for Cybersecurity (CIC) for providing the high-quality [CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html).
- **Individuals:** Special thanks to Dr. Muzammil Behzad for the invaluable guidance and support throughout this project.

## References
[1] R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón, and D. Siracusa, "LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876-889, 2020.

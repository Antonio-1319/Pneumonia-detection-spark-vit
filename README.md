# Scalable Pneumonia Detection using Apache Spark & Vision Transformers

This project presents a scalable Deep Learning pipeline for the automated detection of Pneumonia from Chest X-Ray images dataset. Integrating Apache Spark for distributed data processing with a state-of-the-art Vision Transformer (ViT) architecture, the system classifies medical images with high precision.

A pre-trained ViT model ([nickmuchi/vit-finetuned](https://huggingface.co/nickmuchi/vit-finetuned-chest-xray-pneumonia)) is integrated into a PySpark ML pipeline using Pandas UDFs, merging big-data scalability with the strong feature-learning capabilities of Transformers. The result is a reliable and scalable system that can be applied to large image datasets.

## Project Structure

The project is organized into a single logical pipeline within the notebook, divided into the following sections:

### **0. Load dependencies**
Initialization of the environment, installing necessary libraries (`spark-nlp`, `pyspark`, `transformers`) and configuring the Spark Session with GPU support.

### **1. Data loading**
Automated ingestion of the Chest X-Ray dataset from Kaggle. We utilize Spark's **`binaryFile`** data source to efficiently load unstructured image data as raw bytes, preserving metadata (labels, split type) without decoding the images in the driver.

### **2. Implementing the pipeline**
* **Inference Engine:** We implemented a **Pandas UDF** (`hf_pneumonia_binary_udf`) that leverages GPU acceleration (CUDA) to deserialize raw binary images and run inference using the **"Nickmuchi" Vision Transformer**.
* **Spark Wrapper:** The `HuggingFaceBinaryClassifier` class extends Spark's `Transformer`, integrating the PyTorch model into the pipeline to output standard binary predictions.
### **3. Model Evaluation**
Performance assessment on the independent **Test Set** (624 images) and **Validation Set**. We calculate key metrics such as Accuracy, F1-Score, and AUC-ROC to validate the model's clinical relevance.

### **4. Conclusions**
Final analysis of the results, discussing the scalability of the solution and the robustness of Vision Transformers for medical imaging.

---

## Tools & Technologies

* **Apache Spark (PySpark):** Used for orchestration and ETL. It handles the loading and distribution of binary image data across the cluster.
* **Pandas UDFs (User Defined Functions):** The critical component that allows Spark to execute Python-native Deep Learning code (PyTorch) efficiently on distributed DataFrames.
* **Vision Transformer (ViT):** We use the `nickmuchi/vit-finetuned-chest-xray-pneumonia` model from Hugging Face. ViT processes images as sequences of patches, using self-attention to capture global context.
* **PyTorch:** The underlying framework for running the deep learning model inference.

---

## Results

The model achieved outstanding performance, demonstrating both high accuracy and stability across different data splits.

| Metric | Score |
| :--- | :--- | 
| **Accuracy** | **95.35%** | 
| **F1-Score** | **95.35%** | 
| **AUC-ROC** | **0.9916** | 
| **Validation Acc** | **100%** | 

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/pneumonia-detection-spark-vit.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute the Notebook:**
    Open `Final_project_L_Cano_A_Gil.ipynb` in Google Colab or a local Jupyter environment. You will need a `kaggle.json` API token to download the dataset automatically.

---

## Authors

Final Project - Big Data Engineering - 2025/2026

* **Laura Cano** - *Master in Computational Biology, UPM*
* **Antonio Gil** - *Master in Computational Biology, UPM*

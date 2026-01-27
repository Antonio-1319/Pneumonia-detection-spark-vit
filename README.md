# Scalable Pneumonia Detection using Apache Spark & Vision Transformers

This project presents a scalable Deep Learning pipeline for the automated detection of Pneumonia from Chest X-Ray images. Integrating Apache Spark for distributed data processing with a state-of-the-art Vision Transformer (ViT) architecture, the system classifies medical images with high precision.
A pre-trained ViT model ([nickmuchi/vit-finetuned](https://huggingface.co/nickmuchi/vit-finetuned-chest-xray-pneumonia)) is integrated into a PySpark ML pipeline using Pandas UDFs, merging big-data scalability with the strong feature-learning capabilities of Transformers. The result is a reliable and scalable system that can be applied to large image datasets.

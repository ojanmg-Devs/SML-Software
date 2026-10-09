# SML-Software
SML is a desktop application designed to streamline time-series data analysis and machine learning workflows.
 It integrates multi-domain feature extraction including time, Fourier, and frequency-domain transformations with a diverse library of neural networks and statistical models like MLP, CNNs, LSTMs, and Gradient Boosting. Engineered for multi-modal sensor and time-series data
 SML automates the entire pipeline from feature extraction and hyperparameter tuning to interactive visual analytics and automated Word report generation.


Core Novel Technical Highlights:

 1. Multi-Domain Feature Extraction Pipeline 
Custom Signal Processing: The code implements time-domain sliding windows alongside spectral transformation modes (sliding, fourier, and frequency). 
Advanced Spectral Descriptors: Rather than using simple aggregations, the frequency-domain extractor calculates Spectral Centroid and Spectral Bandwidth via real Fast Fourier Transforms (RFFT) on zero-mean centered signals.
2. Unified Multi-Paradigm Execution Engine
Hybrid Deep Learning & Statistical ML Architecture: The platform bridges PyTorch deep architectures (MLP, 1D-CNN, LSTM, and RBF Networks with learnable Gaussian centers) alongside classical and tree-based estimators (Decision Trees, Random Forest, HistGradientBoosting, SVM, and Linear/Logistic Regression) under a single unified tensor/numpy pipeline.
Automated Dimensionality Handling: The pipeline dynamically manages 2D tabular features versus 3D spatial-temporal tensor shapes (Batch, Seq_Len, Features) depending on the selected estimator family. 

3. Multi-File Dataset Concatenation & Data Normalization
Automated Data Sanitization: The software features non-numeric string cleaning, sniffer-based CSV delimiter parsing, automatic multi-dataset appending, and header validation routines. 
4. Automated Hyperparameter Grid Search Engine 
Automated Model Optimization: Implements custom grid candidate generation and asynchronous thread-safe training evaluation. 
5. Evaluation Reporting
Programmatic Word/DOCX Export: Generates structured research reports directly into Word documents using python-docx (including confusion matrices, class-level precision, recall, F1 breakdowns, and loss,accuracy per epoch curves).  





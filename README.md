# Lowering Emergency Room Patient Wait Times with Machine Learning & Process Mining

## A thesis showcasing how machine learning algorithms and process mining can be utilized for pre-arrival administration of medications and treatments in hospital emergency rooms.

For a detailed descriptions on how the study was conducted please take a look at the thesis pdf document uploaded.

This project is part of my undergraduate thesis, utilizing the MIMIC-IV dataset from Harvard Medical School to explore how medication administration can be predicted using patient vitals (e.g., blood pressure, heart rate), acuity (injury/disease severity), and other factors. The study focuses primarily on Urinary Tract Infections (UTIs) and Brain Trauma Injuries (BTIs), which are challenging to assess due to their variable severity, often leading to time-consuming treatment processes. 

How machine learning and process mining can be used in hospital information systems:
![image](https://github.com/user-attachments/assets/0785d172-482c-4fff-a251-17d149422485)

The project's structure is outlined as follows:

1. Dataset Cleaning and Preprocessing: This involves removing blank entries and nonsensical values, renaming columns, and ensuring consistent values across all datasets.
2. Medication Administration Filtering: This step filters medications based on key ingredients and standardizes names for similar medications of similar doses.
3. Analyzing Important Metrics: The data is further analyzed to retain only key parameters and features, ensuring accurate training of the model.
4. Model Training: The model is trained using Random Forests and Decision Trees to visualize interactions between variables.
5. Model Testing and Performance: The model's strengths and weaknesses are evaluated by comparing predictions with the testing dataset, utilizing tools like pm4py and Graphviz to assess and visualize performance.

By focusing on the structured approach above, the project aims to refine predictive accuracy in emergency department workflows for UTI and BTI patients, with potential implications for improved efficiency and patient care.

## Key Findings

The study successfully formatted complex patient data into structured event logs and developed a predictive machine learning model using patient vitals to anticipate medication administration. However, the model's accuracy was limited by insufficient patient data and encryption constraints, impacting its precision in medication predictions. The research highlights the potential for improved data quality and additional patient-specific information to significantly enhance the model's predictive capabilities and applicability in real-world medical settings, thereby optimizing emergency care processes.

## Key Improvements

The Random Forests model showed improved prediction accuracy for medication administration over traditional decision trees, particularly in multi-classification tasks where a diverse array of decision paths is necessary. However, a class imbalance persisted, leading to poor prediction for less frequent medication classes. Efforts to weight these less frequent classes higher revealed that enhancing their representation in training could lead to greater model accuracy and confidence. Additionally, the study found that the model's precision in predicting specific medications could be improved by incorporating additional patient-specific data and employing stratified sampling to ensure each class is proportionally represented. The findings underscore the potential for future improvements in accuracy by focusing on better data quality and strategic class weighting​.

## How to Use

Since this is an undergraduate thesis study, please do not copy or modify this. However, feel free to use this as a reference or guide for your own projects or studies.

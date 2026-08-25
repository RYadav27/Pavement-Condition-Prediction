# Pavement Condition Assessment and Maintenance Priority Prediction

A Civil Engineering and Machine Learning project for analyzing pavement condition using the Pavement Condition Index (PCI) and assigning maintenance priorities to road segments.

## 📌 Project Overview

Pavement condition assessment is an important part of transportation and highway engineering. Poor pavement conditions can increase vehicle operating costs, reduce road safety, and require expensive rehabilitation.

This project uses pavement condition data containing:

- Road segment ID
- Pavement Condition Index (PCI)
- Pavement condition description

The project cleans the data, analyzes pavement conditions, assigns maintenance priorities, trains a machine learning model, evaluates its performance, and generates visualizations.

## 🎯 Objectives

The main objectives of this project are:

1. Clean and preprocess pavement condition data.
2. Analyze the distribution of pavement condition.
3. Classify road segments according to PCI.
4. Assign maintenance priorities.
5. Train a Random Forest classification model.
6. Evaluate the model using accuracy and classification metrics.
7. Generate useful pavement condition visualizations.

## 📊 Dataset

The project uses a CSV dataset containing pavement condition information.

### Dataset Columns

| Column | Description |
|---|---|
| `seg_id` | Unique road segment identifier |
| `pci` | Pavement Condition Index |
| `pci_desc` | Description of pavement condition |

### PCI

PCI is a numerical indicator of pavement condition ranging from 0 to 100.

- Higher PCI → Better pavement condition
- Lower PCI → Poorer pavement condition

## 🛠️ Maintenance Priority Classification

The project assigns maintenance priority based on PCI:

| PCI Range | Priority | Interpretation |
|---|---|---|
| 0–24 | Critical | Immediate rehabilitation required |
| 25–39 | High | Major maintenance recommended |
| 40–54 | Medium | Maintenance planning recommended |
| 55–69 | Low | Minor maintenance may be required |
| 70–100 | Routine | Routine monitoring |

## 📁 Project Structure

```text
Pavement-Condition-Prediction/
│
├── data/
│   ├── pavement_data.csv
│   └── processed_pavement_data.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── visualization.py
│
├── pavement_model.pkl
├── pci_distribution.png
├── condition_distribution.png
├── maintenance_priority.png
├── pci_by_priority.png
│
├── requirements.txt
└── README.md

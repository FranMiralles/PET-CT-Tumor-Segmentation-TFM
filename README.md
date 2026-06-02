# PET-CT-Tumor-Segmentation-TFM
Research work on segmentation of tumors on PET-CT images for my master thesis.

## Structure of the project

The project follows a modular and scalable structure designed to support the full machine learning lifecycle.

```plaintext
project/
│
├── data/                      # Raw and processed datasets
├── docs/                      # Project documentation
│   └── conceptual_model/      # Data conceptual model and supporting documents
│
├── logs/                      # Logs generated during execution
│
├── notebooks/                 # Exploratory Data Analysis notebooks
│   └── eda
│
├── reports/                   # Generated reports, metrics, and visualizations
│
├── src/                       # Source code (core of the project)
│   ├── config/                # Configuration files
│   ├── data/                  # Data ingestion, validation, transformation
│   ├── entity/                # Data schemas and entity definitions
│   ├── features/              # Feature engineering logic
│   ├── models/                # Model training, evaluation, prediction
│   ├── pipelines/             # Training and inference pipelines
│   └── utils/                 # Utilities (logging, exceptions, helpers)
│
├── tests/                     # Unit and integration tests
├── trained_models/            # Saved trained models (artifacts)
│
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```
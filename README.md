# Project: Boilerplate ML Pipeline

ML Pipeline is a machine learning pipeline system focused on ensuring scalability, reproducibility, and flexibility across various projects. While this documentation demonstrates the usage of BERT and the Rotten Tomatoes dataset as an example, boilerplate's design is modular. This allows developers to quickly and seamlessly integrate other models or datasets into the pipeline.

## Table of Contents

- [Project: Boilerplate ML Pipeline](#project-boilerplate-ml-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Directory Structure](#directory-structure)
  - [Integrating New Models or Datasets](#integrating-new-models-or-datasets)
  - [Logging and Monitoring with MLflow](#logging-and-monitoring-with-mlflow)
  - [Contributions](#contributions)
  - [License](#license)

## Features

1. **Modularity**: Easily plug in different preprocessing, models, or datasets.
2. **Scalability**: Designed with scalability in mind. Effortlessly switch from local experiments to cloud deployments.
3. **Reproducibility**: MLflow integration ensures tracking of every experiment, making them reproducible at any time.
4. **End-to-End Workflow**: From data fetching, cleaning, preparing, to training and testing - it's all in one place.

## Installation

Clone the repository:

```
git clone https://github.com/rmarquet21/boilerplate-ml-pipeline.git
```

Navigate to the project directory:

```
cd boilerplate-ml-pipeline
```

Install the required packages:

```
pip install -r requirements.txt
```

## Quick Start

1. Set up MLflow tracking:

```bash
alfred run:server
```

2. Run the pipeline:

```bash
alfred run:pipeline
```

3. Visit `http://localhost:5000` in your browser to view MLflow's UI and monitor the progress.

## Directory Structure

```
.
├── alfred
│   └── run.py
├── poetry.lock
├── pyproject.toml
├── training_pipeline
│   ├── __init__.py
│   ├── run.py
│   ├── pipeline_context.py
│   └── steps
│       ├── __init__.py
│       ├── base_step.py
│       ├── clean_data_step.py
│       ├── fetch_data_step.py
│       ├── prepare_data_step.py
│       └── train_data_step.py
```

## Integrating New Models or Datasets

1. **Datasets**: To integrate a new dataset, extend the `FetchDataStep` in `fetch_data_step.py`. Use the `load_dataset` method or any other preferred method to fetch your data.

2. **Models**: To work with a different model, extend the `PrepareDataStep` for data tokenization/preparation and the `TrainDataStep` for training the model.

Remember, the pipeline is built with modularity in mind. Each step works as an independent module, ensuring flexibility and scalability.

## Logging and Monitoring with MLflow

Repo comes integrated with MLflow for experiment tracking. Log metrics, parameters, and even save model checkpoints. The `FetchDataStep` demonstrates the basic usage of MLflow. Extend it by logging more parameters, metrics, or artifacts.

## Contributions

Contributions are always welcome. If you want to contribute, please:

1. Fork the project.
2. Create a new branch.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

## License

MIT License. See [LICENSE](LICENSE) for more information.

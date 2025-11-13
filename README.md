# Crypto & Stock Price Prediction MLOps Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-orange)](https://dvc.org/)

Production-ready MLOps pipeline for cryptocurrency and stock price prediction using hybrid LSTM + XGBoost model with automated CI/CD, data drift detection, and model monitoring.

## 🎯 Project Goals

- Demonstrate production-ready ML engineering skills
- Implement end-to-end MLOps best practices
- Showcase automated training, testing, and deployment
- Build scalable and maintainable ML systems

## 🏗️ Architecture

*[Architecture diagram will be added]*

## 🚀 Features

- ✅ Hybrid LSTM + XGBoost model for time series prediction
- ✅ Automated CI/CD pipeline with GitHub Actions
- ✅ Data versioning with DVC
- ✅ Experiment tracking with MLflow
- ✅ Data drift detection with Evidently AI
- ✅ RESTful API with FastAPI
- ✅ Docker containerization
- ✅ Comprehensive testing (unit, integration, model tests)
- ✅ Code quality with Ruff

## 📦 Tech Stack

- **ML Frameworks:** PyTorch (LSTM), XGBoost
- **MLOps:** MLflow, DVC, Evidently AI
- **API:** FastAPI, Uvicorn
- **CI/CD:** GitHub Actions
- **Containerization:** Docker, Docker Compose
- **Code Quality:** Ruff, pre-commit, pytest

## 🛠️ Setup

### Prerequisites

- Python 3.12+
- Docker Desktop
- Git
- **uv** (Python Package Manager)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/crypto-stock-mlops-pipeline.git
cd crypto-stock-mlops-pipeline

# Create and activate virtual environment with Python 3.12
uv venv --python 3.12

# Install all dependencies from pyproject.toml
uv pip install -e .[dev]

# (Optional) To install from the lock file for exact reproducibility
# uv sync --dev

# Install pre-commit hooks
pre-commit install

# Initialize DVC
dvc pull
```

## 📊 Dataset

- **Source:** Kaggle - Daily Crypto & Stocks Market Data (2025 - Present)
- **Features:** Open, High, Low, Close, Volume, Market Cap
- **Assets:** BTC, ETH, AAPL, GOOGL

## 🔄 Development Status

- [x] Project setup and configuration
- [ ] Data pipeline implementation
- [ ] Model development
- [ ] MLflow integration
- [ ] CI/CD pipelines
- [ ] API development
- [ ] Monitoring setup
- [ ] Documentation

## 📝 License

MIT License

## 👤 Author

**Orhan Sonmez**
- GitHub: [@orhansonmeztr](https://github.com/orhansonmeztr)

.PHONY: setup test clean run docker-build docker-run

# Default target
all: setup

# Setup development environment
setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# Run tests
test:
	python -m unittest discover tests

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf outputs/*
	rm -rf data/processed/*

# Run the pipeline on a sample document
run:
	python run_pipeline.py data/samples/sample_invoice.png outputs

# Run the pipeline on all documents in data/raw
run-batch:
	python run_pipeline.py --batch data/raw outputs

# Build Docker image
docker-build:
	docker build -t ocr-nlp-pipeline .

# Run Docker container
docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs ocr-nlp-pipeline python run_pipeline.py --batch data/raw outputs

# Start with docker-compose
docker-up:
	docker-compose up --build

# Help
help:
	@echo "Available targets:"
	@echo "  setup      - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  clean      - Remove generated files"
	@echo "  run        - Run pipeline on sample document"
	@echo "  run-batch  - Run pipeline on all documents in data/raw"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-up    - Start with docker-compose"

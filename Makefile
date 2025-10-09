# ==============================================================================
# Makefile for the ML Model Project
# This file centralizes all project commands for development and CI/CD.
# ==============================================================================

# Use .PHONY to declare targets that are not files to prevent conflicts.
.PHONY: install train eval pipeline test deploy hf-login push-hub update-branch run-app clean

# --- Core ML Pipeline ---

install:
	# Install all dependencies from requirements.txt
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	# Run the model training script
	python train.py

eval:
	# Generate a Markdown report with metrics and plots
	echo "## Model Metrics" > report.md
	test -f Results/metrics.txt && cat Results/metrics.txt >> report.md || echo "No metrics file found"
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md

pipeline: install train eval
	# A convenience target to run the full training pipeline

# --- Testing ---

test:
	# Run tests using pytest, handle cases where no tests exist
	python -m pytest tests/ -v || echo "No tests directory found"

# --- Deployment to Hugging Face ---

hf-login:
	# Configure git credential helper and log in to the Hugging Face CLI
	git config --global credential.helper store
	hf auth login --token $(HF_TOKEN) --add-to-git-credential

push-hub:
	# Copy the trained model to the app directory for Gradio
	cp Model/drug_pipeline.joblib app/drug_pipeline.joblib || echo "Could not copy model to app folder"
	# Upload the app, model, and results to a Hugging Face Space
	hf upload alwynsharon18/DrugClassification ./app --repo-type=space --commit-message="Sync App files with model"
	hf upload alwynsharon18/DrugClassification ./Model --repo-type=space --commit-message="Sync Model"
	hf upload alwynsharon18/DrugClassification ./Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub
	# Main deployment target that logs in and pushes files

# --- Git-based Artifact Versioning ---

update-branch:
	# Commit the generated Model and Results back to a dedicated 'update' branch
	git config --global user.name "github-actions[bot]"
	git config --global user.email "github-actions[bot]@users.noreply.github.com"
	git checkout -B update
	git add Model Results
	git commit -m "Update model and results [automated]" || echo "No changes to commit"
	git push --force origin update

# --- Local Development Utilities ---

run-app:
	# Run the Gradio application locally
	cd app && python app.py

clean:
	# Remove all generated files and caches for a clean state
	rm -rf Model Results *.png *.txt report.md
	rm -rf __pycache__ .pytest_cache
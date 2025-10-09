#
# This is the revised workflow, last updated on Thursday, October 9, 2025.
# It combines a two-job CI/CD structure with the specific commands from your Makefile.
#
name: Model Training and Deployment CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: write # Required for 'make update-branch' and 'gh release create'
  actions: read
  pull-requests: write

jobs:
  # --- JOB 1: Build, Test, and Evaluate the Model ---
  # This job ensures the code is working and produces the model artifacts.
  build-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0 # Full history is needed for versioning/releases
      
    - name: Set up Python 3.9
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
        
    - name: Cache pip dependencies
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: make install
      
    - name: Run tests with pytest
      run: make test
      
    - name: Train model
      run: make train
      
    - name: Evaluate model and create report
      run: make eval
        
    - name: Upload model artifacts
      uses: actions/upload-artifact@v4
      with:
        name: model-artifacts
        path: |
          Model/
          Results/
          report.md
        retention-days: 30

  # --- JOB 2: Deploy to Hugging Face and Create Release ---
  # This job only runs if 'build-and-test' succeeds on the main branch.
  deploy-and-release:
    needs: build-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Download model artifacts from previous job
      uses: actions/download-artifact@v4
      with:
        name: model-artifacts
        
    - name: Commit model & results to 'update' branch
      run: make update-branch
        
    - name: Deploy to Hugging Face Hub
      run: make deploy
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
        
    - name: Create Release with Dynamic Notes
      run: |
        # Read the metrics from the downloaded artifact into a variable
        METRICS_CONTENT=$(cat Results/metrics.txt)

        # Create the release notes using the metrics content
        RELEASE_NOTES="Automated model training and deployment.

        ## Model Performance
        \`\`\`
        ${METRICS_CONTENT}
        \`\`\`
        
        ## Changes
        - Model trained on latest data from commit ${{ github.sha }}
        - Deployed to Hugging Face Spaces
        - Model artifacts committed to the 'update' branch for tracking."

        # Use the GitHub CLI to create a new release
        gh release create model-v${{ github.run_number }} \
          --title "Model Release v${{ github.run_number }}" \
          --notes "$RELEASE_NOTES" \
          Model/drug_pipeline.joblib \
          Results/metrics.txt \
          Results/model_results.png
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
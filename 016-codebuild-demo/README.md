# ☁️ Python APP AWS CodeBuild Demo

A Python application demonstrating a GitHub-integrated AWS CodeBuild CI pipeline with automated builds, logging, and cost-conscious teardown.
![Graphical Summary](attachments/codebuild-demo.png)

## 📂 Table of Contents
- [Overview](#-overview)
- [CI/CD Pipeline Walkthrough](#-cicd-pipeline-walkthrough)
- [Cost Management](#-cost-management)

## 🧠 Overview
This project demonstrates a Python application integrated with an AWS CodeBuild CI pipeline. The focus is on build automation, GitHub integration, and cost-aware cloud usage.

- Integrated GitHub repository with AWS CodeBuild
- Defined build and test phases using buildspec.yml
- Builds triggered via GitHub webhook
- Artifacts stored temporarily in S3

## 📊 CI/CD Pipeline Walkthrough

This CI/CD pipeline automatically builds and tests a Python application whenever changes are pushed to GitHub, using AWS CodeBuild for isolated, reproducible builds.

### Step 1: Source Control & Trigger
The source code is hosted on GitHub. A webhook triggers AWS CodeBuild whenever changes are pushed to the main branch.

### Step 2: Build Environment Setup
AWS CodeBuild provisions an ephemeral Linux build environment using a managed image with Python preinstalled. This ensures consistent builds across runs.

### Build Specification (buildspec.yml)
The pipeline behavior is defined in buildspec.yml, which specifies install, build, and test phases. Dependencies are installed, and application tests are executed automatically.

📸 Screenshot: buildspec.yml snippet

### Step 4: Build Execution & Logs
During execution, CodeBuild streams logs to CloudWatch, providing visibility into each phase of the pipeline and enabling rapid debugging.

📸 Screenshot: Successful build logs

screenshots/
├── 01-overview.png
├── 02-buildspec.png
├── 03-success.png
├── 04-logs.png
└── 05-cleanup.png

### Step 5: Artifacts & Output (If Applicable)
Build artifacts are temporarily stored in S3 for validation purposes. No persistent storage is retained after pipeline validation.

### Step 6: Cost Management & Cleanup
After validating the pipeline, all AWS resources (CodeBuild project, S3 bucket, IAM role) were deleted to ensure zero ongoing cost. This repository preserves the full configuration for reproducibility.

## ❓ Cost Management
Cost ManagementAll AWS resources (CodeBuild projects, S3 artifacts, IAM roles) were deleted after validation to ensure zero ongoing cost. This repository contains the full configuration to reproduce the pipeline.

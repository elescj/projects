# ☁️ Python APP AWS CodeBuild Demo

![Graphical Summary](attachments/aws-demos.png)

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

### Step 3: Build Specification (buildspec.yml)
The pipeline behavior is defined in buildspec.yml, which specifies install, build, and test phases. Dependencies are installed, and application tests are executed automatically.
([View buildspec.yml](buildspec.yml))

### Step 4: Build Execution & Logs
During execution, CodeBuild streams logs to CloudWatch, providing visibility into each phase of the pipeline and enabling rapid debugging.

#### Build history
![Graphical Summary](attachments/builds.png)

#### Build details
![Graphical Summary](attachments/build1.png)
![Graphical Summary](attachments/build2.png)

### Step 5: Artifacts & Output
Build artifacts are temporarily stored in S3 for validation purposes. No persistent storage is retained after pipeline validation.

### Step 6: Cost Management & Cleanup
After validating the pipeline, all AWS resources (CodeBuild project, S3 bucket, IAM role) were deleted to ensure zero ongoing cost. This repository preserves the full configuration for reproducibility.

## ❓ Cost Management
Cost ManagementAll AWS resources (CodeBuild projects, S3 artifacts, IAM roles) were deleted after validation to ensure zero ongoing cost. This repository contains the full configuration to reproduce the pipeline.

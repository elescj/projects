# ☁️ Python APP AWS CodeBuild Demo

## 📂 Table of Contents
- [Overview](#-overview)
- [CI/CD Pipeline Walkthrough](#-cicd-pipeline-walkthrough)
- [Cost Management](#-cost-management)

## Overview

This project demonstrates a Python application integrated with an
AWS CodeBuild CI pipeline. The focus is on build automation, GitHub
integration, and cost-aware cloud usage.

- Integrated GitHub repository with AWS CodeBuild
- Defined build and test phases using buildspec.yml
- Builds triggered via GitHub webhook
- Artifacts stored temporarily in S3

## CI/CD Pipeline Walkthrough
screenshots/
├── 01-overview.png
├── 02-buildspec.png
├── 03-success.png
├── 04-logs.png
└── 05-cleanup.png

## Cost Management

All AWS resources (CodeBuild projects, S3 artifacts, IAM roles)
were deleted after validation to ensure zero ongoing cost.
This repository contains the full configuration to reproduce the pipeline.

# PyTorch Geometric Knowledge Graph Builder

> Serverless pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![AWS Glue](https://img.shields.io/badge/AWS-Glue-orange.svg)](https://aws.amazon.com/glue/)

## Overview

PyTorch Geometric Knowledge Graph Builder is a flexible, serverless pipeline that transforms raw RDF data from multiple heterogeneous sources into enriched knowledge graphs and constructs PyTorch Geometric `HeteroData` objects ready for Graph Neural Network (GNN) training.

The pipeline supports three execution modes to optimize for different workflows:

- **Full Pipeline**: End-to-end RDF enrichment and PyG graph construction
- **Enrichment Only**: Create reusable enriched RDF artifacts
- **PyG Construction Only**: Rapidly experiment with different PyG graph structures from existing enriched RDF

### Key Features

- **Temporal Unification**: Unified temporal entities across all data sources
- **Cross-Source Linking**: Automatic relationship discovery between heterogeneous datasets
- **PyTorch Geometric Output**: Native `HeteroData` objects with configurable node/edge types
- **Flexible Graph Construction**: Experiment with different graph structures without re-enrichment
- **Serverless Architecture**: Fully managed AWS Glue, no infrastructure to maintain
- **Experiment-Friendly**: Rapid iteration on PyG graph structures (5-10 min per experiment)
- **Scalable**: Distributed RDF processing with Apache Spark
- **Reusable Artifacts**: Enriched RDF can generate multiple PyG graphs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Raw Data Sources (S3)                                       │
│ ├── BLS Economic Data (CPI, PPI, JOLTS) - RDF               │
│ ├── SEC Filings - RDF                                       │
│ ├── Stock Market Data - RDF                                 │
│ └── Weather Data - RDF                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ AWS Glue Job: pyg-knowledge-graph-builder                   │
│                                                             │
│ Mode 1: Full Pipeline                                       │
│   Raw RDF → Enrich RDF → Build PyG HeteroData               │
│                                                             │
│ Mode 2: Enrichment Only                                     │
│   Raw RDF → Enrich RDF → Save to S3                         │
│                                                             │
│ Mode 3: PyG Only                                            │
│   Enriched RDF (S3) → Build PyG HeteroData                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Outputs (S3)                                               │
│ ├── Enriched RDF (Turtle format) - Reusable artifact       │
│ └── PyTorch Geometric HeteroData (.pt files) - GNN ready   │
└────────────────────────────────────────────────────────────┘
```
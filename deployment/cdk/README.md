# CDK Deployment — PyG Knowledge Graph Builder

AWS CDK stacks that provision the Glue job, S3 buckets, IAM roles, and SSM parameters for the PyG Knowledge Graph Builder pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ S3Stack                                                     │
│  ├── artifacts bucket  (scripts, wheels, Spark UI logs)     │
│  ├── data bucket       (raw RDF, enriched artifacts)        │
│  └── output bucket     (PyG .pt files, manifests)           │
├─────────────────────────────────────────────────────────────┤
│ IAMStack                                                    │
│  ├── Glue execution role                                    │
│  └── Studio managed policy (attach to your Studio role)     │
├─────────────────────────────────────────────────────────────┤
│ GlueStack                                                   │
│  ├── build-graph Glue job                                   │
│  └── SSM parameters (job name, buckets, defaults, version)  │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.9+
- AWS CDK CLI (`npm install -g aws-cdk`)
- AWS credentials configured
- CDK bootstrapped in target account/region (`cdk bootstrap`)

## Quick Start

```bash
cd deployment/cdk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Deploy to dev
cdk deploy --all -c environment=dev

# Deploy to prod with larger defaults
cdk deploy --all -c environment=prod \
    -c default_worker_type=G.4X \
    -c default_number_of_workers=20
```

## Configuration

All configuration is in `cdk.json` context and can be overridden via `-c` flags:

| Context Key | Default | Description |
|-------------|---------|-------------|
| `project_name` | `pyg-knowledge-graph-builder` | Resource name prefix |
| `environment` | `dev` | Deployment environment |
| `glue_version` | `4.0` | AWS Glue version |
| `default_worker_type` | `G.2X` | Default worker type (overridable per run) |
| `default_number_of_workers` | `10` | Default worker count (overridable per run) |
| `timeout_minutes` | `120` | Job timeout |
| `max_retries` | `0` | Auto-retry count |
| `max_concurrent_runs` | `3` | Parallel runs (for notebook experiments) |
| `enable_spark_ui` | `true` | Spark UI event logging |

## S3 Layout

```
artifacts-bucket/
    glue-scripts/<version>/
        build_graph.py
        glue_jobs.zip
    glue-wheels/<version>/
        rdflib-*.whl
        torch-*.whl
        torch_geometric-*.whl
    spark-ui-logs/

data-bucket/
    raw/
        bls/YYYY/MM/                        # per-source, from Lambda scrapers
        sec/YYYY/MM/
        market/YYYY/MM/
        noaa/YYYY/MM/
    enriched/
        year=YYYY/month=MM/<run_id>/        # merged across all sources
            knowledge_graph.ttl

output-bucket/
    pyg/
        year=YYYY/month=MM/<run_id>/
            hetero_data.pt
    manifests/
        year=YYYY/month=MM/
            <mode>_<timestamp>.json
```

Raw data is partitioned by source because it arrives from separate Lambda scrapers. Enrichment merges all sources into a single graph, so the source dimension collapses — only time period and run ID are preserved. PyG output and manifests mirror the same date partitions for lineage traceability.

## SSM Parameters

The CDK creates SSM parameters so that notebooks and external tooling can discover configuration without hardcoded values:

| Parameter | Example Value |
|-----------|---------------|
| `/<project>/<env>/active-version` | `1.0.0` |
| `/<project>/<env>/glue-job-name/build-graph` | `pyg-...-build-graph` |
| `/<project>/<env>/buckets/data` | `pyg-...-data-123456789` |
| `/<project>/<env>/buckets/output` | `pyg-...-output-123456789` |
| `/<project>/<env>/buckets/artifacts` | `pyg-...-artifacts-123456789` |
| `/<project>/<env>/defaults/worker-type` | `G.2X` |
| `/<project>/<env>/defaults/number-of-workers` | `10` |

## SageMaker Studio Integration

The CDK does **not** create a SageMaker Studio domain — Studio is managed separately. Instead, the IAM stack creates a managed policy (`StudioGlueInvocationPolicy`) that you attach to your existing Studio execution role:

```bash
# Find the policy ARN from CDK outputs
aws cloudformation describe-stacks \
    --stack-name pyg-knowledge-graph-builder-dev-iam \
    --query "Stacks[0].Outputs[?OutputKey=='StudioPolicyArn'].OutputValue" \
    --output text

# Attach to your Studio execution role
aws iam attach-role-policy \
    --role-name <your-studio-execution-role> \
    --policy-arn <policy-arn-from-above>
```

This policy grants:
- `glue:StartJobRun`, `glue:GetJobRun`, `glue:GetJob` scoped to project jobs
- `ssm:GetParameter*` scoped to project parameters
- `s3:GetObject` on output and data buckets
- `logs:GetLogEvents` on Glue log groups

### Invoking from Notebooks

Clone the repository into SageMaker Studio, then use `notebooks/utils/invoke_helpers.py`:

```python
from utils.invoke_helpers import GlueJobInvoker

invoker = GlueJobInvoker()  # discovers everything from SSM

# Full pipeline — enrichment + PyG construction
run_id = invoker.start_run(
    mode='full',
    time_period='2024-12',
    source_prefix='raw/',
)
result = invoker.wait_for_completion(run_id)
hetero_data = invoker.load_pyg_output(run_id)
```

All `build_graph.py` parameters are overridable, plus infrastructure settings:

```python
# Override infrastructure for a heavy run
run_id = invoker.start_run(
    mode='full',
    time_period='2024-12',
    source_prefix='raw/',
    worker_type='G.4X',       # override default G.2X
    number_of_workers=20,     # override default 10
    timeout=240,              # override default 120 min
    enable_ontology_mapping=True,
)
```

```python
# Quick PyG experiment reusing existing enriched data
run_id = invoker.start_run(
    mode='pyg_only',
    time_period='2024-12',
    enriched_rdf_key='enriched/year=2024/month=12/abc123/knowledge_graph.ttl',
    pyg_config={
        'node_types': ['cpi_Index', 'market_PriceObservation'],
        'feature_config': {'normalize': True},
    },
)
result = invoker.wait_for_completion(run_id)
hetero_data = invoker.load_pyg_output(run_id)
```

Output paths are auto-generated with date partitions and a shared run ID:

```
enriched/year=2024/month=12/a1b2c3d4/knowledge_graph.ttl
pyg/year=2024/month=12/a1b2c3d4/hetero_data.pt
```

### Listing and Loading Artifacts

```python
# List all PyG outputs for December 2024
invoker.list_pyg_outputs(year='2024', month='12')

# List all enriched artifacts for 2024
invoker.list_enriched_artifacts(year='2024')

# Load a specific output directly
hetero_data = invoker.load_pyg_output(
    pyg_output_key='pyg/year=2024/month=12/a1b2c3d4/hetero_data.pt'
)
```

## Blue/Green Deployment

The CDK creates stable infrastructure. An external CodeDeploy pipeline (separate repository) performs blue/green deployments by updating the Glue job to point to new versioned artifacts in S3.

### What This CDK Creates

```
SSM Parameters:
  /<project>/<env>/active-version              → "1.0.0"
  /<project>/<env>/glue-job-name/build-graph   → "pyg-...-build-graph"

S3 (versioned artifacts):
  s3://artifacts/glue-scripts/1.0.0/build_graph.py
  s3://artifacts/glue-scripts/1.0.0/glue_jobs.zip
  s3://artifacts/glue-wheels/1.0.0/*.whl

Glue Job (stable name, mutable script location):
  pyg-knowledge-graph-builder-dev-build-graph
    script_location           → s3://artifacts/glue-scripts/<version>/build_graph.py
    --extra-py-files          → s3://artifacts/glue-scripts/<version>/glue_jobs.zip
    --additional-python-modules → s3://artifacts/glue-wheels/<version>/*.whl
```

### What CodeDeploy Does

```
1. CI/CD uploads new artifacts to S3 under new version prefix
2. Pre-hook: validate artifacts exist, optionally run test job
3. Cutover: UpdateJob API → point to new version prefix
4. Post-hook: validate job succeeds with new version
5. Rollback (if post-hook fails): UpdateJob API → revert to previous version
```

### Example UpdateJob Call

```python
# Lives in the CodeDeploy repo, not here
import boto3

glue = boto3.client('glue')
ssm = boto3.client('ssm')

new_version = '1.1.0'
job_name = 'pyg-knowledge-graph-builder-dev-build-graph'
bucket = 'pyg-knowledge-graph-builder-dev-artifacts-123456789'

glue.update_job(
    JobName=job_name,
    JobUpdate={
        'Command': {
            'Name': 'glueetl',
            'PythonVersion': '3',
            'ScriptLocation': f's3://{bucket}/glue-scripts/{new_version}/build_graph.py',
        },
        'DefaultArguments': {
            '--extra-py-files': f's3://{bucket}/glue-scripts/{new_version}/glue_jobs.zip',
            '--additional-python-modules': ','.join([
                f's3://{bucket}/glue-wheels/{new_version}/rdflib-7.1.3-py3-none-any.whl',
                f's3://{bucket}/glue-wheels/{new_version}/torch-2.3.0+cpu-cp310-cp310-linux_x86_64.whl',
                f's3://{bucket}/glue-wheels/{new_version}/torch_geometric-2.5.3-py3-none-any.whl',
            ]),
            '--deployment_version': new_version,
            # ... preserve all other default arguments
        },
    },
)

ssm.put_parameter(
    Name='/pyg-knowledge-graph-builder/dev/active-version',
    Value=new_version,
    Overwrite=True,
)
```

## Useful Commands

```bash
# Synthesize CloudFormation templates
cdk synth

# Diff against deployed stacks
cdk diff

# Deploy all stacks
cdk deploy --all

# Deploy specific stack
cdk deploy pyg-knowledge-graph-builder-dev-glue

# Destroy all stacks (careful — RETAIN buckets won't be deleted)
cdk destroy --all

# List stacks
cdk list
```
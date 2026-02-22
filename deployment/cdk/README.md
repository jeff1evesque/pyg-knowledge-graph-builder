## Blue/Green Deployment Integration Notes

The CDK stack is designed so that an **external CodeDeploy pipeline** (in a separate repository) can perform blue/green deployments by:

### What This CDK Creates (Stable Infrastructure)

```
SSM Parameters:
  /<project>/<env>/active-version        → "1.0.0"
  /<project>/<env>/glue-job-name/build-graph → "pyg-...-build-graph"

S3 Layout (versioned artifacts):
  s3://artifacts/glue-scripts/1.0.0/build_graph.py
  s3://artifacts/glue-scripts/1.0.0/glue_jobs.zip
  s3://artifacts/glue-wheels/1.0.0/*.whl
  s3://artifacts/glue-scripts/1.1.0/build_graph.py    ← new version
  s3://artifacts/glue-scripts/1.1.0/glue_jobs.zip
  s3://artifacts/glue-wheels/1.1.0/*.whl

Glue Job:
  pyg-knowledge-graph-builder-dev-build-graph
    script_location → s3://artifacts/glue-scripts/<active-version>/build_graph.py
    extra-py-files  → s3://artifacts/glue-scripts/<active-version>/glue_jobs.zip
    additional-python-modules → s3://artifacts/glue-wheels/<active-version>/*.whl
```

### What CodeDeploy Does (Separate Repo)

```
1. CI/CD uploads new artifacts to S3 under new version prefix
2. CodeDeploy pre-hook:
   - Validates new artifacts exist in S3
   - Optionally runs a test Glue job with new version
3. CodeDeploy cutover:
   - Calls UpdateJob API to point script_location, extra-py-files,
     and additional-python-modules to new version prefix
   - Updates SSM /<project>/<env>/active-version → "1.1.0"
4. CodeDeploy post-hook:
   - Validates job runs successfully with new version
5. CodeDeploy rollback (if post-hook fails):
   - Calls UpdateJob API to revert to previous version prefix
   - Reverts SSM parameter to previous version
```

### Example CodeDeploy UpdateJob Call

```python
# This would live in the CodeDeploy repo, not here
import boto3

glue_client = boto3.client('glue')
ssm_client = boto3.client('ssm')

new_version = "1.1.0"
job_name = "pyg-knowledge-graph-builder-dev-build-graph"
artifacts_bucket = "pyg-knowledge-graph-builder-dev-artifacts-123456789"

glue_client.update_job(
    JobName=job_name,
    JobUpdate={
        "Command": {
            "Name": "glueetl",
            "PythonVersion": "3",
            "ScriptLocation": (
                f"s3://{artifacts_bucket}/glue-scripts/{new_version}/build_graph.py"
            ),
        },
        "DefaultArguments": {
            "--extra-py-files": (
                f"s3://{artifacts_bucket}/glue-scripts/{new_version}/glue_jobs.zip"
            ),
            "--additional-python-modules": ",".join([
                f"s3://{artifacts_bucket}/glue-wheels/{new_version}/rdflib-7.1.3-py3-none-any.whl",
                f"s3://{artifacts_bucket}/glue-wheels/{new_version}/torch-2.3.0+cpu-cp310-cp310-linux_x86_64.whl",
                f"s3://{artifacts_bucket}/glue-wheels/{new_version}/torch_geometric-2.5.3-py3-none-any.whl",
            ]),
            "--deployment_version": new_version,
            # ... preserve all other default arguments
        },
    },
)

ssm_client.put_parameter(
    Name=f"/pyg-knowledge-graph-builder/dev/active-version",
    Value=new_version,
    Overwrite=True,
)
```
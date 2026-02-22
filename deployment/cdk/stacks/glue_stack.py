from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_glue as glue,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ssm as ssm,
)


class GlueStack(Stack):
    """AWS Glue resources for the PyG Knowledge Graph Builder.

    Creates:
        - Glue Job for build_graph.py (supports full, enrichment-only, pyg-only modes)
        - SSM Parameters for blue/green deployment version tracking
        - CloudWatch log group configuration

    Blue/Green Deployment Support:
        - Script and wheel paths are versioned in S3:
            s3://<artifacts>/glue-scripts/<version>/build_graph.py
            s3://<artifacts>/glue-wheels/<version>/*.whl
        - SSM parameters track active version for external CodeDeploy integration
        - Job name includes no version suffix — CodeDeploy updates the job in-place
          by updating the script location and extra-py-files to the new version
    """

    # Glue job parameter keys used by build_graph.py
    JOB_PARAM_MODE = "--mode"
    JOB_PARAM_INPUT_PATH = "--input_path"
    JOB_PARAM_ENRICHED_PATH = "--enriched_path"
    JOB_PARAM_OUTPUT_PATH = "--output_path"
    JOB_PARAM_PYG_CONFIG = "--pyg_config"
    JOB_PARAM_TIME_PERIOD = "--time_period"

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        environment: str,
        artifacts_bucket: s3.IBucket,
        data_bucket: s3.IBucket,
        output_bucket: s3.IBucket,
        glue_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._project_name = project_name
        self._environment = environment
        self._artifacts_bucket = artifacts_bucket
        self._data_bucket = data_bucket
        self._output_bucket = output_bucket
        self._glue_role = glue_role

        # Read Glue configuration from CDK context
        self._glue_version = self.node.try_get_context("glue_version") or "4.0"
        self._python_version = self.node.try_get_context("python_version") or "3"
        self._worker_type = self.node.try_get_context("worker_type") or "G.2X"
        self._number_of_workers = self.node.try_get_context("number_of_workers") or 10
        self._timeout_minutes = self.node.try_get_context("timeout_minutes") or 120
        self._max_retries = self.node.try_get_context("max_retries") or 0
        self._max_concurrent_runs = (
            self.node.try_get_context("max_concurrent_runs") or 1
        )
        self._enable_spark_ui = self.node.try_get_context("enable_spark_ui") or True
        self._spark_ui_log_prefix = (
            self.node.try_get_context("spark_ui_log_prefix") or "spark-ui-logs/"
        )

        # SSM parameters for blue/green version tracking
        self._version_param = self._create_version_parameter()

        # Main Glue job
        self._build_graph_job = self._create_build_graph_job()

        self._add_outputs()

    def _create_version_parameter(self) -> ssm.StringParameter:
        """SSM parameter to track the active deployment version.

        CodeDeploy (in a separate repo) reads this parameter to determine
        the current active version and updates it during blue/green cutover.

        Value format: semantic version or git SHA, e.g. "1.0.0" or "abc1234"
        This version maps to the S3 prefix:
            s3://<artifacts>/glue-scripts/<version>/
            s3://<artifacts>/glue-wheels/<version>/
        """
        return ssm.StringParameter(
            self,
            "ActiveVersionParam",
            parameter_name=(
                f"/{self._project_name}/{self._environment}/active-version"
            ),
            string_value="initial",
            description=(
                f"Active deployment version for {self._project_name} "
                f"Glue jobs in {self._environment}. Updated by CodeDeploy "
                f"during blue/green deployments."
            ),
            tier=ssm.ParameterTier.STANDARD,
        )

    def _get_script_location(self) -> str:
        """S3 path to the main Glue script.

        Uses 'initial' as the default version. During deployment, the CI/CD
        pipeline uploads scripts to s3://<bucket>/glue-scripts/<version>/
        and CodeDeploy updates the job to point to the new version.
        """
        return (
            f"s3://{self._artifacts_bucket.bucket_name}"
            f"/glue-scripts/initial/build_graph.py"
        )

    def _get_extra_py_files(self) -> str:
        """S3 path to the zipped glue_jobs package.

        The package is zipped as glue_jobs.zip containing:
            glue_jobs/
                __init__.py
                enrichment/
                pyg_builder/
                utils/
        """
        return (
            f"s3://{self._artifacts_bucket.bucket_name}"
            f"/glue-scripts/initial/glue_jobs.zip"
        )

    def _get_extra_jars_or_wheels(self) -> str:
        """S3 paths to pre-built wheel files for dependencies.

        Wheels are uploaded per version:
            s3://<bucket>/glue-wheels/<version>/rdflib-x.y.z-py3-none-any.whl
            s3://<bucket>/glue-wheels/<version>/torch-x.y.z+cpu-....whl
            s3://<bucket>/glue-wheels/<version>/torch_geometric-x.y.z-....whl

        Returns comma-separated S3 paths.
        """
        base = (
            f"s3://{self._artifacts_bucket.bucket_name}/glue-wheels/initial"
        )
        return ",".join(
            [
                f"{base}/rdflib-7.1.3-py3-none-any.whl",
                f"{base}/torch-2.3.0+cpu-cp310-cp310-linux_x86_64.whl",
                f"{base}/torch_geometric-2.5.3-py3-none-any.whl",
            ]
        )

    def _build_default_arguments(self) -> dict:
        """Construct the default arguments dict for the Glue job.

        These are defaults that can be overridden at job run time.
        The --mode parameter controls pipeline behavior:
            'full'       — enrichment + PyG construction
            'enrichment' — enrichment only, save Parquet
            'pyg'        — PyG construction from existing enriched Parquet
        """
        context_defaults = self.node.try_get_context("default_arguments") or {}

        default_args = {
            # Pipeline mode
            self.JOB_PARAM_MODE: "full",
            # S3 paths (defaults, overridable per run)
            self.JOB_PARAM_INPUT_PATH: (
                f"s3://{self._data_bucket.bucket_name}/raw/"
            ),
            self.JOB_PARAM_ENRICHED_PATH: (
                f"s3://{self._data_bucket.bucket_name}/enriched/"
            ),
            self.JOB_PARAM_OUTPUT_PATH: (
                f"s3://{self._output_bucket.bucket_name}/pyg/"
            ),
            # Extra Python files (zipped package)
            "--extra-py-files": self._get_extra_py_files(),
            # Pre-built wheels for rdflib, torch, torch-geometric
            "--additional-python-modules": self._get_extra_jars_or_wheels(),
            # Glue continuous logging
            "--enable-continuous-cloudwatch-log": "true",
            "--continuous-log-logGroup": (
                f"/aws-glue/{self._project_name}/{self._environment}"
            ),
            "--enable-continuous-log-filter": "true",
            # Glue metrics
            "--enable-metrics": "true",
            # Job bookmarks disabled (stateless pipeline)
            "--job-bookmark-option": "job-bookmark-disable",
            # Spark configuration
            "--conf": (
                "spark.serializer=org.apache.spark.serializer.KryoSerializer"
                " --conf spark.sql.adaptive.enabled=true"
                " --conf spark.sql.adaptive.coalescePartitions.enabled=true"
            ),
            # Deployment version tracking (readable by the job if needed)
            "--deployment_version": "initial",
        }

        # Merge context overrides (context values take precedence)
        default_args.update(context_defaults)

        # Spark UI configuration
        if self._enable_spark_ui:
            default_args["--enable-spark-ui"] = "true"
            default_args["--spark-event-logs-path"] = (
                f"s3://{self._artifacts_bucket.bucket_name}"
                f"/{self._spark_ui_log_prefix}"
            )

        return default_args

    def _create_build_graph_job(self) -> glue.CfnJob:
        """Create the main Glue job for the knowledge graph pipeline.

        The job name is stable (no version suffix) to support in-place
        updates during blue/green deployment. CodeDeploy updates the
        script location and extra-py-files to point to the new version.
        """
        job_name = f"{self._project_name}-{self._environment}-build-graph"

        job = glue.CfnJob(
            self,
            "BuildGraphJob",
            name=job_name,
            description=(
                "Builds enriched RDF knowledge graphs and constructs "
                "PyTorch Geometric HeteroData objects from multi-source "
                "economic, financial, market, and environmental data. "
                "Supports full, enrichment-only, and PyG-only modes."
            ),
            role=self._glue_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="glueetl",
                python_version=self._python_version,
                script_location=self._get_script_location(),
            ),
            glue_version=self._glue_version,
            worker_type=self._worker_type,
            number_of_workers=self._number_of_workers,
            timeout=self._timeout_minutes,
            max_retries=self._max_retries,
            execution_property=glue.CfnJob.ExecutionPropertyProperty(
                max_concurrent_runs=self._max_concurrent_runs,
            ),
            default_arguments=self._build_default_arguments(),
            tags={
                "Project": self._project_name,
                "Environment": self._environment,
                "Component": "build-graph",
            },
        )

        # Store job name in SSM for CodeDeploy integration
        ssm.StringParameter(
            self,
            "BuildGraphJobNameParam",
            parameter_name=(
                f"/{self._project_name}/{self._environment}"
                f"/glue-job-name/build-graph"
            ),
            string_value=job_name,
            description=(
                f"Glue job name for build-graph in {self._environment}. "
                f"Used by CodeDeploy for blue/green updates."
            ),
        )

        return job

    def _add_outputs(self) -> None:
        cdk.CfnOutput(
            self,
            "BuildGraphJobName",
            value=self._build_graph_job.name,
            export_name=(
                f"{self._project_name}-{self._environment}-build-graph-job-name"
            ),
        )

        cdk.CfnOutput(
            self,
            "ActiveVersionParamName",
            value=self._version_param.parameter_name,
            export_name=(
                f"{self._project_name}-{self._environment}-active-version-param"
            ),
        )

        cdk.CfnOutput(
            self,
            "ScriptBasePath",
            value=(
                f"s3://{self._artifacts_bucket.bucket_name}/glue-scripts/"
            ),
            description=(
                "Base S3 path for Glue scripts. "
                "Append <version>/build_graph.py for specific version."
            ),
            export_name=(
                f"{self._project_name}-{self._environment}-script-base-path"
            ),
        )

        cdk.CfnOutput(
            self,
            "WheelsBasePath",
            value=(
                f"s3://{self._artifacts_bucket.bucket_name}/glue-wheels/"
            ),
            description=(
                "Base S3 path for dependency wheels. "
                "Append <version>/*.whl for specific version."
            ),
            export_name=(
                f"{self._project_name}-{self._environment}-wheels-base-path"
            ),
        )

    @property
    def build_graph_job(self) -> glue.CfnJob:
        return self._build_graph_job

    @property
    def version_parameter(self) -> ssm.StringParameter:
        return self._version_param
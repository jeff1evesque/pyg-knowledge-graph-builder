from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    aws_s3 as s3,
)


class S3Stack(Stack):
    """S3 buckets for Glue job artifacts, raw data, and outputs.

    Bucket layout:
        artifacts_bucket/
            glue-scripts/
                <version>/
                    build_graph.py
                    glue_jobs.zip          # zipped package
            glue-wheels/
                <version>/
                    rdflib-x.y.z.whl
                    torch-x.y.z.whl
                    torch_geometric-x.y.z.whl
            spark-ui-logs/

        data_bucket/
            raw/
                bls/ sec/ market/ noaa/    # raw RDF N-Triples
            enriched/
                <run_id>/                  # enriched Parquet

        output_bucket/
            pyg/
                <run_id>/                  # HeteroData .pt files
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        environment: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._project_name = project_name
        self._environment = environment

        self._artifacts_bucket = self._create_artifacts_bucket()
        self._data_bucket = self._create_data_bucket()
        self._output_bucket = self._create_output_bucket()

        self._add_outputs()

    def _create_artifacts_bucket(self) -> s3.Bucket:
        """Bucket for Glue scripts, wheels, and Spark UI logs."""
        return s3.Bucket(
            self,
            "ArtifactsBucket",
            bucket_name=f"{self._project_name}-{self._environment}-artifacts-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="cleanup-old-versions",
                    noncurrent_version_expiration=Duration.days(90),
                    enabled=True,
                ),
                s3.LifecycleRule(
                    id="cleanup-spark-ui-logs",
                    prefix="spark-ui-logs/",
                    expiration=Duration.days(30),
                    enabled=True,
                ),
            ],
        )

    def _create_data_bucket(self) -> s3.Bucket:
        """Bucket for raw RDF data and enriched Parquet artifacts."""
        return s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"{self._project_name}-{self._environment}-data-{self.account}",
            versioned=False,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="archive-old-enriched",
                    prefix="enriched/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(30),
                        ),
                    ],
                    enabled=True,
                ),
            ],
        )

    def _create_output_bucket(self) -> s3.Bucket:
        """Bucket for PyG HeteroData .pt output files."""
        return s3.Bucket(
            self,
            "OutputBucket",
            bucket_name=f"{self._project_name}-{self._environment}-output-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="cleanup-old-versions",
                    noncurrent_version_expiration=Duration.days(60),
                    enabled=True,
                ),
            ],
        )

    def _add_outputs(self) -> None:
        cdk.CfnOutput(
            self,
            "ArtifactsBucketName",
            value=self._artifacts_bucket.bucket_name,
            export_name=f"{self._project_name}-{self._environment}-artifacts-bucket",
        )
        cdk.CfnOutput(
            self,
            "DataBucketName",
            value=self._data_bucket.bucket_name,
            export_name=f"{self._project_name}-{self._environment}-data-bucket",
        )
        cdk.CfnOutput(
            self,
            "OutputBucketName",
            value=self._output_bucket.bucket_name,
            export_name=f"{self._project_name}-{self._environment}-output-bucket",
        )

    @property
    def artifacts_bucket(self) -> s3.Bucket:
        return self._artifacts_bucket

    @property
    def data_bucket(self) -> s3.Bucket:
        return self._data_bucket

    @property
    def output_bucket(self) -> s3.Bucket:
        return self._output_bucket
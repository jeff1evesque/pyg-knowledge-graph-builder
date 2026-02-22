from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
)


class IAMStack(Stack):
    """IAM roles and policies for Glue jobs.

    Creates a Glue service role with:
        - Glue service permissions
        - S3 read/write to artifacts, data, and output buckets
        - CloudWatch Logs for continuous logging
        - CloudWatch Metrics for job metrics
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        project_name: str,
        environment: str,
        artifacts_bucket: s3.IBucket,
        data_bucket: s3.IBucket,
        output_bucket: s3.IBucket,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._project_name = project_name
        self._environment = environment
        self._artifacts_bucket = artifacts_bucket
        self._data_bucket = data_bucket
        self._output_bucket = output_bucket

        self._glue_role = self._create_glue_role()

        self._add_outputs()

    def _create_glue_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "GlueJobRole",
            role_name=f"{self._project_name}-{self._environment}-glue-role",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            description=(
                f"Execution role for {self._project_name} Glue jobs "
                f"in {self._environment}"
            ),
        )

        # AWS managed policy for basic Glue service permissions
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSGlueServiceRole"
            )
        )

        # S3 permissions scoped to project buckets
        self._add_s3_permissions(role)

        # CloudWatch permissions for continuous logging and metrics
        self._add_cloudwatch_permissions(role)

        return role

    def _add_s3_permissions(self, role: iam.Role) -> None:
        """Grant S3 access scoped to project buckets only."""

        # Artifacts bucket: read scripts and wheels, write Spark UI logs
        self._artifacts_bucket.grant_read(role)
        self._artifacts_bucket.grant_write(role, "spark-ui-logs/*")

        # Data bucket: read raw RDF, read/write enriched Parquet
        self._data_bucket.grant_read(role, "raw/*")
        self._data_bucket.grant_read_write(role, "enriched/*")

        # Output bucket: write PyG .pt files
        self._output_bucket.grant_write(role, "pyg/*")
        # Also allow read for PyG-only mode (reads existing enriched data)
        self._output_bucket.grant_read(role, "pyg/*")

    def _add_cloudwatch_permissions(self, role: iam.Role) -> None:
        """CloudWatch permissions for Glue continuous logging and metrics."""
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                    "logs:GetLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws-glue/*",
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchMetricsAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "cloudwatch:namespace": "Glue",
                    }
                },
            )
        )

    def _add_outputs(self) -> None:
        cdk.CfnOutput(
            self,
            "GlueRoleArn",
            value=self._glue_role.role_arn,
            export_name=f"{self._project_name}-{self._environment}-glue-role-arn",
        )
        cdk.CfnOutput(
            self,
            "GlueRoleName",
            value=self._glue_role.role_name,
            export_name=f"{self._project_name}-{self._environment}-glue-role-name",
        )

    @property
    def glue_role(self) -> iam.Role:
        return self._glue_role
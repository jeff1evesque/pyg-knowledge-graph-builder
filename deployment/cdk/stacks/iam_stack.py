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
        - S3 read/write scoped to project buckets
        - CloudWatch Logs for continuous logging
        - CloudWatch Metrics for job metrics

    Note: The SageMaker Studio execution role is managed outside this
    stack (typically by the Studio domain/profile setup). That role
    needs:
        - glue:StartJobRun, glue:GetJobRun, glue:GetJob on project jobs
        - s3:GetObject on output bucket (to load .pt files)
        - ssm:GetParameter on project SSM parameters
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
        self._studio_policy = self._create_studio_managed_policy()

        self._add_outputs()

    # ------------------------------------------------------------------
    # Glue role
    # ------------------------------------------------------------------
    def _create_glue_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "GlueJobRole",
            role_name=(
                f"{self._project_name}-{self._environment}-glue-role"
            ),
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            description=(
                f"Execution role for {self._project_name} Glue jobs "
                f"in {self._environment}"
            ),
        )

        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSGlueServiceRole"
            )
        )

        self._add_glue_s3_permissions(role)
        self._add_cloudwatch_permissions(role)

        return role

    def _add_glue_s3_permissions(self, role: iam.Role) -> None:
        """Grant S3 access scoped to project buckets."""
        self._artifacts_bucket.grant_read(role)
        self._artifacts_bucket.grant_write(role, "spark-ui-logs/*")

        self._data_bucket.grant_read(role, "raw/*")
        self._data_bucket.grant_read_write(role, "enriched/*")

        self._output_bucket.grant_read_write(role, "pyg/*")
        self._output_bucket.grant_read_write(role, "manifests/*")

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
                    (
                        f"arn:aws:logs:{self.region}:{self.account}"
                        f":log-group:/aws-glue/*"
                    ),
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

    # ------------------------------------------------------------------
    # SageMaker Studio managed policy
    # ------------------------------------------------------------------
    def _create_studio_managed_policy(self) -> iam.ManagedPolicy:
        """Managed policy to attach to a SageMaker Studio execution role.

        This policy grants the minimum permissions a Studio notebook
        needs to invoke and monitor this project's Glue jobs, read
        outputs from S3, and discover configuration from SSM.

        Attach this policy to your Studio domain/profile execution role:
            - Via the AWS console, or
            - Via a separate CDK stack that manages Studio, or
            - Via CLI:
              aws iam attach-role-policy \\
                --role-name <studio-execution-role> \\
                --policy-arn <this policy ARN>
        """
        policy = iam.ManagedPolicy(
            self,
            "StudioGlueInvocationPolicy",
            managed_policy_name=(
                f"{self._project_name}-{self._environment}"
                f"-studio-glue-policy"
            ),
            description=(
                f"Allows SageMaker Studio to invoke and monitor "
                f"{self._project_name} Glue jobs in {self._environment}. "
                f"Attach to your Studio execution role."
            ),
            statements=[
                # Glue job invocation — scoped to project jobs
                iam.PolicyStatement(
                    sid="GlueJobInvocation",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "glue:StartJobRun",
                        "glue:GetJobRun",
                        "glue:GetJobRuns",
                        "glue:BatchStopJobRun",
                        "glue:GetJob",
                    ],
                    resources=[
                        (
                            f"arn:aws:glue:{self.region}:{self.account}"
                            f":job/{self._project_name}"
                            f"-{self._environment}-*"
                        ),
                    ],
                ),
                # SSM parameter read — discover job names, buckets, defaults
                iam.PolicyStatement(
                    sid="SSMParameterRead",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ssm:GetParameter",
                        "ssm:GetParameters",
                        "ssm:GetParametersByPath",
                    ],
                    resources=[
                        (
                            f"arn:aws:ssm:{self.region}:{self.account}"
                            f":parameter/{self._project_name}/"
                            f"{self._environment}/*"
                        ),
                    ],
                ),
                # CloudWatch logs read — tail Glue job output in notebook
                iam.PolicyStatement(
                    sid="CloudWatchLogsRead",
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "logs:GetLogEvents",
                        "logs:DescribeLogGroups",
                        "logs:DescribeLogStreams",
                        "logs:FilterLogEvents",
                    ],
                    resources=[
                        (
                            f"arn:aws:logs:{self.region}:{self.account}"
                            f":log-group:/aws-glue/*"
                        ),
                    ],
                ),
            ],
        )

        # S3 read on output bucket (load .pt files back into notebook)
        self._output_bucket.grant_read(policy)

        # S3 read on data bucket (inspect enriched artifacts)
        self._data_bucket.grant_read(policy)

        return policy

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _add_outputs(self) -> None:
        cdk.CfnOutput(
            self,
            "GlueRoleArn",
            value=self._glue_role.role_arn,
            export_name=(
                f"{self._project_name}-{self._environment}-glue-role-arn"
            ),
        )
        cdk.CfnOutput(
            self,
            "GlueRoleName",
            value=self._glue_role.role_name,
            export_name=(
                f"{self._project_name}-{self._environment}-glue-role-name"
            ),
        )
        cdk.CfnOutput(
            self,
            "StudioPolicyArn",
            value=self._studio_policy.managed_policy_arn,
            description=(
                "Attach this managed policy to your SageMaker Studio "
                "execution role to enable Glue job invocation from "
                "notebooks."
            ),
            export_name=(
                f"{self._project_name}-{self._environment}"
                f"-studio-policy-arn"
            ),
        )

    @property
    def glue_role(self) -> iam.Role:
        return self._glue_role

    @property
    def studio_policy(self) -> iam.ManagedPolicy:
        return self._studio_policy
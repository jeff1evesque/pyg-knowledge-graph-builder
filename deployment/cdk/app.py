#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.s3_stack import S3Stack
from stacks.iam_stack import IAMStack
from stacks.glue_stack import GlueStack


def main():
    app = cdk.App()

    project_name = app.node.try_get_context("project_name")
    environment = app.node.try_get_context("environment")
    tags = app.node.try_get_context("tags") or {}

    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    )

    stack_prefix = f"{project_name}-{environment}"

    # S3 buckets for artifacts, data, and outputs
    s3_stack = S3Stack(
        app,
        f"{stack_prefix}-s3",
        project_name=project_name,
        environment=environment,
        env=env,
    )

    # IAM roles and policies for Glue
    iam_stack = IAMStack(
        app,
        f"{stack_prefix}-iam",
        project_name=project_name,
        environment=environment,
        artifacts_bucket=s3_stack.artifacts_bucket,
        data_bucket=s3_stack.data_bucket,
        output_bucket=s3_stack.output_bucket,
        env=env,
    )

    # Glue jobs, connections, and catalog
    glue_stack = GlueStack(
        app,
        f"{stack_prefix}-glue",
        project_name=project_name,
        environment=environment,
        artifacts_bucket=s3_stack.artifacts_bucket,
        data_bucket=s3_stack.data_bucket,
        output_bucket=s3_stack.output_bucket,
        glue_role=iam_stack.glue_role,
        env=env,
    )

    glue_stack.add_dependency(s3_stack)
    glue_stack.add_dependency(iam_stack)

    # Apply tags to all stacks
    for key, value in tags.items():
        cdk.Tags.of(app).add(key, value)

    cdk.Tags.of(app).add("Environment", environment)

    app.synth()


if __name__ == "__main__":
    main()
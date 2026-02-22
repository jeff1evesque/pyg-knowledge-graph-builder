"""
Helper functions for invoking Glue jobs from notebooks.

Reads configuration from SSM parameters (populated by CDK) so that
notebooks contain zero hardcoded bucket names, job names, or paths.

All S3 output paths use consistent date partitioning:
    enriched: s3://<data>/enriched/year=YYYY/month=MM/<run_id>/
                  (partitioned Parquet directory)
    pyg:      s3://<output>/pyg/year=YYYY/month=MM/<run_id>/hetero_data.pt
    manifest: s3://<output>/manifests/year=YYYY/month=MM/<mode>_<timestamp>.json

Usage:
    from utils.invoke_helpers import GlueJobInvoker

    invoker = GlueJobInvoker()

    # Full pipeline for December 2024
    run_id = invoker.start_run(
        mode='full',
        time_period='2024-12',
        source_prefix='raw/',
        worker_type='G.4X',
        number_of_workers=20,
    )
    result = invoker.wait_for_completion(run_id)
    hetero_data = invoker.load_pyg_output(run_id)

    # Quick PyG experiment reusing existing enriched Parquet
    run_id = invoker.start_run(
        mode='pyg_only',
        time_period='2024-12',
        enriched_prefix='enriched/year=2024/month=12/abc123/',
        pyg_config={
            'node_types': ['cpi_Index', 'market_PriceObservation'],
            'feature_config': {'normalize': True},
        },
    )
"""
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

import boto3


class GlueJobInvoker:
    """Discovers Glue job configuration from SSM and invokes runs.

    All bucket names, job names, and defaults are read from SSM
    parameters created by the CDK stack. No hardcoded values.

    Args:
        project_name: Project identifier.
            Default: 'pyg-knowledge-graph-builder'
        environment: Deployment environment. Default: 'dev'
        region: AWS region. Default: boto3 session default
    """

    def __init__(
        self,
        project_name: str = "pyg-knowledge-graph-builder",
        environment: str = "dev",
        region: Optional[str] = None,
    ):
        self._project_name = project_name
        self._environment = environment
        self._ssm_prefix = f"/{project_name}/{environment}"

        session = boto3.Session(region_name=region)
        self._ssm = session.client("ssm")
        self._glue = session.client("glue")
        self._s3 = session.client("s3")

        self._config = self._discover_config()

    def _discover_config(self) -> Dict[str, str]:
        """Read all project SSM parameters."""
        config = {}
        paginator = self._ssm.get_paginator("get_parameters_by_path")

        for page in paginator.paginate(
            Path=self._ssm_prefix,
            Recursive=True,
            WithDecryption=False,
        ):
            for param in page.get("Parameters", []):
                key = param["Name"].replace(
                    self._ssm_prefix + "/", ""
                )
                config[key] = param["Value"]

        if not config:
            raise RuntimeError(
                f"No SSM parameters found under {self._ssm_prefix}. "
                f"Has the CDK stack been deployed?"
            )

        return config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def job_name(self) -> str:
        return self._config.get(
            "glue-job-name/build-graph",
            f"{self._project_name}-{self._environment}-build-graph",
        )

    @property
    def output_bucket(self) -> str:
        return self._config["buckets/output"]

    @property
    def data_bucket(self) -> str:
        return self._config["buckets/data"]

    @property
    def default_worker_type(self) -> str:
        return self._config.get("defaults/worker-type", "G.2X")

    @property
    def default_number_of_workers(self) -> int:
        return int(
            self._config.get("defaults/number-of-workers", "10")
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_time_period(
        time_period: Optional[str] = None,
    ) -> tuple:
        """Parse 'YYYY-MM' into (year, month) strings.

        Returns current year/month if time_period is None.
        """
        if time_period:
            parts = time_period.split("-")
            year = parts[0]
            month = parts[1] if len(parts) > 1 else "01"
        else:
            now = datetime.utcnow()
            year = str(now.year)
            month = f"{now.month:02d}"
        return year, month

    def build_enriched_prefix(
        self,
        time_period: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Build a date-partitioned S3 prefix for enriched Parquet output.

        Format:
            enriched/year=YYYY/month=MM/<run_id>/

        This is a directory prefix — Spark writes multiple Parquet part
        files underneath it. Pass this prefix to the Glue job via
        --enriched_prefix.

        Args:
            time_period: 'YYYY-MM' string. Defaults to current month.
            run_id: Unique run identifier. Defaults to short UUID.

        Returns:
            S3 prefix string (trailing slash included)
        """
        year, month = self._parse_time_period(time_period)
        if not run_id:
            run_id = str(uuid.uuid4())[:8]
        return f"enriched/year={year}/month={month}/{run_id}/"

    def build_pyg_output_key(
        self,
        time_period: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Build a date-partitioned S3 key for PyG output.

        Format:
            pyg/year=YYYY/month=MM/<run_id>/hetero_data.pt

        Args:
            time_period: 'YYYY-MM' string. Defaults to current month.
            run_id: Unique run identifier. Defaults to short UUID.

        Returns:
            S3 key string
        """
        year, month = self._parse_time_period(time_period)
        if not run_id:
            run_id = str(uuid.uuid4())[:8]
        return (
            f"pyg/year={year}/month={month}/{run_id}/hetero_data.pt"
        )

    # ------------------------------------------------------------------
    # Job invocation
    # ------------------------------------------------------------------
    def start_run(
        self,
        mode: str = "full",
        time_period: Optional[str] = None,
        source_bucket: Optional[str] = None,
        source_prefix: Optional[str] = None,
        output_bucket: Optional[str] = None,
        data_bucket: Optional[str] = None,
        enriched_prefix: Optional[str] = None,
        pyg_output_key: Optional[str] = None,
        pyg_config: Optional[Dict[str, Any]] = None,
        enable_ontology_mapping: bool = False,
        rdf_format: Optional[str] = None,
        worker_type: Optional[str] = None,
        number_of_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Start a Glue job run with overrides.

        Only parameters you specify are overridden; everything else
        uses the job defaults defined in CDK.

        For 'full' mode, both enriched_prefix and pyg_output_key
        are auto-generated with matching date partitions and a shared
        run_id if not explicitly provided.

        Args:
            mode: 'full', 'enrichment_only', or 'pyg_only'
            time_period: 'YYYY-MM' label
            source_bucket: Override S3 bucket for raw RDF
            source_prefix: S3 prefix for raw RDF
            output_bucket: Override S3 bucket for PyG outputs
            data_bucket: Override S3 bucket for enriched Parquet
            enriched_prefix: S3 prefix for enriched Parquet directory
            pyg_output_key: S3 key for PyG output
            pyg_config: PyG construction config dict
            enable_ontology_mapping: Enable ontology mapping step
            rdf_format: Raw RDF serialization format
            worker_type: Override worker type (e.g., 'G.4X')
            number_of_workers: Override number of workers
            timeout: Override timeout in minutes

        Returns:
            JobRunId string
        """
        # Generate a shared run_id for consistent path pairing
        shared_run_id = str(uuid.uuid4())[:8]

        # Build Arguments — only include overrides
        arguments = {"--mode": mode}

        if time_period:
            arguments["--time_period"] = time_period
        if source_bucket:
            arguments["--source_bucket"] = source_bucket
        if source_prefix:
            arguments["--source_prefix"] = source_prefix
        if output_bucket:
            arguments["--output_bucket"] = output_bucket
        if data_bucket:
            arguments["--data_bucket"] = data_bucket

        # Auto-generate date-partitioned enriched prefix
        if enriched_prefix:
            arguments["--enriched_prefix"] = enriched_prefix
        elif mode in ("full", "enrichment_only"):
            arguments["--enriched_prefix"] = self.build_enriched_prefix(
                time_period=time_period,
                run_id=shared_run_id,
            )

        # Auto-generate date-partitioned PyG output key
        if pyg_output_key:
            arguments["--pyg_output_key"] = pyg_output_key
        elif mode in ("full", "pyg_only"):
            arguments["--pyg_output_key"] = self.build_pyg_output_key(
                time_period=time_period,
                run_id=shared_run_id,
            )

        if pyg_config:
            arguments["--pyg_config"] = json.dumps(pyg_config)
        if enable_ontology_mapping:
            arguments["--enable_ontology_mapping"] = "true"
        if rdf_format:
            arguments["--rdf_format"] = rdf_format

        # Build start_job_run kwargs
        run_kwargs: Dict[str, Any] = {
            "JobName": self.job_name,
            "Arguments": arguments,
        }

        # Infrastructure overrides (top-level, not in Arguments)
        if worker_type:
            run_kwargs["WorkerType"] = worker_type
        if number_of_workers is not None:
            run_kwargs["NumberOfWorkers"] = number_of_workers
        if timeout is not None:
            run_kwargs["Timeout"] = timeout

        response = self._glue.start_job_run(**run_kwargs)
        job_run_id = response["JobRunId"]

        effective_workers = worker_type or self.default_worker_type
        effective_count = (
            number_of_workers
            if number_of_workers is not None
            else self.default_number_of_workers
        )

        print(f"Started Glue job run: {job_run_id}")
        print(f"  Job:        {self.job_name}")
        print(f"  Mode:       {mode}")
        print(f"  Workers:    {effective_workers} x {effective_count}")
        print(f"  Run ID:     {shared_run_id}")
        if "--enriched_prefix" in arguments:
            bucket = data_bucket or self.data_bucket
            print(
                f"  Enriched:   s3://{bucket}/"
                f"{arguments['--enriched_prefix']}"
            )
        if "--pyg_output_key" in arguments:
            bucket = output_bucket or self.output_bucket
            print(
                f"  PyG output: s3://{bucket}/"
                f"{arguments['--pyg_output_key']}"
            )

        return job_run_id

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get the status of a Glue job run.

        Args:
            run_id: Glue JobRunId

        Returns:
            JobRun dict from Glue API
        """
        response = self._glue.get_job_run(
            JobName=self.job_name,
            RunId=run_id,
        )
        return response["JobRun"]

    def wait_for_completion(
        self,
        run_id: str,
        poll_interval: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Poll until a Glue job run completes.

        Args:
            run_id: JobRunId to monitor
            poll_interval: Seconds between status checks
            verbose: Print status updates

        Returns:
            Final JobRun dict

        Raises:
            RuntimeError: If the job fails or is stopped
        """
        terminal_states = {
            "SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT", "ERROR",
        }
        start_time = time.time()

        while True:
            job_run = self.get_run_status(run_id)
            state = job_run["JobRunState"]
            elapsed = time.time() - start_time

            if verbose:
                exec_time = job_run.get("ExecutionTime", 0)
                print(
                    f"  [{elapsed:6.0f}s] {run_id}: {state}"
                    f"  (execution: {exec_time}s)"
                )

            if state in terminal_states:
                if state != "SUCCEEDED":
                    error_msg = job_run.get(
                        "ErrorMessage", "Unknown error"
                    )
                    raise RuntimeError(
                        f"Glue job {run_id} ended with state "
                        f"{state}: {error_msg}"
                    )
                return job_run

            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Output loading
    # ------------------------------------------------------------------
    def load_pyg_output(
        self,
        run_id: Optional[str] = None,
        pyg_output_key: Optional[str] = None,
    ):
        """Download and load a PyG HeteroData from S3.

        Requires torch to be installed in the notebook environment.

        Args:
            run_id: Glue JobRunId — reads output key from run arguments
            pyg_output_key: Direct S3 key (overrides run_id lookup)

        Returns:
            PyG HeteroData object
        """
        import io
        import torch

        if not pyg_output_key and run_id:
            job_run = self.get_run_status(run_id)
            pyg_output_key = job_run.get("Arguments", {}).get(
                "--pyg_output_key", ""
            )

        if not pyg_output_key:
            raise ValueError(
                "Could not determine pyg_output_key. "
                "Provide run_id or pyg_output_key directly."
            )

        print(f"Loading s3://{self.output_bucket}/{pyg_output_key}")

        response = self._s3.get_object(
            Bucket=self.output_bucket,
            Key=pyg_output_key,
        )

        buffer = io.BytesIO(response["Body"].read())
        hetero_data = torch.load(buffer, weights_only=False)

        print("Loaded HeteroData:")
        print(f"  Node types: {hetero_data.node_types}")
        print(f"  Edge types: {hetero_data.edge_types}")
        for nt in hetero_data.node_types:
            store = hetero_data[nt]
            n = (
                store.num_nodes
                if hasattr(store, "num_nodes")
                else "?"
            )
            f = (
                store.x.shape[1]
                if hasattr(store, "x") and store.x is not None
                else 0
            )
            print(f"  [{nt}] nodes={n}, features={f}")

        return hetero_data

    def get_run_output_paths(
        self,
        run_id: str,
    ) -> Dict[str, str]:
        """Get the S3 paths for a completed run's outputs.

        Args:
            run_id: Glue JobRunId

        Returns:
            Dict with keys 'enriched_prefix', 'pyg_output_key'
            (only keys that exist in the run arguments)
        """
        job_run = self.get_run_status(run_id)
        args = job_run.get("Arguments", {})

        paths = {}
        if "--enriched_prefix" in args:
            paths["enriched_prefix"] = (
                f"s3://{self.data_bucket}/"
                f"{args['--enriched_prefix']}"
            )
        if "--pyg_output_key" in args:
            paths["pyg_output_key"] = (
                f"s3://{self.output_bucket}/"
                f"{args['--pyg_output_key']}"
            )

        return paths

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------
    def list_enriched_artifacts(
        self,
        year: Optional[str] = None,
        month: Optional[str] = None,
    ) -> List[str]:
        """List enriched Parquet directories in date-partitioned structure.

        Returns the directory prefixes (run_id level), not individual
        Parquet part files. Identifies complete directories by the
        presence of a _SUCCESS marker file.

        Args:
            year: Filter by year (e.g., '2024')
            month: Filter by month (e.g., '12')

        Returns:
            List of S3 prefix strings (e.g.,
            ['enriched/year=2024/month=12/abc123/'])
        """
        prefix = "enriched/"
        if year:
            prefix += f"year={year}/"
            if month:
                prefix += f"month={month}/"

        success_keys = self._list_s3_keys(
            self.data_bucket, prefix, suffixes=["_SUCCESS"]
        )

        return [
            key.rsplit("_SUCCESS", 1)[0]
            for key in success_keys
        ]

    def list_pyg_outputs(
        self,
        year: Optional[str] = None,
        month: Optional[str] = None,
    ) -> List[str]:
        """List PyG outputs in date-partitioned structure.

        Args:
            year: Filter by year (e.g., '2024')
            month: Filter by month (e.g., '12')

        Returns:
            List of S3 key strings ending in .pt
        """
        prefix = "pyg/"
        if year:
            prefix += f"year={year}/"
            if month:
                prefix += f"month={month}/"

        return self._list_s3_keys(
            self.output_bucket, prefix, suffixes=[".pt"]
        )

    def list_manifests(
        self,
        year: Optional[str] = None,
        month: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[str]:
        """List job manifests in date-partitioned structure.

        Args:
            year: Filter by year (e.g., '2024')
            month: Filter by month (e.g., '12')
            mode: Filter by mode prefix (e.g., 'full', 'pyg_only')

        Returns:
            List of S3 key strings ending in .json
        """
        prefix = "manifests/"
        if year:
            prefix += f"year={year}/"
            if month:
                prefix += f"month={month}/"
                if mode:
                    prefix += f"{mode}_"

        return self._list_s3_keys(
            self.output_bucket, prefix, suffixes=[".json"]
        )

    def load_manifest(self, manifest_key: str) -> Dict[str, Any]:
        """Load a job manifest JSON from S3.

        Args:
            manifest_key: S3 key for the manifest file

        Returns:
            Parsed manifest dict
        """
        response = self._s3.get_object(
            Bucket=self.output_bucket,
            Key=manifest_key,
        )

        content = response["Body"].read().decode("utf-8")
        return json.loads(content)

    def _list_s3_keys(
        self,
        bucket: str,
        prefix: str,
        suffixes: Optional[List[str]] = None,
    ) -> List[str]:
        """List S3 keys under a prefix, optionally filtered by suffix.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix
            suffixes: Optional list of suffix strings to filter by

        Returns:
            List of matching S3 key strings
        """
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffixes is None or any(
                    key.endswith(s) for s in suffixes
                ):
                    keys.append(key)
        return keys
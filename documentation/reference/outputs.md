# Metadata Files

Every PyG build produces seven JSON metadata files written alongside the `.pt` file (locally, and mirrored to S3 when an archive is configured). Six of them are schema-level: they enable downstream training and inference code to consistently use the `HeteroData` object without re-running the pipeline.

The seventh, [`checksums.json`](#checksumsjson), is written last and is not schema-level. It records the size and SHA-256 of the `.pt` and of the other six, so a consumer can tell the bytes it fetched are the bytes the job wrote — which matters here because the `.pt` is a pickle and loading one is running whatever is inside it.

## Output Location

Written under `--local_work_dir` (and mirrored under the S3 archive bucket/key when `--s3_archive_bucket` is set).

The period is written as Hive-style partition directories (`year=2024/month=12`) derived from `--time_period`. Everything a build produces stays under that one directory, so a period can be copied, archived or deleted as a unit:

```
<local_work_dir>/pyg/year=2024/month=12/
├── hetero_data.pt
├── metadata/
│   ├── graph_schema.json
│   ├── feature_spec.json
│   ├── normalization.json
│   ├── encoding_config.json
│   ├── ontology_schema.json
│   ├── slot_mapping.json
│   └── checksums.json
└── node_index/
    └── part-*.parquet
```

Spark's partition discovery reads `key=value` directory names into real columns, so the tabular artifacts read as partitioned tables across every period they hold — a filter on `year`/`month` becomes a PartitionFilter and unmatched periods are never opened:

```python
# enriched triples: the subtree is uniformly Parquet, so read it directly
spark.read.parquet(f"{work_dir}/enriched")
# columns: [subject, predicate, object, year, month]

# node_index: pass basePath, since the period directory also holds the
# .pt blob and the JSON metadata
spark.read.option("basePath", f"{work_dir}/pyg") \
     .parquet(f"{work_dir}/pyg/year=*/month=*/node_index")
# columns: [node_type, node_id, uri, year, month]
```

A `--time_period` that is not `YYYY-MM` is written as a single path segment instead, unpartitioned. There is no `day=` level: `--time_period` is monthly, so it would carry one value per month — path depth with no pruning benefit.

For experiment variants (non-default `--pyg_filename`), the metadata and node-index directories are named after the output file stem, so two variants in one period cannot overwrite each other:

```
<local_work_dir>/pyg/year=2024/month=12/
├── hetero_data_512d.pt
├── hetero_data_512d_metadata/
│   ├── graph_schema.json
│   └── ...
└── hetero_data_512d_node_index/
    └── part-*.parquet
```

The metadata and node-index directories are derived automatically from the `.pt` filename (`--pyg_filename`, and the S3 `--s3_pyg_key` when archiving) by `derive_metadata_prefix()` / `derive_node_index_prefix()` in `metadata_writer.py`. No additional configuration is required.

## The `latest` Alias — a Consumer-Facing Contract

The period layout above is addressable only by a reader who already knows which period is newest. A consumer fetching one fixed URL cannot list the storage to find out, and should not be able to. Every build therefore also writes `graph_schema.json` to a fixed segment that always names the most recent build:

```
<local_work_dir>/pyg/latest/metadata/graph_schema.json
```

This is the **stable key**. It is the same layout as a period path with `year=YYYY/month=MM` replaced by `latest`, so a consumer URL differs from a period URL by exactly one segment. Treat the following as the contract:

| property | guarantee |
|---|---|
| key | `<base>/pyg/latest/metadata/graph_schema.json` |
| content | byte-identical to that build's period-partitioned copy |
| freshness | overwritten by every build; always the most recent |
| variants | a non-default `--pyg_filename` aliases to `<base>/pyg/latest/{stem}_metadata/graph_schema.json`, so variants never collide |
| scope | **only** `graph_schema.json` — there is no `.pt`, no node index and no `checksums.json` under `latest/` |

Only the schema is aliased because it is the only artifact with an external reader. Copying all seven metadata files would advertise `latest/` as a complete build, which it is not.

One consequence to know about: a consumer pinned to the alias has nothing to verify its copy against. The record beside a build covers that build's period directory, and the alias is a copy of one file out of it. Verify against the period copy, or treat the alias as a pointer to which period to fetch rather than as the artifact itself.

`--s3_archive_bucket` mirrors the period copy using the same relative shape — `s3_pyg_key` defaults to `pyg/{period_partition}/{pyg_filename}` — so the archive and work-dir layouts already agree segment for segment. The alias is written to the work dir only; making it reachable to an external consumer (public-read, CORS, CDN) is a hosting concern outside this repository.

> **Edge case:** `--time_period latest` is a legal non-monthly label that renders to this same directory. The collision is benign — that period copy already *is* the newest build — and the alias write is skipped on the equality rather than duplicating it.

## Published Runs

A finished run can be copied to a separate prefix for consumers by [`bin/publish_run.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/bin/publish_run.py), after the run ends. [Publishing a finished run](../operations/testing.md#publishing-a-finished-run) covers running it. The job never writes this layout itself. One run is one folder, holding every variant the run built:

```
<prefix>/<dataset>/year=YYYY/month=MM/<run_id>/
├── index.json
├── 1024d/
│   ├── hetero_data_1024d.pt
│   ├── graph_schema.json
│   ├── ...
│   ├── checksums.json
│   └── node_index/
│       └── part-*.parquet
├── no_edge_features/
│   └── ...
├── enriched/
│   ├── dataset.json
│   └── triples/
│       └── part-*.parquet
└── manifests/
    └── *.json
```

It is the work directory with four changes:

1. the period moves above the run id and is dropped below it, so `enriched/year=YYYY/month=MM/triples/` becomes `enriched/triples/`
2. `hetero_data_<v>_metadata/*.json` becomes `<v>/*.json`
3. `hetero_data_<v>_node_index/` becomes `<v>/node_index/`
4. `hetero_data_<v>.pt` keeps its name, inside `<v>/`

`<dataset>` names the source set, from the job's `--dataset` or, when the job had none, `PYG_PUBLISH_DATASET`. `<run_id>` is the time the run started, in UTC. Left out: `pyg/latest/`, which means nothing inside a folder for one run, `checkpoints/`, and Hadoop's `.crc` files.

Unlike `--s3_archive_bucket`, which mirrors one build's `.pt`, metadata and manifest from inside the job in the work-directory shape, this copies a whole finished run: every variant, the node index, the enriched Parquet and the manifests.

**`index.json`** is written last, so a run folder without one is a publish that did not finish. It records `run_id`, `dataset`, `time_period`, `data_day` (the day the sources were cut from, which `month=` cannot show), `sources`, `source_of_run` (the run directory's name), `published`, each variant's folder and `notebook_label` (the notebook's name for the leg that built it), and the files each variant folder holds.

**`checksums.json` keeps the work directory's names.** Its paths are relative to the period directory, so inside a published variant folder `hetero_data_<v>_metadata/<name>` is `<v>/<name>`. Every entry resolves by its last path segment:

```python
import hashlib, json
from pathlib import Path

variant = Path("1024d")
record = json.loads((variant / "checksums.json").read_text())
for name, entry in record["artifacts"].items():
    body = (variant / name.rsplit("/", 1)[-1]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == entry["sha256"], name
```

## File Descriptions

### `graph_schema.json`

Complete inventory of every node type and edge type in the graph. The entry point for any consumer of the graph.

**Schema version: `1.2`.** 1.2 adds `relation_groups` and the `index` / `relation_group` / `src_type_index` / `dst_type_index` fields — additive, every 1.1 field keeps its name and meaning. Node-type `has_features` changed meaning in 1.1 — see below. Check the `version` field before relying on it.

**Contents:**
- Every node type with its count, source ontology URI, category tag, and `has_features`
  - `has_features` means the type carries **literal-value features**: a non-zero `literal_values` segment (the last of the three node-vector segments). On the e2e fixtures 76 of 100 node types qualify; the other 24 are pure taxonomy types (`EconomicSector`, `GeographicRegion`, `TimePeriod`, …) that carry ontology structure but no measurements.
  - It cannot usefully mean "a feature tensor exists" — `constructor.py` gives *every* node type an `x`, falling back to a zeros placeholder — nor "any non-zero value", since the `ontology_structure` segment is populated for every typed node. Literal values are the segment that actually varies between types.
  - **Through schema `1.0` this field was `count > 0`** — the node count, not features at all. It was therefore true for essentially every type, and `summary.node_types_with_literal_features` was identical to `total_node_types` by construction (a real build reported 100 and 100). A consumer filtering node types on `has_features` against a 1.0 artifact gets every type back.
  - Caveat: a type whose literal values all normalize to exactly 0.0 (a constant numeric property under z-score) reads as `false`. That is the honest answer to "does this type carry usable literal signal", but it is not the same question as "were literals present in the source".
- Every edge type as a full three-part tuple with its count, predicate URI, origin, whether it has edge features, and if so the feature dimension
  - Edge-type `has_features` was always correct — it comes from `edge_feature_flags`, which reflects whether `edge_attr` was actually produced (23 of 499 on the same build).
  - `origin` is one of **`raw`** (the relationship was stated in the source RDF), **`enrichment`** (this pipeline inferred it), or **`unification`** (a cross-source identity link). Enrichment adds ~91,000 triples on top of the raw data, so some edges are reported facts and others are derived — a distinction a model consumer should weight differently. Classified by `classify_edge_origin()` in `rdf_utils.py` from the predicate namespace **and both endpoint node types**, since the pipeline marks its output in two places: inferred links carry a minted *predicate* (under `jefflevesque.com/ontology/bls/`, …), while unification links carry a minted *node* but a standard predicate (`unified:November owl:sameAs cpi:November`). Checking the predicate alone reports every unification edge as `raw`.
  - Deliberately three values rather than separating intra- from cross-source enrichment: those share a namespace, so nothing at this layer can tell them apart, and a field promising a distinction it never emits is worse than a narrower honest one. It records *that* an edge was derived, not *why* — per-edge lineage is a much larger change, worth building only once a model's predictions need explaining.
- **`relation_groups`** (1.2): which edge types are the same relation and may **share GNN weights**. A PyG edge type is `(src_type, relation, dst_type)` and a heterogeneous conv allocates one weight matrix per edge type, so a relation spanning many endpoint pairs multiplies out — on the e2e fixtures **69 relations produce 770 edge types**, and a single `HeteroConv` 1024→128 over that is ~101M parameters for ~10k edges, with 67 edge types holding exactly one edge.
  - The `.pt` cannot collapse them: `edge_index` values are node IDs **local to their node type**, so `(cpi_Category, r, X)` and `(eci_Industry, r, X)` sharing a key would put two ID spaces in the same row. The multiplicity is forced by the container, so the pipeline publishes the grouping instead of leaving a consumer to guess it by string-splitting edge-type keys.
  - Each group carries `edge_types` (the keys it covers), `edge_type_count`, summed `count`, `predicate_uri`, `origin`, `has_features` and `feature_dim`. The grouping **partitions** the edge types — every key is in exactly one group.
  - `origin` is the one group field whose edge types can genuinely disagree, since origin reads the endpoints and a group spans endpoint pairs by definition: `jolts:hasIndustry` is `raw` across 21 of its edge types and `enrichment` on the one leaving a pipeline-minted node. Such a group reports **`mixed`**, meaning *ask the edge types* — deliberately not a fourth origin value, so a consumer switching on `origin` cannot read it as a trust level. Naming one member's origin instead is how the field was wrong before: `origin` was keyed by relation name, so all of a relation's edge types collapsed to whichever the builder wrote last, and 3 edge types (90 edges) published `raw` for links the pipeline had inferred.
  - Nothing is lost by tying: each edge type also carries `src_type_index` / `dst_type_index` into the node-type table (whose entries now carry a stable `index`, assigned by sorted name), so a shared relation weight can still condition on endpoint type through a node-type embedding — one table of *N* types rather than *N×M* matrices.
- Summary statistics: total node types, total edge types, total relation groups, total nodes, total edges, edge types with features
- Build metadata: time period, build timestamp, pipeline config

**Generated by:** `constructor.py` after HeteroData assembly, from `node_mapper.node_counts`, `node_mapper.get_type_uri_mapping()`, `edge_mapper.build_edge_indices()`, and `edge_feature_extractor.get_edge_classification()`. The per-type `has_features` flag is read off the assembled feature tensors themselves (`MetadataCollector.register_node_literal_features`), so it cannot drift from the `.pt` it describes.

**Changes between builds:** Yes — counts change every time period; new types may appear when new data sources are added.

---

### `feature_spec.json`

Defines the structure of the 1024-d node feature vector and the 32-d edge feature vectors. Tells training code what each segment means and how to route dimensions through the model architecture.

**Contents:**
- Total node feature dimension with all segment and sub-segment boundaries (start index, end index, dim, name, type)
- Flag indicating structural dimensions are shared within a node type
- Total edge feature dimension with all segment and sub-segment boundaries
- List of edge types that carry features and list that do not
- Per-relation derivation method (temporal, option_stock, escalation, correlation)

**Generated by:** `constructor.py` from `feature_extractor.get_layout().to_dict()` and `edge_feature_extractor.get_layout().to_dict()`

**Changes between builds:** Rarely — only when the feature vector design changes (new segment layout, different dimensions).

---

### `normalization.json`

Per-property normalization statistics used to z-score numeric literal values during feature encoding. Required to encode new data into the same feature space the model was trained on.

**Contents:**
- Normalization method (z-score)
- Per-property statistics: predicate URI, mean, standard deviation, count of non-null values
- List of zero-variance properties (sigma was 0, set to constant 1.0)

**Precision:** Statistics are rounded to 12 significant digits on write. `stddev` is a parallel reduction, and parallel float reductions are not order-deterministic — two runs over identical data could otherwise differ in the last ULP of a float64, making the file non-reproducible byte-for-byte. The rounding is far below float32 feature precision, so feature tensors are unaffected (they compared equal even when this file did not).

**Generated by:** `feature_extractor._collect_normalization_metadata()` during the stats aggregation pass — a single-pass `groupBy().agg()` on the numeric literals DataFrame, collecting one row per predicate (typically <200 rows)

**Changes between builds:** Yes — distribution statistics shift every time period as new data arrives. A model trained on December 2024 normalization stats expects inference data normalized with those same stats.

---

### `encoding_config.json`

Every parameter needed to deterministically reproduce the hash-based encoding. If any of these values change, the same ontology class or property hashes to different vector positions and the trained model breaks.

**Contents:**
- Hash algorithm name (`spark_murmur3`)
- Per-segment encoding parameters: dimension, number of hash functions, seed values
- Class identity seeds, class hierarchy seeds and decay function, ontology membership method
- Property presence seeds and encoding convention (1.0 present, -1.0 absent, 0.0 not in schema)
- Domain/range seeds, numeric value hashing seed, categorical value hashing seeds
- Edge feature encoding parameters: relation classification fragments, temporal normalization divisor, ratio clamp value, cross-property derivation seeds
- Total node and edge feature dimensions (`node_features.total_dim`, `edge_features.total_dim`)
- `checksum` — a **SHA-256 digest of the encoding contract**: everything above, hashed together. This is a *contract* hash, not a data hash, so rebuilding a different time period with the same settings yields the same digest, while changing any seed, dimension, segment boundary or namespace table changes it. A deployed model can compare the digest it was trained against with the one shipped alongside a graph and refuse to run on a mismatch — otherwise it would load cleanly and return plausible, silently wrong numbers, with every feature in a different slot than the weights expect.

> **Note:** this field previously held only `{"total_node_feature_dim": N}` — a dimension, not a checksum, which detected nothing (two builds with different seeds but the same vector width compared equal). It was also lost to a key collision when the node and edge configs were merged. The digest is now computed once over the merged config, so both halves contribute.

**Generated by:** `constructor.py` by merging `feature_extractor.get_encoding_config()` and `edge_feature_extractor.get_encoding_config()`; the digest is stamped by `MetadataCollector`, the only place that sees the whole contract

**Changes between builds:** Rarely — only when the encoding scheme is redesigned. Should be identical across all time periods that feed the same model.

---

### `ontology_schema.json`

Frozen snapshot of the ontology structure at build time. Contains the class hierarchies, property definitions, domain/range declarations, and namespace mappings used to compute the structural and schema segments of the feature vector.

**Contents:**
- Per node type (keyed by PyG name): source type URI, ordered superclass chain with depths, namespace, defined properties with their range types
- URI-to-PyG-name mapping for all type URIs encountered
- Namespace prefix table
- `ontology_mapping_enabled` / `ontology_mapping_evidence` — whether the ontology-mapping phase ran over the triples this build read, and what that verdict was based on
- `hierarchy_source` / `property_schema_source` / `property_hierarchy_source` — where each axiom set came from, or why there is none
- `provenance` — per axiom set, whether it was **declared by the source**, **curated**, or **observed**
- `property_schema_coverage` — how many predicates got a domain and a range, and which ones did not
- `derived_axioms` — per axiom set: how many axioms each derivation route produced (including `declared`), and which axioms those were

**Provenance: derived is not declared.** No source in this project declares `rdfs:subClassOf`, `rdfs:domain`, `rdfs:range` or `rdfs:subPropertyOf` — verified across the real 130k-triple run and both fixture sets. All four are therefore *derived* by `OntologyMapper`, and once in the graph a derived axiom is shaped exactly like a declared one. They do not license the same reasoning: "this property is used on class C in this month's data" is weaker than "this property's domain is C", and a property seen with one class here could be broader in general.

| Axiom set | Derivation routes |
|---|---|
| `class_hierarchy` | `curated class mappings` + `class naming` |
| `property_hierarchy` | `curated property mappings` (shared `PROPERTY_MAPPINGS` targets) |
| `property_domain` | `observed subject types` |
| `property_range` | `observed object types` + `declared literal datatype` |

**The `*_source` fields name the route, not the predicate.** `hierarchy_source` used to read `"rdfs:subClassOf"` — true of the predicate the encoder consumed, and false as an answer to "did a source declare this hierarchy". It now reads like `derived: curated class mappings (24), class naming (10)`, or `mixed: …` when a source genuinely declares some. Only a set with no derivation markers at all still reports the bare predicate name. Empty sets keep their existing "no rdfs:subClassOf in source data…" reason — that is about *absent data*, which provenance does not change.

**Counting axioms in an enriched graph does not answer "does any source declare them."** Since the mapper emits them, an enriched build contains 34 `rdfs:subClassOf` triples and zero of them came from a source. `derived_axioms.<set>.counts.declared` is the field that answers it, and it is `0` on every build since the derivations landed.

**Curated and guessed edges are separated.** The two are not equally trustworthy: a person chose each curated edge, while the naming rules produced `AllItemsLessShelter -> Shelter` and three more inversions before the negating-qualifier guard caught them. `derived_axioms.class_hierarchy.axioms` lists the edges of each route, and every entry in a node type's `superclass_chain` carries its own `provenance` — so chasing a wrong superclass tells you whether to edit `CLASS_MAPPINGS` or fix a naming rule, without reading the code. Links at `depth > 1` report `"transitive closure of the direct edges"`, since they are this pipeline's closure rather than asserted edges with a route of their own.

**Why some predicates get no domain or range.** `rdfs:domain` is an axiom with *intersection* semantics: asserting `p rdfs:domain A` and `p rdfs:domain B` says every subject of `p` is both an A and a B. Measured over the fixtures, 77 of 160 predicates are used on more than one class (`market:askPrice` on `EquitySnapshot` and `OptionSnapshot`; `rdfs:label` on 68), so emitting both would state something false. A predicate with more than one candidate therefore gets **no axiom**, and `property_schema_coverage` names it. Range applies the same rule across both routes at once, which also rules out the two predicates used with entities *and* literals (`cpi:hasMonth` is 189 typed URIs and 11 literals).

Skipping costs less than the count suggests: a predicate used on one class carries real signal about that class, while one used on 22 says little that `class_identity` does not already encode about the node itself.

**Measured coverage** over the full committed fixture set (all four sources, both loaders — 169 data predicates after enrichment):

| | Covered | Gap | Why the gap |
|---|---|---|---|
| `rdfs:domain` | 83 (49.1%) | 86 | 77 predicates are used on more than one class; the rest carry no typed subject |
| `rdfs:range` | 135 (79.9%) | 34 | 16 ambiguous (several object classes, or entity *and* literal use); the remainder are CAP predicates whose URI objects are never typed in these fixtures |

Every one of those gaps is listed by URI in `property_schema_coverage.without_domain` / `without_range`, so it is a known set rather than something to be inferred from a thin sub-segment. Resulting occupancy, against 0.00% for all three before this: `class_hierarchy` 1.25%, `domain_range` 2.83% (carried by 77 of 102 node types), `property_hierarchy` 2.31%.

**Reading an empty hierarchy.** When every `superclass_chain` is `[]`, `ontology_mapping_enabled` tells you which of the two causes you are looking at, because they demand opposite responses:

| `ontology_mapping_enabled` | Meaning | What to do |
|---|---|---|
| `true` | The mapping phase ran and the sources genuinely declare no subsumption | Nothing — that is the data |
| `false` | The hierarchy was never computed; `class_hierarchy` (64 of 1024 dims, 6.25% of every node vector) is structurally zero | Re-run enrichment with `--enable_ontology_mapping true` |

The flag arrived in schema `version` `1.1`, `provenance` / `property_schema_coverage` in `1.2`, and the route-aware `*_source` strings plus `derived_axioms` in `1.3`; a `1.0` file predates all of them and its empty hierarchy stays ambiguous. It is detected from the triples — the presence of `owl:equivalentProperty` / `owl:equivalentClass`, which only `OntologyMapper` emits — rather than read off a config flag. That is the only signal that stays truthful in `pyg_only` mode, where enrichment ran in a separate job and this job's own `--enable_ontology_mapping` describes a phase it never reaches. For the same reason a `pyg_only` job manifest records `"enable_ontology_mapping": null` rather than the inert flag — that field says nothing about the enriched Parquet the run consumed, and this file is what answers the question for it.

**Generated by:** `feature_extractor._collect_ontology_schema_metadata()` — collects from small distinct/aggregated DataFrames: type URIs (~500 rows), class hierarchy transitive closure (~5000 rows), property schema (~500 rows). All collect calls target aggregated DataFrames, never raw triples.

**Changes between builds:** Sometimes — when new data sources or ontologies are added. If the ontology structure changes between the training build and an inference build, the structural segment of the feature vector will differ. This file lets you detect that.

---

### `slot_mapping.json`

Maps specific vector dimensions back to their semantic meaning. Purely for interpretability — no training or inference code depends on this file.

**Contents:**
- Per numeric property: predicate URI, local name, hash slot within the numeric sub-segment, global dimension index
- Per categorical property: predicate URI, local name, hash slots (multiple due to multi-hot), global dimension indices
- Per class: class URI, PyG name, hash slots in the class identity sub-segment, global dimension indices
- Per superclass: which slots in the hierarchy sub-segment each superclass contributes to
- Per namespace: which slot in the ontology source sub-segment each namespace occupies
- Hash collision report: collision counts and rates per sub-segment. For `class_identity` (slot mapping **1.1**) this reports *separability* rather than slot occupancy — see below.

> **`class_identity` in the collision report (slot mapping 1.1).** Class identity is a **multi-hot** code: each class occupies `num_hashes` slots and is identified by the *set*, not by any single slot. Slot reuse is therefore not identity loss. Through slot mapping 1.0 this sub-segment reported `collisions` / `collision_rate` computed as raw slot occupancy, which reads as lost identity and is misleading: with more hash entries than dimensions, pigeonhole forces a high value however healthy the code is. A real build with 44 classes × 4 hashes into 64 dims reported `collision_rate: 0.67` while every one of the 44 codes was distinct, the code matrix was full rank, and its condition number was ~12 — identity fully recoverable.
>
> Those keys are now `slot_reuse` / `slot_reuse_rate`, and the fields that do bound identity are reported alongside: `distinct_codes`, `classes_sharing_a_code` (classes that are genuinely indistinguishable), `max_pairwise_slot_overlap`, `capacity_classes`, `headroom_classes`, `code_matrix_rank`, `rank_deficiency`, and `linearly_separable`.
>
> **`linearly_separable` is measured, not inferred.** It was previously computed as `num_classes <= dim and not classes_sharing_a_code` — two conditions that are each *necessary* but not together *sufficient*, reported under the name of a rank test. Distinct codes inside the ceiling can still be linearly dependent: the 4-hot codes `{0,1,2,3}`, `{0,1,4,5}`, `{2,3,6,7}`, `{4,5,6,7}` are pairwise distinct and fit in 8 dims, yet A + D − B − C = 0, so one class is exactly a blend of the others and no linear readout can recover it. That build reported `linearly_separable: true` and shipped. The field is now the measured rank of the `num_classes × dim` code matrix (`code_matrix_rank == total_classes`), with `rank_deficiency` giving how many classes are unrecoverable.
>
> This also means `capacity_classes` is a **ceiling, not a guarantee** — `d` 4-hot codes drawn into `d` dims measure rank `d−2`, so plan headroom rather than aiming at the limit.
>
> A build that is not separable — over-subscribed, sharing a code, or rank-deficient — now **raises `ClassIdentityCapacityError`** instead of logging. A warning was the wrong severity: the artifact ships, every consistency check passes, and the only symptom is a model that never learns to tell two classes apart, weeks downstream with nothing pointing back at the build. `feature_config.allow_class_identity_oversubscription=true` restores warn-and-continue. Approaching the limit (past 85% of the segment width) still only warns.

**Generated by:** `feature_extractor._collect_slot_mapping_metadata()` — computes approximate slot assignments on the driver using a Python hash approximation of Spark's murmur3. The approximation may not match exactly for all inputs; this file is for interpretability only and is never used by training or inference code.

**Changes between builds:** Only when the encoding config or ontology schema changes. If neither changes, the slot mapping is identical across time periods.

### `node_index/` (Parquet)

The identity map: which real-world entity each row of the graph is. `hetero_data.pt` stores only `x` and `num_nodes` per node type, so without this the graph is **anonymous** — row 5 of `cpi_Index` is a specific CPI series and nothing else on disk records which one.

The six JSON files above are all schema-level: together they answer *"what does column 37 mean?"*. This is the one artifact that answers *"who is row 5?"*.

**Contents:** one row per node — `node_type`, `node_id`, `uri`. Sorted by `(node_type, node_id)` and coalesced to a single file so the artifact is content-stable run to run.

**Why Parquet, not an eighth JSON:** production volume is 322.7M triples for a single four-source day, so this can reach millions of rows. A single JSON would be hundreds of MB and would need parsing in full to resolve one entity; Parquet supports predicate pushdown and matches how the rest of the pipeline stores bulk data.

**Needed for:**
- **Training** — labels arrive keyed by entity; without this there is nothing to join them on, so a target tensor aligned to the graph cannot be built
- **Inference** — to look up a specific entity's row, and to attribute a prediction ("row 5 scores 0.93") back to something meaningful
- **Rebuilds** — node IDs come from `row_number()` over a `uri`-ordered window, so they are deterministic for a given input, but adding or removing one entity shifts every row below it. A model trained on one period cannot be applied to the next without each build's own map.

**Generated by:** `constructor._node_index()`, written by `save_node_index()` in `build_graph.py`. Written by Spark directly from the executors, so it lands in object storage when `--local_work_dir` is an `s3a://` URI (as on the cluster) — it is not part of the boto3 mirror, being a distributed dataset rather than a driver-side blob.

**Changes between builds:** Every build — it is per-period by nature.

### `checksums.json`

What a build wrote, how big each file was, and its SHA-256. Written last, into the same metadata directory, so a period directory can be checked against itself:

```json
{
  "algorithm": "sha256",
  "written": "2026-09-09T23:14:02Z",
  "artifacts": {
    "hetero_data.pt":             {"bytes": 45566402048, "sha256": "…"},
    "metadata/graph_schema.json": {"bytes": 184320,      "sha256": "…"}
  }
}
```

**Why it exists:** nothing else a build produced could be checked after the fact. A truncated upload, a half-copied period directory and a deliberately swapped file all read as a valid build. The last case is not a hypothetical inconvenience — `hetero_data.pt` is a pickle and every reader loads it with `torch.load(..., weights_only=False)`, so a replaced artifact runs whatever code is in it on whoever trains against the graph. Checking the digest first is what makes that risk manageable.

**Paths are relative to the period directory**, not to the metadata directory the file sits in and not absolute. The period directory is the unit that gets copied, archived or deleted, so relative names keep resolving after a copy. A variant build (`--pyg_filename hetero_data_512d.pt`) records `hetero_data_512d.pt` and `hetero_data_512d_metadata/…`, in that variant's own metadata directory — two variants in one period cannot overwrite each other's record.

**What it does not answer:** whether a rebuild would produce these bytes. `torch.save` is not byte-stable across environments and `normalization.json` is documented as non-reproducible byte-for-byte in the last digit of a float. This says *"is this the file the job wrote"*, which is the question a consumer actually has.

**Not covered:** `node_index/part-*.parquet`. Spark writes those from the executors, so digesting them would mean the driver re-reading every part file — real cost on a 322.7M-triple day.

**Verifying a copy:**

```python
import hashlib, json
from pathlib import Path

period = Path("pyg/year=2024/month=12")
record = json.loads((period / "metadata" / "checksums.json").read_text())
for name, entry in record["artifacts"].items():
    body = (period / name).read_bytes()
    assert hashlib.sha256(body).hexdigest() == entry["sha256"], name
```

**On S3:** the mirror carries its own copy of the record, over its own objects, so an archived period prefix verifies the same way. Every upload also sets `ChecksumAlgorithm="SHA256"`, so S3 checks the transfer on receipt and `HeadObject` returns a digest without downloading the object. For the `.pt` that stored value is a checksum *of the part checksums* with a `-N` suffix, because an upload that size is multipart — the whole-file digest is the one in `checksums.json`. (The same reason ETag is not a content hash.)

**Not tamper evidence.** The same role writes the data and the digests, so this detects corruption, not an adversary who can write to the bucket.

**Generated by:** `write_checksums_to_local()` / `write_checksums_to_s3()` in `metadata_writer.py`, from digests taken during each write by `save_pyg_local()`, `save_pyg_to_s3()` and the metadata writers. The same block is also recorded in the job manifest under `result.artifacts`.

**Changes between builds:** Every build.

---

## Consumer Matrix

| File | Training (model init) | Inference (encode new data) | Inference (model load) | Human exploration | Experiment tracking |
|---|---|---|---|---|---|
| `graph_schema.json` | **Yes** — architecture decisions, data splits | **Yes** — validate compatible types | No | **Yes** — first file to read | **Yes** — what each experiment contained |
| `feature_spec.json` | **Yes** — layer construction, conv routing | No | **Yes** — reconstruct same architecture | Occasionally | **Yes** — compare feature designs |
| `normalization.json` | No | **Yes** — scale new data identically | No | No | Occasionally — detect distribution drift |
| `node_index/` | **Yes** — join labels to rows | No | **Yes** — resolve entity ↔ row, attribute predictions | **Yes** — the only way to tell what a row is | No |
| `encoding_config.json` | No | **Yes** — hash to same slots | **Yes** — compare `checksum.contract_digest` and refuse a mismatch | No | **Yes** — the digest identifies the encoding contract exactly |
| `ontology_schema.json` | No | **Yes** — encode new nodes with correct hierarchy | No | Occasionally | **Yes** — detect schema drift |
| `slot_mapping.json` | No | No | No | **Yes** — interpret model attention | No |
| `checksums.json` | **Yes** — check the `.pt` before `torch.load` unpickles it | No | **Yes** — same check on the artifact being loaded | Occasionally — confirm a copy arrived intact | **Yes** — names the exact bytes a run produced |


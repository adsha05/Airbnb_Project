# Airbnb Data Science Pipeline SVG

![Airbnb data science pipeline](assets/airbnb-ds-pipeline-overview.svg)

## Purpose

`assets/airbnb-ds-pipeline-overview.svg` is a conceptual view of an end-to-end
Airbnb machine-learning system. It extends the modeling work in
`Copy_of_Eeaman6.ipynb` into the surrounding data, training, serving,
experimentation, monitoring, and retraining infrastructure required for a
production system.

The diagram is an architecture reference, not an implementation of the
pipeline and not proof of Airbnb's current internal production architecture.
The named technologies illustrate how such a system can be built.

## Pipeline represented by the diagram

| Stage | Responsibility | Example implementation |
| --- | --- | --- |
| 1. Raw events | Capture guest searches, host actions, transactions, reviews, messages, and request context | Versioned Protobuf events sent through Kafka |
| 2. Processing and storage | Build real-time aggregates and historical training datasets | Flink for streams, Spark for batch processing, Parquet in S3, and an online/offline feature store |
| 3. Feature engineering | Produce guest, listing, market, and trust signals | Point-in-time joins that exclude information unavailable at prediction time |
| 4. ML domains | Rank search results, optimize prices, and detect trust or safety risks | Ranking models, forecasting and causal models, and XGBoost classifiers |
| 5. Training | Run reproducible scheduled training jobs and register model artifacts | Kubernetes compute, MLflow, a model registry, and Airflow |
| 6. Offline evaluation | Compare model quality, latency, drift, and segment behavior | NDCG/MRR, WMAPE/MAE, PR-AUC, latency percentiles, and slice analysis |
| 7. Real-time serving | Retrieve, rank, and post-process listings within a latency budget | Elasticsearch/ANN retrieval followed by LightGBM or DNN ranking |
| 8. Experimentation and monitoring | Measure business impact and detect production regressions | Guest-level A/B tests, guardrails, drift alerts, and automatic rollback |
| 9. Deployment and feedback | Roll out safely and turn outcomes into future labels | Canary deployment, multi-region serving, label collection, and scheduled retraining |

The feedback edge joins impression logs to later booking outcomes. Because
bookings can arrive days after an impression, label generation and experiment
analysis must account for delayed outcomes.

## Relationship to this repository

The notebook currently implements only part of the larger architecture:

| Repository capability | Diagram stage |
| --- | --- |
| CSV loading and joining | Simplified Stage 2 |
| Text, host, location, pricing, and availability features | Stage 3 |
| Random Forest and XGBoost rating classification | Stage 4 |
| ROC AUC, TPR, FPR, accuracy, and threshold analysis | Stage 6 |

The repository does not yet implement event ingestion, a feature store,
scheduled training, a model registry, online serving, A/B testing, production
monitoring, canary deployment, or automated feedback-based retraining.

For this notebook's perfect-rating classifier, a practical implementation of
the diagram would be:

1. Define a versioned listing schema and stable listing identifier.
2. Join listing, amenity, host, and label data with point-in-time correctness.
3. Put every transformation into one fitted preprocessing pipeline.
4. Track the source revision, data snapshot, random seed, parameters, and
   metrics for each XGBoost run.
5. Select a threshold on validation data using the required false-positive
   constraint.
6. Register the complete preprocessing-and-model artifact.
7. Serve probabilities and threshold decisions through one versioned
   endpoint.
8. Monitor input drift, calibration, TPR/FPR, latency, and performance by
   market and property type.
9. Join later rating outcomes back to predictions for controlled retraining.

## How the SVG is implemented

The file is self-contained XML using standard SVG primitives:

- The root has `width="100%"` and `viewBox="0 0 719.11 2020"`. This makes the
  tall diagram scale to the width of its container while preserving its
  coordinate system and aspect ratio.
- `<title>` and `<desc>` provide an accessible name and detailed description,
  while `role="img"` identifies the document as an image.
- `<g>` groups collect related boxes and labels.
- Rounded `<rect>` elements form stages, model domains, and metric cards.
- `<text>` elements contain the visible labels.
- `<line>` and `<path>` elements create connectors and the retraining loop.
- A reusable `<marker id="arrow">` supplies arrowheads.
- A mask creates gaps where connectors would otherwise cross text.
- Inline styles define the purple feature/model palette, neutral borders,
  font sizes, and a system-font fallback stack.

The SVG contains no raster images, JavaScript, external stylesheets, or remote
assets. It can therefore render directly in GitHub Markdown and modern
browsers without a build step.

Despite the visible “Click any node to explore that stage in depth” footer,
the current SVG is static. There are no `<a>` elements, event handlers, or
scripts. To make nodes navigable without JavaScript, wrap each relevant group
in an SVG anchor:

```xml
<a href="#stage-3-feature-engineering" aria-label="Feature engineering">
  <g id="stage-3">
    <!-- Existing stage shapes and text -->
  </g>
</a>
```

For links to work meaningfully, the destination document must expose matching
section anchors.

## Editing guidelines

1. Keep the `viewBox` unless the canvas geometry changes.
2. Edit visible labels in `<text>` elements and XML-escape characters such as
   `&`, `<`, and `>`.
3. Move all shapes, labels, and connectors in a stage together so their
   coordinates remain aligned.
4. Reuse the existing color and typography styles rather than introducing
   one-off values.
5. Preserve `<title>`, `<desc>`, `role="img"`, and meaningful link labels.
6. Do not add embedded scripts or remote assets; GitHub may sanitize them and
   they make the artifact harder to review.
7. Open the SVG directly in a browser after every change and verify it at both
   narrow and wide viewport sizes.

## Embedding

Markdown:

```markdown
![Airbnb data science pipeline](docs/assets/airbnb-ds-pipeline-overview.svg)
```

HTML with explicit sizing:

```html
<img
  src="docs/assets/airbnb-ds-pipeline-overview.svg"
  alt="Nine-stage Airbnb data science pipeline"
  width="720"
/>
```

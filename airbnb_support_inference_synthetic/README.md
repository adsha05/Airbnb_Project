# Airbnb Support Inference Synthetic Data

This project generates synthetic Airbnb support data and demonstrates
repeat-contact prediction, explainability, routing-policy simulation, and
causal analysis for senior-agent routing.

## Structure

```text
airbnb_support_inference_synthetic/
├── generate_airbnb_support_data.py
├── requirements.txt
├── README.md
├── PRODUCTION_ROLLOUT.md
├── src/
│   └── generate_data.py
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_eda_support_inference.ipynb
│   ├── 03_repeat_contact_prediction.ipynb
│   ├── 04_strong_model_explainability.ipynb
│   └── 05_causal_inference_senior_routing.ipynb
├── docs/
│   └── production-rollout.svg
└── data/
    ├── dim_guest.csv
    ├── dim_host.csv
    ├── dim_listing.csv
    ├── fact_booking.csv
    ├── fact_support_ticket.csv
    └── mart_ticket_modeling.csv
```

The generated modeling mart contains 10,000 synthetic support tickets and 52
fields. The notebooks read data from the root `data/` directory.

## Production rollout

See [PRODUCTION_ROLLOUT.md](PRODUCTION_ROLLOUT.md) for the production
architecture, feature contracts, release gates, causal experiment design,
monitoring plan, and phased rollout strategy.

![Production rollout architecture](docs/production-rollout.svg)

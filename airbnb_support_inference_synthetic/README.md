# Airbnb Support Inference Synthetic Data

This directory contains the scaffold for generating and storing synthetic
Airbnb support-inference data.

## Structure

```text
airbnb_support_inference_synthetic/
├── generate_airbnb_support_data.py
├── README.md
└── data/
    ├── dim_guest.csv
    ├── dim_host.csv
    ├── dim_listing.csv
    ├── fact_booking.csv
    ├── fact_support_ticket.csv
    ├── fact_support_message.csv
    ├── fact_agent_action.csv
    ├── mart_ticket_modeling.csv
    └── data_dictionary.json
```

The CSV files are empty placeholders. Add generation logic to
`generate_airbnb_support_data.py` once the schemas and synthetic-data rules
are defined.

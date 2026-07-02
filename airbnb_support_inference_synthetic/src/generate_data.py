import numpy as np
import pandas as pd
from pathlib import Path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_airbnb_support_data(n_tickets=10000, seed=42, output_dir="data"):
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. Create guests
    # -----------------------------
    n_guests = 4000

    guests = pd.DataFrame({
        "guest_id": [f"G{i:05d}" for i in range(n_guests)],
        "guest_account_age_days": np.random.randint(1, 3000, n_guests),
        "guest_past_bookings_count": np.random.poisson(5, n_guests),
        "guest_past_support_tickets_12m": np.random.poisson(1.2, n_guests),
        "guest_past_refund_requests_12m": np.random.poisson(0.4, n_guests),
        "guest_lifetime_booking_value": np.random.gamma(4, 600, n_guests).round(2),
        "guest_id_verified": np.random.choice([0, 1], n_guests, p=[0.12, 0.88])
    })

    # -----------------------------
    # 2. Create hosts
    # -----------------------------
    n_hosts = 1500

    hosts = pd.DataFrame({
        "host_id": [f"H{i:05d}" for i in range(n_hosts)],
        "host_superhost": np.random.choice([0, 1], n_hosts, p=[0.65, 0.35]),
        "host_response_rate": np.random.beta(12, 3, n_hosts).round(3),
        "host_cancellation_rate_12m": np.random.beta(2, 30, n_hosts).round(3),
        "host_avg_response_time_minutes": np.random.gamma(2, 60, n_hosts).round(1),
        "host_past_guest_complaints_12m": np.random.poisson(2, n_hosts),
        "host_avg_rating": np.random.normal(4.65, 0.25, n_hosts).clip(3.2, 5).round(2)
    })

    # -----------------------------
    # 3. Create listings
    # -----------------------------
    n_listings = 3000

    markets = [
        "New York", "Los Angeles", "Miami", "Austin", "Chicago",
        "Seattle", "Denver", "Boston", "San Francisco", "Nashville"
    ]

    listings = pd.DataFrame({
        "listing_id": [f"L{i:05d}" for i in range(n_listings)],
        "host_id": np.random.choice(hosts["host_id"], n_listings),
        "market": np.random.choice(markets, n_listings),
        "property_type": np.random.choice(
            ["entire_home", "private_room", "hotel_room"],
            n_listings,
            p=[0.72, 0.22, 0.06]
        ),
        "nightly_price": np.random.lognormal(5.1, 0.45, n_listings).clip(40, 900).round(2),
        "listing_review_score": np.random.normal(4.55, 0.30, n_listings).clip(2.8, 5).round(2),
        "listing_cleanliness_score": np.random.normal(4.50, 0.35, n_listings).clip(2.5, 5).round(2),
        "listing_accuracy_score": np.random.normal(4.45, 0.35, n_listings).clip(2.5, 5).round(2),
        "listing_review_count": np.random.negative_binomial(4, 0.25, n_listings),
        "cancellation_policy": np.random.choice(
            ["flexible", "moderate", "strict"],
            n_listings,
            p=[0.35, 0.42, 0.23]
        ),
        "instant_book_enabled": np.random.choice([0, 1], n_listings, p=[0.40, 0.60])
    })

    # -----------------------------
    # 4. Create bookings
    # -----------------------------
    n_bookings = n_tickets

    sampled_listings = listings.sample(n_bookings, replace=True).reset_index(drop=True)

    bookings = pd.DataFrame({
        "booking_id": [f"B{i:06d}" for i in range(n_bookings)],
        "guest_id": np.random.choice(guests["guest_id"], n_bookings),
        "host_id": sampled_listings["host_id"],
        "listing_id": sampled_listings["listing_id"],
        "nights": np.random.randint(1, 10, n_bookings),
        "days_until_checkin_at_ticket": np.random.randint(-2, 45, n_bookings)
    })

    bookings["total_booking_value"] = (
        bookings["nights"] * sampled_listings["nightly_price"] * np.random.normal(1.15, 0.08, n_bookings)
    ).round(2)

    bookings["booking_status"] = np.random.choice(
        ["completed", "cancelled", "active"],
        n_bookings,
        p=[0.78, 0.12, 0.10]
    )

    bookings["cancelled_by"] = np.where(
        bookings["booking_status"] == "cancelled",
        np.random.choice(["guest", "host"], n_bookings, p=[0.62, 0.38]),
        "none"
    )

    # -----------------------------
    # 5. Create support tickets
    # -----------------------------
    issue_categories = [
        "refund_request",
        "host_cancellation",
        "listing_not_as_described",
        "checkin_problem",
        "payment_problem",
        "safety_issue",
        "policy_question"
    ]

    tickets = pd.DataFrame({
        "ticket_id": [f"T{i:06d}" for i in range(n_tickets)],
        "booking_id": bookings["booking_id"],
        "issue_category": np.random.choice(
            issue_categories,
            n_tickets,
            p=[0.20, 0.14, 0.13, 0.15, 0.12, 0.06, 0.20]
        ),
        "contact_channel": np.random.choice(
            ["chat", "phone", "email", "self_serve"],
            n_tickets,
            p=[0.46, 0.20, 0.23, 0.11]
        ),
        "priority_initial": np.random.choice(
            ["low", "medium", "high"],
            n_tickets,
            p=[0.44, 0.42, 0.14]
        )
    })

    # -----------------------------
    # 6. Join context into ticket table
    # -----------------------------
    df = (
        tickets
        .merge(bookings, on="booking_id", how="left")
        .merge(guests, on="guest_id", how="left")
        .merge(hosts, on="host_id", how="left")
        .merge(listings, on=["listing_id", "host_id"], how="left")
    )

    # -----------------------------
    # 7. Feature signals
    # -----------------------------
    df["same_week_checkin_flag"] = (df["days_until_checkin_at_ticket"] <= 7).astype(int)
    df["strict_policy_flag"] = (df["cancellation_policy"] == "strict").astype(int)
    df["host_cancelled_booking_flag"] = (df["cancelled_by"] == "host").astype(int)
    df["high_value_booking_flag"] = (
        df["total_booking_value"] >= df["total_booking_value"].quantile(0.80)
    ).astype(int)

    # sentiment: more negative for serious issues
    df["sentiment_score"] = np.random.normal(-0.1, 0.35, n_tickets)

    df.loc[df["issue_category"] == "safety_issue", "sentiment_score"] -= 0.55
    df.loc[df["issue_category"] == "host_cancellation", "sentiment_score"] -= 0.35
    df.loc[df["issue_category"] == "listing_not_as_described", "sentiment_score"] -= 0.25
    df.loc[df["same_week_checkin_flag"] == 1, "sentiment_score"] -= 0.15

    df["sentiment_score"] = df["sentiment_score"].clip(-1, 1).round(3)

    # urgency score
    urgency_raw = (
        -0.8
        + 1.1 * df["same_week_checkin_flag"]
        + 1.8 * (df["issue_category"] == "safety_issue").astype(int)
        + 1.0 * (df["issue_category"] == "host_cancellation").astype(int)
        + 0.5 * df["high_value_booking_flag"]
        + 0.4 * (df["contact_channel"] == "phone").astype(int)
        + np.random.normal(0, 0.5, n_tickets)
    )

    df["urgency_score"] = sigmoid(urgency_raw).round(3)

    # first response time
    df["time_to_first_agent_response_minutes"] = np.random.gamma(2, 18, n_tickets).round(1)

    df.loc[df["priority_initial"] == "high", "time_to_first_agent_response_minutes"] *= 0.65
    df.loc[df["priority_initial"] == "low", "time_to_first_agent_response_minutes"] *= 1.35

    df["time_to_first_agent_response_minutes"] = (
        df["time_to_first_agent_response_minutes"].clip(1, 480).round(1)
    )

    df["message_count_first_30min"] = np.random.poisson(
        1.2 + df["urgency_score"] * 2.3
    ).clip(1, 15)

    # -----------------------------
    # 8. Agent interventions
    # -----------------------------
    # This is intentionally confounded.
    # Higher-risk tickets are more likely to get senior routing.
    senior_routing_prob = sigmoid(
        -2.0
        + 1.8 * df["urgency_score"]
        + 1.2 * (df["priority_initial"] == "high").astype(int)
        + 1.2 * (df["issue_category"] == "safety_issue").astype(int)
        + 0.8 * (df["issue_category"] == "host_cancellation").astype(int)
    )

    df["senior_agent_routing"] = np.random.binomial(1, senior_routing_prob)

    coupon_prob = sigmoid(
        -2.4
        + 1.5 * (df["issue_category"] == "host_cancellation").astype(int)
        + 0.9 * df["same_week_checkin_flag"]
        + 0.5 * df["high_value_booking_flag"]
    )

    df["rebooking_coupon_offered"] = np.random.binomial(1, coupon_prob)

    # -----------------------------
    # 9. Outcomes
    # -----------------------------
    baseline_repeat_risk = sigmoid(
        -2.2
        + 1.7 * df["urgency_score"]
        + 0.9 * (df["sentiment_score"] < -0.45).astype(int)
        + 0.5 * df["same_week_checkin_flag"]
        + 1.0 * (df["issue_category"] == "safety_issue").astype(int)
        + 0.8 * (df["issue_category"] == "host_cancellation").astype(int)
        + 0.4 * df["strict_policy_flag"]
        + 2.8 * df["host_cancellation_rate_12m"]
        + 0.5 * (df["host_response_rate"] < 0.75).astype(int)
        + 0.5 * (df["guest_past_support_tickets_12m"] >= 3).astype(int)
    )

    # Treatment effect:
    # senior routing and coupons reduce repeat contact for the right cases.
    repeat_risk_after_treatment = (
        baseline_repeat_risk
        - 0.10 * df["senior_agent_routing"]
        - 0.08 * df["rebooking_coupon_offered"] * (df["issue_category"] == "host_cancellation").astype(int)
        - 0.04 * df["rebooking_coupon_offered"] * df["same_week_checkin_flag"]
    )

    df["repeat_contact_probability"] = repeat_risk_after_treatment.clip(0.01, 0.95).round(3)
    df["repeat_contact_7d"] = np.random.binomial(1, df["repeat_contact_probability"])

    escalation_prob = sigmoid(
        -2.5
        + 2.0 * df["repeat_contact_probability"]
        + 1.4 * (df["issue_category"] == "safety_issue").astype(int)
        + 0.8 * df["repeat_contact_7d"]
        + 0.4 * (df["time_to_first_agent_response_minutes"] > 90).astype(int)
    )

    df["escalation_probability"] = escalation_prob.round(3)
    df["escalated"] = np.random.binomial(1, escalation_prob)

    df["resolution_hours"] = np.random.gamma(2.2, 4, n_tickets)

    df["resolution_hours"] *= (
        1
        + 1.2 * df["escalated"]
        + 0.7 * df["repeat_contact_7d"]
        + 0.5 * (df["issue_category"] == "safety_issue").astype(int)
        - 0.10 * df["senior_agent_routing"]
    )

    df["resolution_hours"] = df["resolution_hours"].clip(0.1, 240).round(2)

    csat = (
        4.6
        - 1.0 * df["repeat_contact_7d"]
        - 0.6 * df["escalated"]
        - 0.4 * (df["resolution_hours"] > 24).astype(int)
        + 0.2 * df["senior_agent_routing"]
        + 0.15 * df["rebooking_coupon_offered"]
        + np.random.normal(0, 0.45, n_tickets)
    )

    df["csat_score"] = np.round(csat).clip(1, 5).astype(int)
    df["poor_csat"] = (df["csat_score"] <= 2).astype(int)
    df["long_resolution"] = (df["resolution_hours"] > 24).astype(int)

    # -----------------------------
    # 10. Save tables
    # -----------------------------
    guests.to_csv(output_dir / "dim_guest.csv", index=False)
    hosts.to_csv(output_dir / "dim_host.csv", index=False)
    listings.to_csv(output_dir / "dim_listing.csv", index=False)
    bookings.to_csv(output_dir / "fact_booking.csv", index=False)
    tickets.to_csv(output_dir / "fact_support_ticket.csv", index=False)
    df.to_csv(output_dir / "mart_ticket_modeling.csv", index=False)

    print("Data generated successfully.")
    print(f"Rows in modeling table: {len(df):,}")
    print(f"Saved to: {output_dir.resolve()}")

    return df


if __name__ == "__main__":
    generate_airbnb_support_data()

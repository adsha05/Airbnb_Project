"""Entry point for generating synthetic Airbnb support-inference data."""

from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    """Ensure the synthetic-data output directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()

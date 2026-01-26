"""
Chronological data splitting for time-series user feedback.

This module provides functions to split user timeline data into train/dev sets
while maintaining strict chronological order to prevent data leakage.
"""

from typing import Dict, List
from data_schema import FeedbackItem, UserTimeline


class DataSplit:
    """Container for train/dev data splits"""

    def __init__(
        self,
        train: List[FeedbackItem],
        dev: List[FeedbackItem]
    ):
        self.train = train
        self.dev = dev

    def __repr__(self) -> str:
        return f"DataSplit(train={len(self.train)}, dev={len(self.dev)})"

    def to_dict(self) -> Dict[str, List[FeedbackItem]]:
        """Convert to dictionary format"""
        return {
            "train": self.train,
            "dev": self.dev
        }


def split_user_data(
    user_timeline: UserTimeline,
    train_ratio: float = 0.8
) -> DataSplit:
    """
    Split user timeline data chronologically into train/dev sets.

    Logic:
    - Strictly chronological to prevent data leakage
    - Train Set (D_train): Earliest data (e.g., Days 1 to T-1)
      Used for generating candidate Prompts via Verbalizer
    - Dev Set (D_dev): Latest data (e.g., Day T)
      Used to re-rank and select the best Prompt A*, and for evaluation

    Args:
        user_timeline: UserTimeline object with chronologically sorted feedback items
        train_ratio: Proportion of data for training (default: 0.8, dev gets remaining 0.2)

    Returns:
        DataSplit object containing train and dev lists

    Raises:
        ValueError: If user_timeline is empty or has only 1 item
    """
    feedback_items = user_timeline.feedback_items
    n = len(feedback_items)

    if n == 0:
        raise ValueError("Cannot split empty user timeline")

    if n == 1:
        raise ValueError(
            "Cannot split timeline with only 1 item. Need at least 2 items for train/dev split."
        )

    # Calculate split index (ensure at least 1 item in each set)
    train_end = max(1, int(n * train_ratio))
    train_end = min(train_end, n - 1)  # Ensure at least 1 item in dev

    return DataSplit(
        train=feedback_items[:train_end],
        dev=feedback_items[train_end:]
    )


def split_multiple_users(
    user_timelines: List[UserTimeline],
    train_ratio: float = 0.8
) -> Dict[str, DataSplit]:
    """
    Split data for multiple users.

    Args:
        user_timelines: List of UserTimeline objects
        train_ratio: Proportion of data for training (default: 0.8)

    Returns:
        Dictionary mapping user_name to DataSplit
    """
    results = {}

    for timeline in user_timelines:
        try:
            split = split_user_data(timeline, train_ratio=train_ratio)
            results[timeline.user_name] = split
        except ValueError as e:
            print(f"Warning: Skipping {timeline.user_name}: {e}")

    return results


def print_split_summary(split: DataSplit, user_name: str = "User") -> None:
    """
    Print a summary of the data split.

    Args:
        split: DataSplit object
        user_name: Name of the user (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"Data Split Summary for {user_name}")
    print(f"{'='*60}")
    print(f"Train Set: {len(split.train)} interactions")
    if split.train:
        print(f"  Date range: {split.train[0].timestamp} to {split.train[-1].timestamp}")

    print(f"\nDev Set: {len(split.dev)} interactions")
    if split.dev:
        print(f"  Date range: {split.dev[0].timestamp} to {split.dev[-1].timestamp}")
    print(f"{'='*60}\n")

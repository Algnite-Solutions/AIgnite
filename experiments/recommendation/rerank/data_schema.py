"""
Data schema for user feedback and timeline management.

This module defines Pydantic models representing user interactions with papers:
- FeedbackItem: A single interaction at a specific timestamp
- UserTimeline: A chronologically sorted collection of FeedbackItems for a user
"""

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class FeedbackItem(BaseModel):
    """
    Represents a single user interaction with a set of recommended papers.

    Attributes:
        timestamp: Time of the interaction (t_j in the mathematical notation)
        user_name: Name/ID of the user
        query_context: User interest profile/query at that time (q_j)
        candidate_set: List of paper IDs shown to the user (P_j)
        labels: Dictionary mapping paper IDs to feedback labels (+1: liked, -1: disliked, 0: viewed but neutral)
        search_strategy: The search strategy used (e.g., "vector", "hybrid")
        top_k_ids: The top-k papers shown to the user (subset of candidate_set)
    """
    timestamp: datetime = Field(description="Time of the interaction")
    user_name: str = Field(description="User identifier")
    query_context: str = Field(description="User query or interest profile")
    candidate_set: List[str] = Field(description="List of paper IDs shown")
    labels: Dict[str, int] = Field(
        description="Mapping of paper_id to feedback: +1 (liked), -1 (disliked), 0 (viewed neutral)"
    )
    search_strategy: str = Field(default="vector", description="Search strategy used")
    top_k_ids: Optional[List[str]] = Field(default=None, description="Top-k papers shown")

    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure labels are in {-1, 0, +1}"""
        for paper_id, label in v.items():
            if label not in {-1, 0, 1}:
                raise ValueError(f"Label for {paper_id} must be -1, 0, or +1, got {label}")
        return v

    @classmethod
    def from_jsonl_entry(cls, entry: dict) -> "FeedbackItem":
        """
        Create a FeedbackItem from a JSONL entry.

        Data format:
        - viewed: list of true/false booleans
        - liked: list of null/true/false values

        Label mapping:
        - liked = true -> +1 (positive feedback)
        - liked = false -> -1 (negative feedback)
        - liked = null AND viewed = true -> 0 (viewed but neutral)
        - liked = null AND viewed = false -> no label (no interaction)

        Args:
            entry: Dictionary with fields: user_name, query, date, retrieved_ids,
                   top_k_ids, viewed, liked

        Returns:
            FeedbackItem instance
        """
        # Parse the timestamp
        timestamp = datetime.fromisoformat(entry['date'])

        # Build labels dictionary from viewed and liked lists
        labels = {}
        retrieved_ids = entry['retrieved_ids']
        top_k_ids = entry.get('top_k_ids', retrieved_ids[:15])
        viewed = entry.get('viewed', [])
        liked = entry.get('liked', [])

        # Ensure viewed and liked have same length as top_k_ids
        if len(viewed) < len(top_k_ids):
            viewed = viewed + [False] * (len(top_k_ids) - len(viewed))
        if len(liked) < len(top_k_ids):
            liked = liked + [None] * (len(top_k_ids) - len(liked))

        # Build labels based on the feedback data
        for i, paper_id in enumerate(top_k_ids):
            if i < len(liked) and liked[i] is not None:
                # liked = true -> +1, liked = false -> -1
                labels[paper_id] = 1 if liked[i] else -1
            elif i < len(viewed) and viewed[i]:
                # viewed = true but liked = null -> 0 (neutral/viewed)
                labels[paper_id] = 0
            # viewed = false and liked = null -> no label (no interaction)

        return cls(
            timestamp=timestamp,
            user_name=entry['user_name'],
            query_context=entry['query'],
            candidate_set=retrieved_ids,
            labels=labels,
            search_strategy=entry.get('search_strategy', 'vector'),
            top_k_ids=top_k_ids
        )

    def get_positive_papers(self) -> List[str]:
        """Return paper IDs with positive feedback (+1)"""
        return [pid for pid, label in self.labels.items() if label == 1]

    def get_negative_papers(self) -> List[str]:
        """Return paper IDs with negative feedback (-1)"""
        return [pid for pid, label in self.labels.items() if label == -1]

    def get_viewed_papers(self) -> List[str]:
        """Return paper IDs that were viewed (any label)"""
        return list(self.labels.keys())


class UserTimeline(BaseModel):
    """
    Represents a chronologically sorted collection of user interactions.

    Attributes:
        user_name: User identifier
        feedback_items: List of FeedbackItems sorted by timestamp
    """
    user_name: str = Field(description="User identifier")
    feedback_items: List[FeedbackItem] = Field(
        description="List of feedback items sorted chronologically"
    )

    @field_validator('feedback_items')
    @classmethod
    def validate_chronological_order(cls, v: List[FeedbackItem]) -> List[FeedbackItem]:
        """Ensure feedback items are sorted chronologically"""
        if len(v) > 1:
            for i in range(len(v) - 1):
                if v[i].timestamp > v[i + 1].timestamp:
                    raise ValueError("Feedback items must be sorted chronologically")
        return v

    @classmethod
    def from_jsonl_entries(cls, entries: List[dict]) -> "UserTimeline":
        """
        Create a UserTimeline from a list of JSONL entries.

        Args:
            entries: List of dictionaries representing user interactions

        Returns:
            UserTimeline instance with sorted feedback items
        """
        if not entries:
            raise ValueError("Cannot create UserTimeline from empty entries list")

        user_name = entries[0]['user_name']

        # Convert all entries to FeedbackItems
        feedback_items = [FeedbackItem.from_jsonl_entry(entry) for entry in entries]

        # Sort by timestamp
        feedback_items.sort(key=lambda x: x.timestamp)

        return cls(user_name=user_name, feedback_items=feedback_items)

    def __len__(self) -> int:
        """Return number of feedback items"""
        return len(self.feedback_items)

    def get_date_range(self) -> tuple[datetime, datetime]:
        """Return the earliest and latest timestamps"""
        if not self.feedback_items:
            raise ValueError("No feedback items in timeline")
        return self.feedback_items[0].timestamp, self.feedback_items[-1].timestamp

    def get_all_positive_papers(self) -> List[str]:
        """Return all paper IDs with positive feedback across the timeline"""
        positive_papers = []
        for item in self.feedback_items:
            positive_papers.extend(item.get_positive_papers())
        return positive_papers

    def get_all_negative_papers(self) -> List[str]:
        """Return all paper IDs with negative feedback across the timeline"""
        negative_papers = []
        for item in self.feedback_items:
            negative_papers.extend(item.get_negative_papers())
        return negative_papers

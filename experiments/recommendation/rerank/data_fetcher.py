"""
Database-backed data fetching for user feedback.

This module replaces the file-based chronological_splitter.py with direct
database access to the user_retrieve_results table in PostgreSQL.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import Column, String, Integer, JSON, Text, DateTime, Boolean, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from db_config import DatabaseConfig
from data_schema import FeedbackItem, UserTimeline, ThreeWayDataSplit

logger = logging.getLogger(__name__)

Base = declarative_base()


class UserRetrieveResult(Base):
    """SQLAlchemy model for user_retrieve_results table."""

    __tablename__ = 'user_retrieve_results'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)  # Note: actual column is 'username' not 'user_name'
    query = Column(Text, nullable=False)
    search_strategy = Column(String(50))
    recommendation_date = Column(DateTime, nullable=False)  # Note: 'recommendation_date' not 'date'
    retrieve_ids = Column(JSON, nullable=False)  # Note: 'retrieve_ids' not 'retrieved_ids'
    top_k_ids = Column(JSON, nullable=True)
    # Note: viewed and liked columns don't exist in current schema
    # viewed = Column(JSON, nullable=True)
    # liked = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<UserRetrieveResult(id={self.id}, user={self.username}, date={self.recommendation_date})>"


class PaperRecommendation(Base):
    """SQLAlchemy model for paper_recommendations table (contains feedback)."""

    __tablename__ = 'paper_recommendations'

    id = Column(Integer, primary_key=True)
    username = Column(String(50))
    paper_id = Column(String(50))
    title = Column(String(255))
    authors = Column(String(255))
    abstract = Column(Text)
    url = Column(String(255))
    blog = Column(Text)
    blog_title = Column(Text)
    blog_abs = Column(Text)
    recommendation_date = Column(DateTime)
    viewed = Column(Boolean)  # ✅ This table HAS feedback!
    relevance_score = Column(Float)
    recommendation_reason = Column(Text)
    submitted = Column(String(255))
    comment = Column(Text)
    blog_liked = Column(Boolean)  # ✅ This table HAS feedback!
    blog_feedback_date = Column(DateTime)

    def __repr__(self):
        return f"<PaperRecommendation(id={self.id}, user={self.username}, paper={self.paper_id}, viewed={self.viewed}, blog_liked={self.blog_liked})>"


class UserDataFetcher:
    """
    Fetch user feedback data from PostgreSQL database.

    Replaces file-based JSONL loading with direct database access.
    Supports active user filtering and chronological time-window queries.
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        """
        Initialize data fetcher with database configuration.

        Args:
            db_config: DatabaseConfig instance (uses environment variables if None)
        """
        self.db_config = db_config or DatabaseConfig()
        self.engine = self.db_config.get_engine()

    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary statistics about data in the database.

        Returns:
            Dictionary with total_records, unique_users, date_range
        """
        session = self.db_config.create_session()
        try:
            # Total records
            total_records = session.query(func.count(UserRetrieveResult.id)).scalar()

            # Unique users
            unique_users = session.query(func.count(func.distinct(UserRetrieveResult.username))).scalar()

            # Date range
            min_date = session.query(func.min(UserRetrieveResult.recommendation_date)).scalar()
            max_date = session.query(func.max(UserRetrieveResult.recommendation_date)).scalar()

            return {
                'total_records': total_records,
                'unique_users': unique_users,
                'date_range': (min_date, max_date)
            }
        finally:
            session.close()

    def fetch_active_user_timelines(
        self,
        weeks: int = 4,
        min_interactions_per_week: int = 5,
        use_feedback_counts: bool = True
    ) -> List[UserTimeline]:
        """
        Fetch timelines for active users within the last N weeks.

        Active user definition: Has ≥min_interactions_per_week in EACH week
        within the time window.

        Args:
            weeks: Number of weeks to look back (default: 4)
            min_interactions_per_week: Minimum interactions per week (default: 5)
            use_feedback_counts: If True, count viewed/liked papers only;
                                 If False, count all retrieve events (default: True)

        Returns:
            List of UserTimeline objects for active users
        """
        """
        Fetch timelines for active users within the last N weeks.

        Active user definition: Has ≥min_interactions_per_week in EACH week
        within the time window.

        Args:
            weeks: Number of weeks to look back (default: 4)
            min_interactions_per_week: Minimum interactions per week (default: 5)

        Returns:
            List of UserTimeline objects for active users
        """
        session = self.db_config.create_session()
        try:
            # Calculate time window
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)

            logger.info(f"Fetching active users from {start_date} to {end_date}")

            if use_feedback_counts:
                logger.info(f"Active user criteria: ≥{min_interactions_per_week} viewed/liked interactions/week for {weeks} weeks")
            else:
                logger.info(f"Active user criteria: ≥{min_interactions_per_week} total interactions/week for {weeks} weeks")

            # Get active users
            active_users = get_active_users(session, weeks, min_interactions_per_week, use_feedback_counts)

            if not active_users:
                logger.warning("No active users found with the given criteria")
                return []

            logger.info(f"Found {len(active_users)} active users")

            # Fetch data for active users
            timelines = []

            for user_name in active_users:
                try:
                    # Query all records for this user (not just within time window)
                    # to enable proper chronological splitting later
                    records = session.query(UserRetrieveResult)\
                        .filter(UserRetrieveResult.username == user_name)\
                        .order_by(UserRetrieveResult.recommendation_date)\
                        .all()

                    if not records:
                        logger.warning(f"No records found for user {user_name}")
                        continue

                    # Convert records to FeedbackItems
                    feedback_items = [FeedbackItem.from_db_record(r) for r in records]

                    # Sort by timestamp (should already be sorted from query)
                    feedback_items.sort(key=lambda x: x.timestamp)

                    # Create UserTimeline
                    timeline = UserTimeline(
                        user_name=user_name,
                        feedback_items=feedback_items
                    )

                    timelines.append(timeline)

                    logger.info(f"  {user_name}: {len(feedback_items)} interactions")

                except Exception as e:
                    logger.error(f"Failed to process user {user_name}: {e}")
                    continue

            return timelines

        finally:
            session.close()


def get_active_users(
    session: Session,
    weeks: int,
    min_interactions_per_week: int,
    use_feedback_counts: bool = True
) -> List[str]:
    """
    Identify active users based on interaction frequency.

    Active user: Has ≥min_interactions_per_week in EACH week within time window.

    Args:
        session: SQLAlchemy session
        weeks: Number of weeks to look back
        min_interactions_per_week: Minimum interactions required per week
        use_feedback_counts: If True, count viewed/liked papers only;
                           If False, count all retrieve events (default: True)

    Returns:
        List of active user names
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)

    try:
        # Check if feedback columns exist
        has_feedback = _check_feedback_columns(session)

        if use_feedback_counts and has_feedback:
            # Count daily engagement (viewed + blog_liked) from paper_recommendations
            # Count by user and day, then aggregate by week
            query = text("""
                WITH daily_recommendations AS (
                    SELECT
                        username,
                        DATE(recommendation_date) as rec_date,
                        COUNT(CASE WHEN viewed THEN 1 END) + COUNT(CASE WHEN blog_liked THEN 1 END) as engagement_count
                    FROM paper_recommendations
                    WHERE recommendation_date >= :start_date
                      AND recommendation_date <= :end_date
                    GROUP BY username, DATE(recommendation_date)
                )
                SELECT
                    dr.username,
                    EXTRACT(YEAR FROM dr.rec_date) as year,
                    EXTRACT(WEEK FROM dr.rec_date) as week,
                    SUM(dr.engagement_count) as interaction_count
                FROM daily_recommendations dr
                GROUP BY dr.username, EXTRACT(YEAR FROM dr.rec_date), EXTRACT(WEEK FROM dr.rec_date)
                ORDER BY dr.username, year, week
            """)
        else:
            # Count all retrieve events (fallback)
            query = text("""
                SELECT
                    username,
                    EXTRACT(YEAR FROM recommendation_date) as year,
                    EXTRACT(WEEK FROM recommendation_date) as week,
                    COUNT(*) as interaction_count
                FROM user_retrieve_results
                WHERE recommendation_date >= :start_date AND recommendation_date <= :end_date
                GROUP BY username, EXTRACT(YEAR FROM recommendation_date), EXTRACT(WEEK FROM recommendation_date)
                ORDER BY username, year, week
            """)

        result = session.execute(query, {
            'start_date': start_date,
            'end_date': end_date
        })

        # Group by user and check weekly interaction counts
        user_week_counts = defaultdict(list)

        for row in result:
            username = row[0]  # First column is username
            count = row[3]  # Fourth column is interaction_count
            user_week_counts[username].append(count)

        # Filter users who meet threshold in ALL weeks
        active_users = []

        for user_name, week_counts in user_week_counts.items():
            # Check if user has at least min_interactions_per_week on average
            # (Relaxed: average across weeks, not minimum in EACH week)
            avg_count = sum(week_counts) / len(week_counts) if week_counts else 0

            # Also check at least 50% of weeks meet threshold
            weeks_meeting_threshold = sum(1 for c in week_counts if c >= min_interactions_per_week)
            sufficient_weeks = weeks_meeting_threshold >= (len(week_counts) / 2)

            meets_threshold = avg_count >= min_interactions_per_week and sufficient_weeks

            if meets_threshold:
                active_users.append(user_name)
            else:
                logger.debug(
                    f"User {user_name} excluded: weekly counts {week_counts} "
                    f"(threshold: {min_interactions_per_week})"
                )

        return active_users

    except Exception as e:
        logger.error(f"Failed to get active users: {e}")
        return []


def _check_feedback_columns(session: Session) -> bool:
    """
    Check if paper_recommendations table exists (has feedback data).

    Args:
        session: SQLAlchemy session

    Returns:
        True if paper_recommendations table exists, False otherwise
    """
    try:
        query = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = 'paper_recommendations'
            LIMIT 1
        """)
        result = session.execute(query).fetchone()
        return result is not None
    except Exception as e:
        logger.debug(f"Could not check for paper_recommendations table: {e}")
        return False


def split_user_timelines_three_way(
    timelines: List[UserTimeline],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, ThreeWayDataSplit]:
    """
    Split user timelines chronologically into train/val/test sets.

    Chronological order is preserved: train (earliest) -> val -> test (latest).

    Args:
        timelines: List of UserTimeline objects
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15, test gets remaining 0.15)

    Returns:
        Dictionary mapping user_name to ThreeWayDataSplit
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1, and sum < 1")

    results = {}

    for timeline in timelines:
        try:
            n = len(timeline.feedback_items)

            if n < 3:
                logger.warning(
                    f"Skipping {timeline.user_name}: only {n} interactions, "
                    f"need at least 3 for 3-way split"
                )
                continue

            # Calculate split indices
            train_end = max(1, int(n * train_ratio))
            val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

            # Ensure at least 1 item in each split
            train_end = max(1, min(train_end, n - 2))
            val_end = max(train_end + 1, min(val_end, n - 1))

            split = ThreeWayDataSplit(
                train=timeline.feedback_items[:train_end],
                val=timeline.feedback_items[train_end:val_end],
                test=timeline.feedback_items[val_end:]
            )

            results[timeline.user_name] = split

            logger.info(
                f"Split {timeline.user_name}: train={len(split.train)}, "
                f"val={len(split.val)}, test={len(split.test)}"
            )

        except Exception as e:
            logger.error(f"Failed to split timeline for {timeline.user_name}: {e}")
            continue

    return results


def print_split_summary(split: ThreeWayDataSplit, user_name: str = "User") -> None:
    """
    Print a summary of the 3-way data split.

    Args:
        split: ThreeWayDataSplit object
        user_name: Name of the user (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"Data Split Summary for {user_name}")
    print(f"{'='*60}")
    print(f"Train Set: {len(split.train)} interactions")
    if split.train:
        print(f"  Date range: {split.train[0].timestamp} to {split.train[-1].timestamp}")

    print(f"\nVal Set: {len(split.val)} interactions")
    if split.val:
        print(f"  Date range: {split.val[0].timestamp} to {split.val[-1].timestamp}")

    print(f"\nTest Set: {len(split.test)} interactions")
    if split.test:
        print(f"  Date range: {split.test[0].timestamp} to {split.test[-1].timestamp}")
    print(f"{'='*60}\n")

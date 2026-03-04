"""
Database configuration for user feedback data fetching.

This module provides database configuration management using environment variables
for secure credential handling.
"""

import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """
    Database configuration manager for PostgreSQL connection.

    Loads credentials from environment variables and provides SQLAlchemy
    engine and session factory for database operations.

    Environment Variables:
        DB_HOST: Database host address
        DB_PORT: Database port (default: 5432)
        DB_USER: Database username
        DB_PASSWORD: Database password
        DB_NAME: Database name (default: paperignition_user)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize database configuration from environment variables or parameters.

        Args:
            host: Database host (overrides DB_HOST env var)
            port: Database port (overrides DB_PORT env var)
            user: Database user (overrides DB_USER env var)
            password: Database password (overrides DB_PASSWORD env var)
            database: Database name (overrides DB_NAME env var)
        """
        self.host = host or os.getenv("DB_HOST")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.user = user or os.getenv("DB_USER")
        self.password = password or os.getenv("DB_PASSWORD")
        self.database = database or os.getenv("DB_NAME", "paperignition_user")

        # Validate required fields
        if not all([self.host, self.user, self.password, self.database]):
            missing = []
            if not self.host:
                missing.append("DB_HOST")
            if not self.user:
                missing.append("DB_USER")
            if not self.password:
                missing.append("DB_PASSWORD")
            if not self.database:
                missing.append("DB_NAME")

            raise ValueError(
                f"Missing required database credentials: {', '.join(missing)}. "
                f"Set environment variables or pass parameters."
            )

        self._engine = None
        self._Session = None

    @property
    def connection_string(self) -> str:
        """
        Build SQLAlchemy connection string.

        Returns:
            PostgreSQL connection URL
        """
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            self._engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False            # Set to True for SQL query logging
            )
            logger.info(f"Created database engine for {self.database}@{self.host}")
        return self._engine

    def get_session_factory(self):
        """
        Get or create SQLAlchemy session factory.

        Returns:
            sessionmaker configured with the engine
        """
        if self._Session is None:
            self._Session = sessionmaker(bind=self.get_engine())
        return self._Session

    def create_session(self):
        """
        Create a new database session.

        Returns:
            SQLAlchemy Session instance
        """
        Session = self.get_session_factory()
        return Session()

    def __repr__(self) -> str:
        """String representation (hides password)"""
        return f"DatabaseConfig(host={self.host}, port={self.port}, user={self.user}, database={self.database})"

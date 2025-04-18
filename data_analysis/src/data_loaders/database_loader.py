"""
Database Loader Module.

This module provides a data loader for SQL databases with support for
various database backends, query execution, and efficient data retrieval.
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Iterator, Callable, Tuple

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from urllib.parse import quote_plus

from .base_loader import BaseDataLoader

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseLoader(BaseDataLoader):
    """
    Data loader for SQL databases.
    
    This class provides methods for connecting to databases, executing
    queries, and loading data into pandas DataFrames, with support for
    various database backends like SQLite, PostgreSQL, MySQL, etc.
    """
    
    def __init__(self, connection_string: Optional[str] = None, 
                 engine: Optional[sqlalchemy.engine.Engine] = None,
                 verbose: bool = True):
        """
        Initialize the database loader.
        
        Args:
            connection_string: SQLAlchemy connection string for the database.
                Can be provided later via connect() if not provided here.
            engine: An existing SQLAlchemy engine. If provided, connection_string is ignored.
            verbose: Whether to display progress bars and detailed logging.
        """
        super().__init__(verbose=verbose)
        self._engine = None
        self._metadata = None
        
        if engine:
            self._engine = engine
            logger.info(f"Using provided database engine: {self._engine.url}")
            self._metadata = MetaData()
            self._metadata.reflect(bind=self._engine)
        elif connection_string:
            self.connect(connection_string)
    
    def connect(self, connection_string: str) -> None:
        """
        Connect to a database using a connection string.
        
        Args:
            connection_string: SQLAlchemy connection string for the database.
                For example:
                - SQLite: 'sqlite:///path/to/database.db'
                - PostgreSQL: 'postgresql://username:password@localhost:5432/database'
                - MySQL: 'mysql+pymysql://username:password@localhost:3306/database'
                
        Raises:
            ValueError: If connection fails or if connection string is invalid.
        """
        try:
            logger.info(f"Connecting to database with connection string: {self._mask_sensitive_info(connection_string)}")
            self._engine = create_engine(connection_string)
            
            # Test the connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Reflect the database schema
            self._metadata = MetaData()
            self._metadata.reflect(bind=self._engine)
            
            logger.info(f"Successfully connected to database: {self._engine.url.database}")
            
        except Exception as e:
            self.log_error("Error connecting to database", e)
            self._engine = None
            self._metadata = None
            raise ValueError(f"Failed to connect to database: {str(e)}")
    
    def _mask_sensitive_info(self, connection_string: str) -> str:
        """
        Mask sensitive information in a connection string for logging.
        
        Args:
            connection_string: The connection string to mask.
                
        Returns:
            Masked connection string with password replaced by '***'.
        """
        # Very basic masking - should work for common connection strings
        # but may not cover all edge cases
        if '://' in connection_string:
            parts = connection_string.split('://')
            if '@' in parts[1] and ':' in parts[1].split('@')[0]:
                user_pass = parts[1].split('@')[0]
                if ':' in user_pass:
                    user = user_pass.split(':')[0]
                    masked = f"{parts[0]}://{user}:***@{parts[1].split('@')[1]}"
                    return masked
        
        # For simple connection strings or those that don't match the pattern
        return connection_string.replace("&password=", "&password=***").replace("password=", "password=***")
    
    def load_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string to execute.
            params: Parameters to pass to the query for parameterized queries.
                
        Returns:
            A pandas DataFrame containing the query results.
            
        Raises:
            ValueError: If no database connection has been established.
            Exception: If the query execution fails.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            logger.info(f"Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Log parameters with sensitive values masked
            if params and self.verbose:
                masked_params = {k: '***' if k.lower() in ['password', 'key', 'secret', 'token'] else v 
                                for k, v in params.items()}
                logger.info(f"Query parameters: {masked_params}")
            
            # Execute the query
            if params:
                df = pd.read_sql_query(text(query), self._engine, params=params)
            else:
                df = pd.read_sql_query(text(query), self._engine)
            
            logger.info(f"Query returned {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.log_error(f"Error executing query: {query[:100]}{'...' if len(query) > 100 else ''}", e)
            raise
    
    def load_table(self, table_name: str, schema: Optional[str] = None,
                  columns: Optional[List[str]] = None, 
                  where_clause: Optional[str] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from a database table into a pandas DataFrame.
        
        Args:
            table_name: Name of the database table to load.
            schema: Database schema name (if applicable).
            columns: List of column names to select. If None, selects all columns.
            where_clause: WHERE clause for filtering rows (without the 'WHERE' keyword).
            limit: Maximum number of rows to return.
                
        Returns:
            A pandas DataFrame containing the table data.
            
        Raises:
            ValueError: If the table does not exist or if no connection has been established.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            # Build the query
            cols_str = ", ".join(columns) if columns else "*"
            query = f"SELECT {cols_str} FROM "
            
            if schema:
                query += f"{schema}."
                
            query += f"{table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
                
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Loading data from table: {table_name}")
            return self.load_query(query)
            
        except Exception as e:
            self.log_error(f"Error loading table: {table_name}", e)
            raise
    
    def load_tables_as_dict(self, table_names: List[str], 
                           schema: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load multiple tables into a dictionary of DataFrames.
        
        Args:
            table_names: List of table names to load.
            schema: Database schema name (if applicable).
                
        Returns:
            Dictionary mapping table names to their respective DataFrames.
        """
        result = {}
        
        for table_name in table_names:
            try:
                result[table_name] = self.load_table(table_name, schema)
                logger.info(f"Loaded table '{table_name}' successfully")
            except Exception as e:
                logger.error(f"Error loading table '{table_name}': {str(e)}")
                result[table_name] = pd.DataFrame()  # Empty DataFrame for failed tables
        
        return result
    
    def load_query_iterator(self, query: str, 
                           chunk_size: int = 10000,
                           params: Optional[Dict[str, Any]] = None) -> Iterator[pd.DataFrame]:
        """
        Create an iterator for loading query results in chunks.
        
        Args:
            query: SQL query string to execute.
            chunk_size: Number of rows to fetch per chunk.
            params: Parameters to pass to the query for parameterized queries.
                
        Yields:
            Chunks of the query results as pandas DataFrames.
            
        Raises:
            ValueError: If no database connection has been established.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            logger.info(f"Creating query iterator with chunk size {chunk_size}")
            
            # Execute the query with chunking
            if params:
                chunks = pd.read_sql_query(text(query), self._engine, 
                                         params=params, chunksize=chunk_size)
            else:
                chunks = pd.read_sql_query(text(query), self._engine, 
                                         chunksize=chunk_size)
            
            # Yield chunks
            for i, chunk in enumerate(chunks):
                logger.info(f"Yielding chunk {i+1} with {len(chunk)} rows")
                yield chunk
                
        except Exception as e:
            self.log_error(f"Error in query iterator: {query[:100]}{'...' if len(query) > 100 else ''}", e)
            raise
    
    def load_query_streaming(self, query: str, 
                            chunk_size: int = 10000,
                            params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL query in chunks and combine the results.
        
        Args:
            query: SQL query string to execute.
            chunk_size: Number of rows to fetch per chunk.
            params: Parameters to pass to the query for parameterized queries.
                
        Returns:
            A pandas DataFrame containing all the query results.
            
        Raises:
            ValueError: If no database connection has been established.
        """
        chunks = []
        
        # Initialize progress if the connection contains row count info
        total_rows = self._estimate_query_rows(query)
        progress_bar = self.create_progress_bar(total_rows or 0, "Loading data in chunks")
        
        try:
            # Process each chunk
            for chunk in self.load_query_iterator(query, chunk_size, params):
                chunks.append(chunk)
                
                # Update progress
                if progress_bar:
                    self.update_progress(progress_bar, len(chunk))
                
            # Combine all chunks
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded all data with {len(result)} rows")
                return result
            else:
                logger.warning("Query returned no data")
                return pd.DataFrame()
                
        except Exception as e:
            self.log_error("Error loading data in streaming mode", e)
            raise
            
        finally:
            if progress_bar:
                self.close_progress(progress_bar)
    
    def _estimate_query_rows(self, query: str) -> Optional[int]:
        """
        Attempt to estimate the number of rows that will be returned by a query.
        This is used for progress bars and is a best-effort estimation.
        
        Args:
            query: SQL query string to analyze.
                
        Returns:
            Estimated number of rows or None if estimation is not possible.
        """
        try:
            # Extract table name from query (very basic implementation)
            from_parts = query.lower().split(' from ')
            if len(from_parts) < 2:
                return None
                
            table_parts = from_parts[1].strip().split(' ')
            if not table_parts:
                return None
                
            table_name = table_parts[0].strip().strip(';').strip('`"\'')
            
            # If we can identify the table, try to get its row count
            with self._engine.connect() as conn:
                # Different databases use different ways to get row counts
                if 'sqlite' in self._engine.url.drivername:
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    return res.scalar()
                elif 'postgres' in self._engine.url.drivername:
                    res = conn.execute(text(f"""
                        SELECT reltuples::bigint AS estimate 
                        FROM pg_class 
                        WHERE relname = '{table_name}'
                    """))
                    return res.scalar()
                elif 'mysql' in self._engine.url.drivername:
                    res = conn.execute(text(f"""
                        SELECT table_rows 
                        FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    """))
                    return res.scalar()
                    
            return None
            
        except Exception as e:
            logger.debug(f"Could not estimate query rows: {str(e)}")
            return None
    
    def get_table_schema(self, table_name: str, 
                        schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed schema information for a specific table.
        
        Args:
            table_name: Name of the table.
            schema: Schema name (if applicable).
                
        Returns:
            Dictionary containing column definitions and other table metadata.
            
        Raises:
            ValueError: If the table does not exist or if no connection has been established.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            # Get the inspector
            inspector = inspect(self._engine)
            
            # Get column information
            columns = inspector.get_columns(table_name, schema)
            
            # Get primary key information
            pk = inspector.get_primary_keys(table_name, schema)
            
            # Get foreign key information
            fk = inspector.get_foreign_keys(table_name, schema)
            
            # Get index information
            indexes = inspector.get_indexes(table_name, schema)
            
            result = {
                'table_name': table_name,
                'schema': schema,
                'columns': columns,
                'primary_key': pk,
                'foreign_keys': fk,
                'indexes': indexes
            }
            
            logger.info(f"Retrieved schema for table '{table_name}'")
            return result
            
        except Exception as e:
            self.log_error(f"Error getting schema for table: {table_name}", e)
            raise
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        List all tables in the database or within a specific schema.
        
        Args:
            schema: Schema name to list tables from (if applicable).
                
        Returns:
            List of table names.
            
        Raises:
            ValueError: If no database connection has been established.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            inspector = inspect(self._engine)
            tables = inspector.get_table_names(schema)
            
            logger.info(f"Found {len(tables)} tables{f' in schema {schema}' if schema else ''}")
            return tables
            
        except Exception as e:
            self.log_error(f"Error listing tables{f' in schema {schema}' if schema else ''}", e)
            raise
    
    def list_schemas(self) -> List[str]:
        """
        List all schemas in the database.
        
        Returns:
            List of schema names.
            
        Raises:
            ValueError: If no database connection has been established.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            inspector = inspect(self._engine)
            schemas = inspector.get_schema_names()
            
            logger.info(f"Found {len(schemas)} schemas")
            return schemas
            
        except Exception as e:
            self.log_error("Error listing schemas", e)
            raise
    
    def execute_query(self, query: str, 
                     params: Optional[Dict[str, Any]] = None) -> int:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE, etc.) and return affected rows.
        
        Args:
            query: SQL query string to execute.
            params: Parameters to pass to the query for parameterized queries.
                
        Returns:
            Number of affected rows.
            
        Raises:
            ValueError: If no database connection has been established.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
        
        try:
            logger.info(f"Executing non-SELECT query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            with self._engine.connect() as conn:
                with conn.begin():
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    
                    # Get affected row count if available
                    affected_rows = result.rowcount if hasattr(result, 'rowcount') else -1
                    
            logger.info(f"Query executed successfully, affected {affected_rows} rows")
            return affected_rows
            
        except Exception as e:
            self.log_error(f"Error executing query: {query[:100]}{'...' if len(query) > 100 else ''}", e)
            raise
    
    def export_to_table(self, df: pd.DataFrame, 
                       table_name: str,
                       schema: Optional[str] = None,
                       if_exists: str = 'fail',
                       index: bool = False,
                       dtype: Optional[Dict[str, Any]] = None) -> int:
        """
        Export a pandas DataFrame to a database table.
        
        Args:
            df: The DataFrame to export.
            table_name: Name of the target database table.
            schema: Target schema (if applicable).
            if_exists: What to do if the table already exists ('fail', 'replace', or 'append').
            index: Whether to include the DataFrame's index as a column.
            dtype: Dictionary mapping column names to SQLAlchemy types.
                
        Returns:
            Number of rows exported.
            
        Raises:
            ValueError: If no database connection has been established or if if_exists is invalid.
        """
        if not self._engine:
            raise ValueError("No database connection. Call connect() method first.")
            
        if if_exists not in ['fail', 'replace', 'append']:
            raise ValueError("if_exists must be one of 'fail', 'replace', or 'append'")
        
        try:
            table_path = f"{schema+'.' if schema else ''}{table_name}"
            logger.info(f"Exporting DataFrame with {len(df)} rows to table '{table_path}'")
            
            # Export the DataFrame to the database
            df.to_sql(
                name=table_name,
                con=self._engine,
                schema=schema,
                if_exists=if_exists,
                index=index,
                dtype=dtype
            )
            
            logger.info(f"Successfully exported {len(df)} rows to table '{table_path}'")
            return len(df)
            
        except Exception as e:
            self.log_error(f"Error exporting DataFrame to table '{table_name}'", e)
            raise
    
    def create_connection_string(self, dbtype: str, 
                                host: str,
                                database: str,
                                user: Optional[str] = None,
                                password: Optional[str] = None,
                                port: Optional[int] = None,
                                **kwargs) -> str:
        """
        Create a SQLAlchemy connection string based on components.
        
        Args:
            dbtype: Database type ('sqlite', 'postgresql', 'mysql', 'oracle', etc.).
            host: Database server host (or file path for SQLite).
            database: Database name.
            user: Username for connection.
            password: Password for connection.
            port: Port number.
            **kwargs: Additional connection parameters.
                
        Returns:
            A formatted connection string.
        """
        # Normalize database type
        dbtype = dbtype.lower()
        
        if dbtype == 'sqlite':
            # SQLite uses file paths
            return f"sqlite:///{host}"
        
        # For other database types
        driver_map = {
            'postgresql': 'postgresql',
            'postgres': 'postgresql',
            'mysql': 'mysql+pymysql',
            'oracle': 'oracle',
            'mssql': 'mssql+pyodbc',
            'sqlserver': 'mssql+pyodbc'
        }
        
        driver = driver_map.get(dbtype, dbtype)
        
        # Build authentication part
        auth = ""
        if user:
            auth = user
            if password:
                encoded_password = quote_plus(str(password))
                auth += f":{encoded_password}"
            auth += "@"
        
        # Build host part
        host_part = host
        if port:
            host_part += f":{port}"
        
        # Build query parameters
        query_params = ""
        if kwargs:
            param_strings = [f"{k}={quote_plus(str(v))}" for k, v in kwargs.items()]
            query_params = f"?{'&'.join(param_strings)}"
        
        # Assemble the connection string
        conn_str = f"{driver}://{auth}{host_part}/{database}{query_params}"
        
        return conn_str
    
    def close(self) -> None:
        """
        Close the database connection.
        """
        if self._engine:
            logger.info("Closing database connection")
            self._engine.dispose()
            self._engine = None
            self._metadata = None
    
    def __del__(self):
        """
        Ensure connection is closed when the object is deleted.
        """
        self.close()
    
    def __enter__(self):
        """
        Enable context manager support.
        
        Returns:
            Self reference for use in with statement.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close connection when exiting context.
        
        Args:
            exc_type: Exception type if an exception was raised in the context.
            exc_val: Exception value if an exception was raised in the context.
            exc_tb: Exception traceback if an exception was raised in the context.
        """
        self.close() 
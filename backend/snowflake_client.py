import os
from urllib.parse import quote_plus
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Force reload of .env to capture changes
load_dotenv(override=True)

class SnowflakeManager:
    @staticmethod
    def _resolve_config(config: Dict[str, str]) -> Dict[str, str]:
        """Helper to fall back to environment variables if config values are empty."""
        def get_val(key: str, env_key: str, default: str = "") -> str:
            val = config.get(key)
            if val is not None and str(val).strip() != "":
                # Strip outside quotes if accidentally included in inputs
                return str(val).strip().strip("'").strip('"')
            env_val = os.getenv(env_key, default)
            return str(env_val).strip().strip("'").strip('"')

        return {
            "account": get_val("account", "SNOWFLAKE_ACCOUNT"),
            "user": get_val("user", "SNOWFLAKE_USER"),
            "password": get_val("password", "SNOWFLAKE_PASSWORD"),
            "database": get_val("database", "SNOWFLAKE_DATABASE"),
            "schema": get_val("schema", "SNOWFLAKE_SCHEMA", "PUBLIC"),
            "warehouse": get_val("warehouse", "SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "role": get_val("role", "SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        }

    @classmethod
    def create_conn_uri(cls, config: Dict[str, str]) -> str:
        """Constructs SQLAlchemy Snowflake connection string with URL-encoded credentials."""
        resolved = cls._resolve_config(config)
        
        # URL-encode password so '#' converts to '%23'
        encoded_password = quote_plus(resolved['password'])
        
        return (
            f"snowflake://{resolved['user']}:{encoded_password}@{resolved['account']}/"
            f"{resolved['database']}/{resolved['schema']}"
            f"?warehouse={resolved['warehouse']}&role={resolved['role']}"
        )

    @classmethod
    def get_engine(cls, config: Dict[str, str]) -> Engine:
        """Creates an engine for a specific account configuration."""
        uri = cls.create_conn_uri(config)
        return create_engine(uri, pool_pre_ping=True)

    @classmethod
    def test_connection(cls, config: Dict[str, str]) -> Tuple[bool, str]:
        """Validates credentials for a given Snowflake account."""
        try:
            engine = cls.get_engine(config)
            with engine.connect() as conn:
                res = conn.execute(text("SELECT CURRENT_VERSION(), CURRENT_ACCOUNT()")).fetchone()
                if res is None:
                    return False, "Connection test returned no results."
                return True, f"Connected to Account: {res[1]} (Snowflake v{res[0]})"
        except Exception as e:
            return False, str(e)

    @classmethod
    def fetch_dynamic_schema(cls, config: Dict[str, str], max_tables: int = 10) -> str:
        """Dynamically inspects tables and columns in the active Snowflake schema."""
        schema_query = """
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = UPPER(:schema_name)
        ORDER BY table_name, ordinal_position;
        """
        try:
            resolved = cls._resolve_config(config)
            engine = cls.get_engine(config)
            with engine.connect() as conn:
                result = conn.execute(text(schema_query), {"schema_name": resolved["schema"]}).fetchall()
            
            if not result:
                return "No tables found in specified database/schema."

            schema_map: Dict[str, List[str]] = {}
            for row in result:
                tbl, col, dtype = row[0], row[1], row[2]
                if tbl not in schema_map:
                    if len(schema_map) >= max_tables:
                        continue
                    schema_map[tbl] = []
                schema_map[tbl].append(f"{col} ({dtype})")

            formatted_schema = []
            for tbl, cols in schema_map.items():
                formatted_schema.append(f"Table: {tbl}\nColumns:\n  - " + "\n  - ".join(cols))

            return "\n\n".join(formatted_schema)
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

    @classmethod
    def execute_query(cls, config: Dict[str, str], sql: str) -> Tuple[List[str], List[Any]]:
        """Executes SQL against the configured Snowflake instance."""
        engine = cls.get_engine(config)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(row) for row in result.fetchall()]
            return columns, rows
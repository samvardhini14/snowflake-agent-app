import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from snowflake_client import SnowflakeManager

# Explicitly load from backend/.env or root .env
load_dotenv(dotenv_path=os.path.join(os.getcwd(), "backend", ".env"), override=True)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

class DataAgent:
    def __init__(self, openai_api_key: str = None, model_name: str = "gpt-4o"):
        # Resolve key from parameter or environment
        resolved_key = openai_api_key if (openai_api_key and openai_api_key.strip()) else os.getenv("OPENAI_API_KEY")
        
        if not resolved_key:
            raise ValueError("Missing OpenAI API Key. Please configure OPENAI_API_KEY in your .env file.")

        base_url = os.getenv("OPENAI_BASE_URL", None)

        kwargs = {
            "model": model_name,
            "temperature": 0,
            "api_key": SecretStr(resolved_key),
        }
        if base_url:
            kwargs["base_url"] = base_url

        self.llm = ChatOpenAI(**kwargs)

    def is_read_only_sql(self, sql: str) -> bool:
        """Enforces execution safety guardrails."""
        cleaned = sql.strip().upper()
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT"]
        if not cleaned.startswith(("SELECT", "WITH")):
            return False
        return not any(re.search(rf"\b{kw}\b", cleaned) for kw in forbidden)

    def process_query(self, user_query: str, account_config: Dict[str, str]) -> Dict[str, Any]:
        """Processes queries with specialized Pharmaceutical domain context."""
        schema_context = SnowflakeManager.fetch_dynamic_schema(account_config)

        sql_prompt = f"""
        You are an expert Pharmaceutical Data Analyst and Snowflake SQL Specialist.
        Your job is to answer business and scientific questions about clinical trials, drug catalogs, adverse events, dosages, and prescription revenues.

        DOMAIN & SCHEMA MAPPING RULES:
        1. Always use UPPER() for case-insensitive text matching on drug names, therapeutic areas, or regions.
        2. Field Mappings:
           - 'Efficacy' / 'Success rate' -> EFFICACY_RATE
           - 'Adverse events' / 'Side effects' / 'Complications' -> ADVERSE_EVENTS_COUNT
           - 'Phase' / 'Trial Phase' / 'Approval' -> APPROVAL_STATUS
           - 'Revenue' / 'Sales' -> REVENUE_USD
        3. Never modify data. Generate ONLY executable Snowflake SELECT queries without markdown formatting or code blocks.

        DATABASE SCHEMA:
        {schema_context}

        USER QUESTION: {user_query}
        SQL QUERY:
        """

        response_content = self.llm.invoke(sql_prompt).content
        cleaned_sql = str(response_content).replace("```sql", "").replace("```", "").strip()

        if not self.is_read_only_sql(cleaned_sql):
            return {
                "answer": "Security Alert: Query generation blocked. Only read-only SELECT operations are permitted.",
                "generated_sql": cleaned_sql,
                "columns": [],
                "data": []
            }

        try:
            columns, data_rows = SnowflakeManager.execute_query(account_config, cleaned_sql)
        except Exception as e:
            return {
                "answer": f"Database Execution Error: {str(e)}",
                "generated_sql": cleaned_sql,
                "columns": [],
                "data": []
            }

        preview_data = data_rows[:25]
        narrative_prompt = f"""
        You are a Pharmaceutical Insights Lead. Summarize the retrieved query results in concise executive prose directly answering the question. Include specific drug names, metrics, or revenue figures if present.

        QUESTION: {user_query}
        EXECUTED SQL: {cleaned_sql}
        RETRIEVED DATA: {preview_data}

        EXECUTIVE INSIGHT:
        """
        narrative = str(self.llm.invoke(narrative_prompt).content).strip()

        return {
            "answer": narrative,
            "generated_sql": cleaned_sql,
            "columns": columns,
            "data": data_rows
        }
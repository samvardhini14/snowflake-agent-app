import os
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import snowflake.connector

from snowflake_client import SnowflakeManager
from agent import DataAgent

app = FastAPI(title="Pharma AI Agent API", version="0.1.0")

# --- CORS Configuration ---
# allow_origins=["*"] enables Power Apps custom connectors, Azure App Service, and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ConnectionRequest(BaseModel):
    config: Dict[str, str]

class ChatRequest(BaseModel):
    config: Dict[str, str]
    openai_key: str
    user_prompt: str

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

# --- Environment-based Snowflake Helper ---
def connect_from_env():
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError("Missing Snowflake environment variables: " + ", ".join(missing))
    
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"].strip().strip("'").strip('"'),
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.getenv("SNOWFLAKE_ROLE") or None,
        login_timeout=20,
        network_timeout=30,
    )

# --- Governed Intent Templates ---
INTENTS = [
    {
        "id": "adverse_events",
        "keywords": ["adverse event", "side effect", "safety event", "safety signal"],
        "title": "Adverse event summary",
        "sql": """SELECT PRODUCT_NAME, EVENT_TYPE, SEVERITY, COUNT(*) AS EVENT_COUNT
                  FROM PHARMA_ADVERSE_EVENTS
                  WHERE EVENT_DATE >= DATEADD(day, -90, CURRENT_DATE())
                  GROUP BY PRODUCT_NAME, EVENT_TYPE, SEVERITY
                  ORDER BY EVENT_COUNT DESC
                  LIMIT 100""",
        "description": "Recent adverse-event records grouped by product, event type, and severity."
    },
    {
        "id": "clinical_trials",
        "keywords": ["clinical trial", "trial enrollment", "enrollment", "study site", "recruitment"],
        "title": "Clinical trial enrollment",
        "sql": """SELECT STUDY_ID, STUDY_NAME, SITE_NAME, COUNTRY, ENROLLED_COUNT, TARGET_COUNT,
                         ROUND(100.0 * ENROLLED_COUNT / NULLIF(TARGET_COUNT, 0), 1) AS ENROLLMENT_PCT
                  FROM PHARMA_TRIAL_ENROLLMENT
                  ORDER BY ENROLLMENT_PCT ASC
                  LIMIT 100""",
        "description": "Enrollment progress by clinical study and site."
    },
    {
        "id": "sales",
        "keywords": ["sales", "revenue", "prescription", "market performance", "product performance"],
        "title": "Product sales performance",
        "sql": """SELECT PRODUCT_NAME, REGION, SUM(NET_SALES) AS NET_SALES,
                         SUM(UNITS_SOLD) AS UNITS_SOLD
                  FROM PHARMA_PRODUCT_SALES
                  WHERE SALES_DATE >= DATEADD(month, -3, CURRENT_DATE())
                  GROUP BY PRODUCT_NAME, REGION
                  ORDER BY NET_SALES DESC
                  LIMIT 100""",
        "description": "Product sales and units by region for the last three months."
    },
    {
        "id": "inventory",
        "keywords": ["inventory", "stock", "shortage", "supply", "batch", "expiry"],
        "title": "Inventory and supply status",
        "sql": """SELECT PRODUCT_NAME, LOT_NUMBER, SITE_NAME, QUANTITY_ON_HAND, EXPIRY_DATE, STATUS
                  FROM PHARMA_INVENTORY
                  WHERE STATUS IN ('LOW_STOCK', 'AT_RISK', 'EXPIRED')
                     OR EXPIRY_DATE <= DATEADD(day, 90, CURRENT_DATE())
                  ORDER BY EXPIRY_DATE ASC
                  LIMIT 100""",
        "description": "Low-stock, at-risk, and near-expiry inventory."
    },
]

def choose_intent(question: str):
    q = question.lower()
    scored = [(sum(1 for kw in item["keywords"] if kw in q), item) for item in INTENTS]
    score, intent = max(scored, key=lambda x: x[0])
    return intent if score else None

# --- API Endpoints ---

# 1. Health Check (Used by Azure App Service ping)
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pharma-snowflake-agent"}

# 2. Suggested Examples Endpoint
@app.get("/api/examples")
def examples():
    return {"examples": [
        "Show adverse events by product and severity",
        "Which clinical trial sites are behind enrollment targets?",
        "Compare recent product sales by region",
        "Show inventory that is low stock or near expiry",
    ]}

# 3. Governed `/api/ask` Endpoint (Template-driven)
@app.post("/api/ask")
def ask(req: AskRequest):
    intent = choose_intent(req.question)
    if not intent:
        return {
            "answer": "I couldn't confidently map that question to a configured pharma data domain. Try asking about adverse events, clinical trial enrollment, product sales, or inventory.",
            "intent": None, "columns": [], "rows": [], "row_count": 0,
            "sql": None, "note": "Only approved query templates are enabled in this starter."
        }
    try:
        conn = connect_from_env()
        cur = conn.cursor()
        cur.execute(intent["sql"])
        columns = [d[0] for d in cur.description] if cur.description else []
        raw_rows = cur.fetchall()
        
        def safe(v: Any):
            if hasattr(v, "isoformat"):
                return v.isoformat()
            if hasattr(v, "as_tuple"):
                return float(v)
            return v
            
        rows = [{col: safe(val) for col, val in zip(columns, row)} for row in raw_rows]
        cur.close()
        conn.close()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Snowflake query failed. Validate table/column names in Snowflake.")

    return {
        "answer": f"Found {len(rows)} row(s) for **{intent['title']}**. {intent['description']}",
        "intent": intent["title"],
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "sql": intent["sql"],
        "note": "Results are retrieved live from Snowflake. Validate clinical/safety interpretation with domain owners."
    }

# 4. Connection Check Endpoint
@app.post("/api/connect")
async def connect_dynamic(req: ConnectionRequest):
    success, msg = SnowflakeManager.test_connection(req.config)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

# 5. Dynamic LLM Agent Endpoint
@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        agent = DataAgent(openai_api_key=req.openai_key)
        result = agent.process_query(req.user_prompt, req.config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
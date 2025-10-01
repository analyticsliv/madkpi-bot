import os
import re
import json
import logging
import difflib
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery
from google.oauth2 import service_account
import google.generativeai as genai

# Logging master
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration 
class Config:
    def __init__(self):
        self.bq_project = os.getenv("BQ_PROJECT", "credible-glow-442213-j8")
        self.bq_dataset = os.getenv("BQ_DATASET", "MADKPI")
        self.default_table = os.getenv("BQ_TABLE", "Madkpi_data_view")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.max_scan_bytes = int(os.getenv("MAX_SCAN_BYTES", 10 * 1024**3))
        self.service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def setup(self):
        try:
            if self.service_account_file:
                creds = service_account.Credentials.from_service_account_file(self.service_account_file)
                bq_client = bigquery.Client(credentials=creds, project=self.bq_project)
            else:
                bq_client = bigquery.Client(project=self.bq_project)
        except Exception as e:
            logger.error(f"Failed to create BigQuery client: {e}")
            raise

        if not self.google_api_key:
            logger.warning("GOOGLE_API_KEY not set. Gemini calls will fail if used.")
        else:
            genai.configure(api_key=self.google_api_key)

        return bq_client

config = Config()

# Domain / Report mapping
REPORT_CONFIG = {
    "Madkpi_data_view": {
        "table": os.getenv("PARTNER_TABLE", "Madkpi_data_view"),
        "row_limit": int(os.getenv("PARTNER_ROW_LIMIT", "1000")),
        "description": "Advertiser/Partner level performance table."
    },
    "LI_level_ads_monitoring_view": {
        "table": os.getenv("CREATIVE_TABLE", "LI_level_ads_monitoring_view"),
        "row_limit": int(os.getenv("CREATIVE_ROW_LIMIT", "1000")),
        "description": "Line Item level performance table."
    }
}

# Keywords for in-scope detection
SCHEMA_LI = [
    "date","partner_name","partner_id","advertiser_name","advertiser_id","campaign_name",
    "campaign_id","io_name","io_id","li_name","li_id","fl_activity_name","fl_activity_id",
    "advertiser_currency","advertiser_timezone","device_type","operating_system","country",
    "region","impressions","clicks","billable_impressions","active_view_viewable_impressions",
    "total_conversions","post_view_conversions","ctr","post_click_conversions","revenue",
    "complete_views_video","midpoint_views_video","media_cost","cron_ts"
]

SCHEMA_MADKPI = [
    "date","advertiser_id","advertiser_name","advertiser_currency",
    "partner_currency","impressions","clicks","revenue_partner_currency"
]

# Build DV360 keywords dynamically from schemas
DV360_KEYWORDS = sorted(set(SCHEMA_MADKPI + SCHEMA_LI))
GENERIC_IN_SCOPE = ["row", "rows", "data", "sample", "show table", "schema"]

# Helpers
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

def is_in_scope(question: str, threshold: float = 0.7) -> bool:
    q_lower = question.lower()
    tokens = re.findall(r"\w+", q_lower)
    token_set = set(tokens)

    # Exact token matches
    for kw in DV360_KEYWORDS:
        kw_tokens = re.findall(r"\w+", kw.lower())
        if all(t in token_set for t in kw_tokens):
            return True

    if any(g in q_lower for g in GENERIC_IN_SCOPE):
        return True

    # Fuzzy token match
    for word in tokens:
        best = difflib.get_close_matches(word, DV360_KEYWORDS + GENERIC_IN_SCOPE, n=1, cutoff=threshold)
        if best:
            logger.info(f"Fuzzy matched '{word}' -> '{best[0]}'")
            return True

    return False

def tokenize_keywords(keywords):
    """Expand schema keywords into tokens (split underscores)."""
    expanded = []
    for kw in keywords:
        parts = kw.lower().replace("_", " ").split()
        expanded.extend(parts)
    return set(expanded)

# Preprocess schema tokens
madkpi_tokens = tokenize_keywords(SCHEMA_MADKPI)
li_level_tokens = tokenize_keywords(SCHEMA_LI)

def fuzzy_score(tokens, keyword_tokens, cutoff=0.8):
    score = 0
    for word in tokens:
        if word in keyword_tokens:
            score += 1
        else:
            best = difflib.get_close_matches(word, keyword_tokens, n=1, cutoff=cutoff)
            if best:
                score += 1
    return score

def detect_report_level(question: str) -> Optional[str]:
    tokens = re.findall(r"\w+", question.lower())
    madkpi_score = fuzzy_score(tokens, madkpi_tokens)
    li_score = fuzzy_score(tokens, li_level_tokens)
    logger.info(f"Schema detection → Madkpi={madkpi_score}, LI={li_score}")

    if madkpi_score > li_score:
        return "Madkpi_data_view"
    elif li_score > madkpi_score:
        return "LI_level_ads_monitoring_view"
    else:
        return None

def find_best_table_match(table_candidates: List[str], preferred_names: List[str], threshold: float = 0.6) -> Optional[str]:
    lower_candidates = {t.lower(): t for t in table_candidates}
    for name in preferred_names:
        if not name:
            continue
        if name.lower() in lower_candidates:
            return lower_candidates[name.lower()]
    for pref in preferred_names:
        if not pref:
            continue
        for t in table_candidates:
            if pref.lower() in t.lower():
                return t
    for pref in preferred_names:
        if not pref:
            continue
        matches = difflib.get_close_matches(pref.lower(), [t.lower() for t in table_candidates], n=1, cutoff=threshold)
        if matches:
            picked_lower = matches[0]
            return lower_candidates.get(picked_lower, picked_lower)
    return None

# BigQuery Manager with CACHING and LAZY LOADING
class BigQueryManager:
    # Class-level cache for schema info
    _schema_cache = {}
    
    def __init__(self, client: bigquery.Client, dataset: str, table: str):
        self.client = client
        self.dataset = dataset
        self.table = table
        self.available_columns: List[str] = []
        # LAZY LOADING - only fetch columns when needed
        self._columns_loaded = False

    def _ensure_columns_loaded(self):
        """Lazy load columns only when needed"""
        if not self._columns_loaded:
            cache_key = f"{self.client.project}.{self.dataset}.{self.table}"
            if cache_key in BigQueryManager._schema_cache:
                self.available_columns = BigQueryManager._schema_cache[cache_key]
                logger.info(f"Loaded columns from cache for {cache_key}")
            else:
                self.available_columns = self._get_table_columns()
                BigQueryManager._schema_cache[cache_key] = self.available_columns
                logger.info(f"Fetched and cached columns for {cache_key}")
            self._columns_loaded = True

    def _get_table_columns(self) -> List[str]:
        """Optimized column fetching - only metadata, no data"""
        try:
            table_name_only = self.table.split('.')[-1]
            # Only fetch column metadata - very fast
            query = f"""
            SELECT column_name, data_type
            FROM `{self.client.project}.{self.dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE LOWER(table_name) = LOWER('{table_name_only}')
            ORDER BY ordinal_position
            """
            results = self.client.query(query).result()
            return [f"{row.column_name} ({row.data_type})" for row in results]
        except Exception as e:
            logger.error(f"Failed to get table columns: {e}")
            return []

    def execute_query(self, sql_query: str, dry_run: bool = False) -> pd.DataFrame:
        try:
            job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=True)  # Enable cache
            job = self.client.query(sql_query, job_config=job_config)
            if dry_run:
                return job.total_bytes_processed
            results = job.result()
            return results.to_dataframe(create_bqstorage_client=False)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def get_sample_data(self, limit: int = 3) -> pd.DataFrame:
        """Optimized sample data - with date filter and reduced rows"""
        # Add date filter to avoid full table scan
        query = f"""
        SELECT * 
        FROM `{self.client.project}.{self.dataset}.{self.table}` 
        WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        LIMIT {limit}
        """
        return self.execute_query(query)

    def get_table_schema(self) -> str:
        """Optimized schema info - uses cached columns, minimal sample"""
        try:
            self._ensure_columns_loaded()  # Load columns lazily
            
            # Return lightweight schema without sample data
            schema_info = f"Table: {self.client.project}.{self.dataset}.{self.table}\n"
            schema_info += f"Columns ({len(self.available_columns)}): {', '.join(self.available_columns[:20])}"
            if len(self.available_columns) > 20:
                schema_info += f"... and {len(self.available_columns) - 20} more"
            
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return f"Table: {self.client.project}.{self.dataset}.{self.table}"

    def get_detailed_schema_with_sample(self) -> Dict:
        """Separate method for when sample data is actually needed"""
        self._ensure_columns_loaded()
        try:
            sample_df = self.get_sample_data(3)
            return {
                "schema": f"Columns: {', '.join(self.available_columns)}",
                "sample": sample_df.to_string()
            }
        except Exception as e:
            logger.error(f"Failed to get detailed schema: {e}")
            return {"schema": f"Columns: {', '.join(self.available_columns)}", "sample": ""}

    def list_tables_in_dataset(self) -> List[str]:
        """Cached table listing"""
        cache_key = f"tables_{self.client.project}_{self.dataset}"
        if cache_key in BigQueryManager._schema_cache:
            return BigQueryManager._schema_cache[cache_key]
        
        try:
            tables = self.client.list_tables(f"{self.client.project}.{self.dataset}")
            table_list = [t.table_id for t in tables]
            BigQueryManager._schema_cache[cache_key] = table_list
            return table_list
        except Exception as e:
            logger.error(f"Failed to list tables in dataset {self.dataset}: {e}")
            return []

# Gemini / SQL generation
class GeminiManager:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.gemini_model
        if not config.google_api_key:
            logger.warning("Gemini API key not configured. Generation will fail if attempted.")

    def generate_sql_from_prompt(self, user_prompt: str, table_schema: str, full_table_name: str, row_limit: int) -> str:
        # Optimized prompt - more concise
        prompt = f"""
You are a BigQuery SQL expert for DV360. Generate ONLY the SQL query.

Table: {full_table_name}
{table_schema}

User: "{user_prompt}"

Rules:
- Use ONLY `{full_table_name}` in FROM
- ALWAYS add: WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
- Use SAFE_DIVIDE for divisions
- Cast NUMERIC to FLOAT64 when aggregating
- Case-insensitive LIKE for partial names
- Add LIMIT {row_limit} if no aggregation
- If out of scope: return "Out of scope"

Return ONLY SQL, no markdown.
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            sql_query = response.text.strip()
            return sql_query.replace("```sql", "").replace("```", "").strip()
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise

    def analyze_results(self, user_prompt: str, sql_query: str, results_df: pd.DataFrame) -> str:
        try:
            # Reduce sample size for analysis
            df_sample = results_df.head(5).copy()
            for col in df_sample.columns:
                df_sample[col] = df_sample[col].astype(str)
            
            prompt = f"""
User: {user_prompt}
SQL: {sql_query}
Sample (5 rows):
{df_sample.to_string()}

Provide brief:
1) Summary
2) Key insights
3) Recommendations
"""
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return f"Analysis generation failed: {e}"

# Analytics Service (Core)
class AnalyticsService:
    def __init__(self):
        self.bq_client = config.setup()
        self.gemini_manager = GeminiManager(config.gemini_model)

    def auto_select_table(self, question: str, report_level: Optional[str] = None, table_hint: Optional[str] = None) -> str:
        bq_manager = BigQueryManager(self.bq_client, config.bq_dataset, config.default_table)
        dataset_tables = bq_manager.list_tables_in_dataset()

        if table_hint:
            match = find_best_table_match(dataset_tables, [table_hint, table_hint.split('.')[-1]], threshold=0.5)
            if match:
                logger.info(f"Using table match from hint: {match}")
                return match

        candidates = []
        if report_level and report_level in REPORT_CONFIG:
            candidates.append(REPORT_CONFIG[report_level]["table"])
        else:
            detected = detect_report_level(question)
            if detected and detected in REPORT_CONFIG:
                candidates.append(REPORT_CONFIG[detected]["table"])

        match = find_best_table_match(dataset_tables, candidates, threshold=0.5)
        if match:
            logger.info(f"Auto-selected table: {match}")
            return match

        if config.default_table in dataset_tables:
            return config.default_table

        if dataset_tables:
            return dataset_tables[0]

        return config.default_table

    async def stream_analysis(self, user_prompt: str, report_level: Optional[str] = None, table_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        def block(t: str, data: Dict):
            return json.dumps({"type": t, "timestamp": datetime.now().isoformat(), "data": data}, cls=DecimalEncoder) + "\n"

        try:
            if not is_in_scope(user_prompt):
                yield block("error", {"message": "Out of scope: I can only answer DV360-related questions."})
                return

            yield block("progress", {"message": "Detecting best table...", "percentage": 5})
            selected_table = self.auto_select_table(user_prompt, report_level, table_override)
            full_table_name = f"{self.bq_client.project}.{config.bq_dataset}.{selected_table}"
            yield block("progress", {"message": f"Selected table: {selected_table}", "percentage": 10})

            bq_manager = BigQueryManager(self.bq_client, config.bq_dataset, selected_table)
            # Get lightweight schema (no sample data)
            table_schema = bq_manager.get_table_schema()

            row_limit = 1000
            if report_level and report_level in REPORT_CONFIG:
                row_limit = REPORT_CONFIG[report_level]["row_limit"]
            else:
                detected = detect_report_level(user_prompt)
                if detected and detected in REPORT_CONFIG:
                    row_limit = REPORT_CONFIG[detected]["row_limit"]

            yield block("progress", {"message": "Generating SQL query...", "percentage": 20})
            try:
                sql_query = self.gemini_manager.generate_sql_from_prompt(user_prompt, table_schema, full_table_name, row_limit)
            except Exception as e:
                yield block("error", {"message": "SQL generation failed", "details": str(e)})
                return

            if not sql_query or "out of scope" in sql_query.lower() or sql_query.strip().startswith("❌"):
                yield block("error", {"message": "Out of scope. I can only give DV360 related query"})
                return

            yield block("code", {"language": "sql", "content": sql_query, "title": "Generated SQL Query"})
            yield block("progress", {"message": "Validating query...", "percentage": 40})

            try:
                scanned_bytes = bq_manager.execute_query(sql_query, dry_run=True)
                scanned_mb = round(scanned_bytes / 1e6, 2)
            except Exception as e:
                yield block("error", {"message": "Query validation failed", "details": str(e)})
                return

            yield block("progress", {"message": f"Will scan ~{scanned_mb} MB", "percentage": 50})
            if scanned_bytes and scanned_bytes > config.max_scan_bytes:
                yield block("error", {"message": "⚠️ Query too expensive. Please refine your question."})
                return

            yield block("progress", {"message": "Executing query...", "percentage": 60})
            try:
                results_df = bq_manager.execute_query(sql_query, dry_run=False)
            except Exception as e:
                yield block("error", {"message": "Query execution failed", "details": str(e)})
                return

            if results_df.empty:
                yield block("error", {"message": "No results found", "details": "The query returned no rows."})
                return

            preview_rows = results_df.head(100)
            table_data = preview_rows.replace({np.nan: None}).to_dict(orient="records")
            columns_meta = [{"key": col, "label": col, "sortable": True} for col in results_df.columns]

            yield block("metrics", {
                "title": "Query Results Summary",
                "metrics": [
                    {"label": "Total Rows", "value": len(results_df), "format": "number"},
                    {"label": "Columns", "value": len(results_df.columns), "format": "number"},
                    {"label": "Table", "value": selected_table, "format": "string"}
                ]
            })

            yield block("table", {
                "title": f"Results (showing first {min(100, len(results_df))} rows)",
                "columns": columns_meta,
                "data": table_data,
                "pagination": True
            })

            yield block("progress", {"message": "Generating insights...", "percentage": 80})
            try:
                analysis = self.gemini_manager.analyze_results(user_prompt, sql_query, results_df)
            except Exception as e:
                analysis = f"Analysis generation failed: {e}"

            yield block("markdown", {"title": "Insights & Recommendations", "content": f"## AI Analysis\n\n{analysis}"})
            yield block("suggestions", {"title": "Follow-up questions", "suggestions": [
                "Show trend over time",
                "Break down by dimensions",
                "Top performing segments"
            ]})
            yield block("progress", {"message": "Complete!", "percentage": 100})

        except Exception as e:
            logger.exception("Unexpected error during analysis stream")
            yield block("error", {"message": "Analysis failed", "details": str(e)})

# FastAPI App
app = FastAPI(title="DV360 Analytics API", version="1.0.0", description="Natural language analytics for DV360 data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    question: str
    report_level: Optional[str] = None
    table: Optional[str] = None

analytics_service: Optional[AnalyticsService] = None

@app.on_event("startup")
async def startup_event():
    global analytics_service
    try:
        _ = config.setup()
        analytics_service = AnalyticsService()
        logger.info("Analytics service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analytics service: {e}")
        analytics_service = None

@app.get("/")
async def root():
    return {"status": "healthy", "service": "DV360 Analytics API"}

@app.get("/health")
async def health_check():
    if analytics_service is None:
        return {"status": "degraded", "bigquery": "not-initialized", "gemini": "not-initialized", "timestamp": datetime.now().isoformat()}
    return {"status": "healthy", "bigquery": "connected", "gemini": "configured" if config.google_api_key else "not-configured", "timestamp": datetime.now().isoformat()}

@app.get("/tables")
async def list_tables():
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        return {"dataset": config.bq_dataset, "tables": list(REPORT_CONFIG.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema(table: Optional[str] = Query(None, description="Optional table name override")):
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    table_name = table or config.default_table
    try:
        bq_mgr = BigQueryManager(analytics_service.bq_client, config.bq_dataset, table_name)
        detailed = bq_mgr.get_detailed_schema_with_sample()
        bq_mgr._ensure_columns_loaded()
        return {
            "schema": detailed["schema"],
            "sample_data": bq_mgr.get_sample_data(5).to_dict("records"),
            "columns": bq_mgr.available_columns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_class=StreamingResponse)
async def analyze(request: AnalysisRequest = Body(...)):
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    generator = analytics_service.stream_analysis(request.question, request.report_level, request.table)
    return StreamingResponse(generator, media_type="application/x-ndjson", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
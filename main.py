import os
import re
import json
import logging
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
        "description": "Advertiser/Partner level performance data with impressions, clicks, revenue",
        "schema_summary": "date, advertiser_id, advertiser_name, advertiser_currency, partner_currency, impressions, clicks, revenue_partner_currency",
        "key_columns": {
            "cost": "NOT AVAILABLE - use revenue_partner_currency instead",
            "advertiser": "advertiser_name, advertiser_id",
            "revenue": "revenue_partner_currency"
        }
    },
    "LI_level_ads_monitoring_view": {
        "table": os.getenv("CREATIVE_TABLE", "LI_level_ads_monitoring_view"),
        "row_limit": int(os.getenv("CREATIVE_ROW_LIMIT", "1000")),
        "description": "Line Item level granular performance including campaign, IO, creative details",
        "schema_summary": "date, partner_name, advertiser_name, campaign_name, io_name, li_name, device_type, country, impressions, clicks, total_conversions, ctr, revenue, media_cost",
        "key_columns": {
            "cost": "media_cost (NOT 'cost' or 'spend')",
            "advertiser": "advertiser_name, advertiser_id",
            "campaign": "campaign_name, campaign_id",
            "conversions": "total_conversions, post_click_conversions, post_view_conversions",
            "revenue": "revenue (in advertiser currency)"
        }
    }
}

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

# LLM-based scope and table detection
class IntelligentScopeDetector:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def is_in_scope(self, question: str) -> Dict[str, any]:
        """
        Use LLM to determine if question is related to DV360/advertising analytics
        Returns: {"in_scope": bool, "reason": str, "confidence": str}
        """
        prompt = f"""You are a classifier for a DV360 advertising analytics system.

Available data domains:
- Advertiser/Partner performance (impressions, clicks, revenue, dates)
- Campaign and Line Item performance (conversions, CTR, media cost, device types, countries)
- DV360 advertising metrics and KPIs

Question: "{question}"

Determine if this question can be answered using DV360 advertising data.

Respond ONLY with valid JSON:
{{
  "in_scope": true/false,
  "reason": "brief explanation",
  "confidence": "high/medium/low"
}}

Examples of IN SCOPE:
- "top advertisers by revenue"
- "show me campaign performance last week"
- "which countries have highest CTR"
- "compare impressions across partners"

Examples of OUT OF SCOPE:
- "what's the weather today"
- "tell me a joke"
- "how to cook pasta"
- "latest news headlines"
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            logger.info(f"Scope detection: {result}")
            return result
        except Exception as e:
            logger.error(f"Scope detection failed: {e}, defaulting to in-scope")
            # Default to in-scope to avoid blocking legitimate queries
            return {"in_scope": True, "reason": "Classification error, allowing query", "confidence": "low"}
    
    def detect_report_level(self, question: str) -> Dict[str, any]:
        """
        Use LLM to determine which table/report level is best for the question
        Returns: {"table": str, "reason": str, "confidence": str}
        """
        tables_info = "\n".join([
            f"- {name}: {config['description']}\n  Columns: {config['schema_summary']}\n  Key: {config.get('key_columns', {})}"
            for name, config in REPORT_CONFIG.items()
        ])
        
        prompt = f"""You are a data expert for DV360 advertising analytics.

Available tables:
{tables_info}

Question: "{question}"

Select the MOST APPROPRIATE table based on the columns needed.

CRITICAL RULES:
1. For CPC (cost per click), CPA (cost per action), ROAS, or any COST/SPEND analysis:
   → Use LI_level_ads_monitoring_view (has media_cost column)
   
2. For revenue-only analysis without cost:
   → Can use Madkpi_data_view (has revenue_partner_currency)
   
3. For campaign/line-item/creative level details:
   → Use LI_level_ads_monitoring_view (granular data)
   
4. For high-level advertiser/partner summaries:
   → Use Madkpi_data_view (aggregated data)

Respond ONLY with valid JSON:
{{
  "table": "exact_table_name",
  "reason": "why this table is best",
  "confidence": "high/medium/low"
}}
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # Validate table name
            if result.get("table") not in REPORT_CONFIG:
                logger.warning(f"Invalid table selected: {result.get('table')}, using default")
                result["table"] = config.default_table
                result["confidence"] = "low"
            
            logger.info(f"Table detection: {result}")
            return result
        except Exception as e:
            logger.error(f"Table detection failed: {e}, using default")
            return {
                "table": config.default_table,
                "reason": f"Classification error: {str(e)}",
                "confidence": "low"
            }

# BigQuery Manager with CACHING and LAZY LOADING
class BigQueryManager:
    _schema_cache = {}
    
    def __init__(self, client: bigquery.Client, dataset: str, table: str):
        self.client = client
        self.dataset = dataset
        self.table = table
        self.available_columns: List[str] = []
        self._columns_loaded = False

    def _ensure_columns_loaded(self):
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
        try:
            table_name_only = self.table.split('.')[-1]
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
            job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=True)
            job = self.client.query(sql_query, job_config=job_config)
            if dry_run:
                return job.total_bytes_processed
            results = job.result()
            return results.to_dataframe(create_bqstorage_client=False)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def get_sample_data(self, limit: int = 3) -> pd.DataFrame:
        query = f"""
        SELECT * 
        FROM `{self.client.project}.{self.dataset}.{self.table}` 
        WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        LIMIT {limit}
        """
        return self.execute_query(query)

    def get_table_schema(self) -> str:
        """Get comprehensive schema with ALL columns for accurate SQL generation"""
        try:
            self._ensure_columns_loaded()
            # Return ALL columns, not just first 20 - critical for accurate SQL generation
            schema_info = f"Table: {self.client.project}.{self.dataset}.{self.table}\n"
            schema_info += f"Available Columns:\n"
            for col in self.available_columns:
                schema_info += f"  - {col}\n"
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return f"Table: {self.client.project}.{self.dataset}.{self.table}"

    def validate_sql_columns(self, sql_query: str) -> tuple[bool, str]:
        """
        Lightweight validation - just check the actual column references
        Returns: (is_valid, error_message)
        """
        self._ensure_columns_loaded()
        
        # Extract clean column names from schema
        available_col_names = set([col.split(' (')[0].lower() for col in self.available_columns])
        
        import re
        
        # Find column references after table alias (e.g., t.column_name)
        # This is more reliable than trying to parse entire SQL
        alias_column_pattern = r'\w+\.([a-z_][a-z0-9_]*)'
        alias_columns = re.findall(alias_column_pattern, sql_query.lower())
        
        # Also find bare column names in common contexts (more conservative)
        # Only in SELECT and GROUP BY to avoid false positives
        select_match = re.search(r'select\s+(.+?)\s+from', sql_query.lower(), re.DOTALL)
        groupby_match = re.search(r'group\s+by\s+(.+?)(?:order|having|limit|$)', sql_query.lower(), re.DOTALL)
        
        bare_columns = []
        if select_match:
            select_part = select_match.group(1)
            # Remove function calls like SUM(...), CAST(...)
            select_part = re.sub(r'\b\w+\s*\([^)]*\)', '', select_part)
            # Extract remaining identifiers
            bare_columns.extend(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b', select_part))
        
        if groupby_match:
            groupby_part = groupby_match.group(1)
            bare_columns.extend(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b', groupby_part))
        
        # Combine all potential columns
        all_potential = set(alias_columns + bare_columns)
        
        # SQL keywords to filter out
        sql_keywords = {
            'select', 'from', 'where', 'group', 'order', 'having', 'limit',
            'desc', 'asc', 'distinct', 'date', 'float64', 'string', 'int64',
            'cast', 'interval', 'current_date', 'date_sub'
        }
        
        # Find invalid columns
        invalid = []
        for col in all_potential:
            if col in sql_keywords or len(col) < 3:
                continue
            if col not in available_col_names:
                invalid.append(col)
        
        if invalid:
            # Show helpful suggestions
            suggestions = []
            for inv_col in invalid[:3]:
                # Try to find similar column names
                similar = [ac for ac in available_col_names if inv_col in ac or ac in inv_col]
                if similar:
                    suggestions.append(f"'{inv_col}' (try: {', '.join(similar[:2])})")
                else:
                    suggestions.append(f"'{inv_col}'")
            
            return False, f"Columns not found: {', '.join(suggestions)}."
        
        return True, ""
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
        prompt = f"""You are a BigQuery SQL expert for DV360 advertising analytics. Generate ONLY the SQL query.

{table_schema}

User Question: "{user_prompt}"

CRITICAL RULES:
1. Use ONLY columns that exist in the schema above - check carefully!
2. Column names are CASE-SENSITIVE - use exact names from schema
3. Use ONLY `{full_table_name}` in FROM clause
4. ALWAYS add: WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
5. Use SAFE_DIVIDE for divisions to avoid divide-by-zero errors
6. Cast numeric columns to FLOAT64 when aggregating: CAST(column_name AS FLOAT64)
7. For case-insensitive text matching use: LOWER(column_name) LIKE LOWER('%search%')
8. Add LIMIT {row_limit} if no aggregation (no GROUP BY)
9. For "top" queries, use ORDER BY DESC with LIMIT
10. Use meaningful column aliases

Common column mappings for this domain:
- Cost/Spend → use "media_cost" (not "cost" or "spend")
- Advertiser → use "advertiser_name" and "advertiser_id"
- Campaign → use "campaign_name" and "campaign_id"
- Impressions → use "impressions"
- Clicks → use "clicks"
- CTR → use "ctr" or calculate as SAFE_DIVIDE(clicks, impressions)
- Revenue → use "revenue" (in advertiser currency)
- Conversions → use "total_conversions" or specific types

Return ONLY the SQL query with no markdown formatting, no explanations, no code blocks.
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            sql_query = response.text.strip()
            # Remove any markdown formatting
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            return sql_query
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise

    def analyze_results(self, user_prompt: str, sql_query: str, results_df: pd.DataFrame) -> str:
        try:
            df_sample = results_df.head(5).copy()
            for col in df_sample.columns:
                df_sample[col] = df_sample[col].astype(str)
            
            prompt = f"""User Question: {user_prompt}
SQL Query: {sql_query}
Sample Results (5 rows):
{df_sample.to_string()}

Provide brief analysis:
1) Summary of findings
2) Key insights (2-3 points)
3) Actionable recommendations
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
        self.scope_detector = IntelligentScopeDetector(config.gemini_model)

    async def stream_analysis(self, user_prompt: str, report_level: Optional[str] = None, table_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        def block(t: str, data: Dict):
            return json.dumps({"type": t, "timestamp": datetime.now().isoformat(), "data": data}, cls=DecimalEncoder) + "\n"

        try:
            # Step 1: LLM-based scope detection
            yield block("progress", {"message": "Understanding your question...", "percentage": 5})
            scope_result = self.scope_detector.is_in_scope(user_prompt)
            
            if not scope_result.get("in_scope", False):
                yield block("error", {
                    "message": f"Out of scope: {scope_result.get('reason', 'Question not related to DV360 analytics')}",
                    "suggestion": "Please ask questions about advertisers, campaigns, impressions, clicks, revenue, or other DV360 metrics."
                })
                return

            # Step 2: LLM-based table detection
            yield block("progress", {"message": "Selecting best data source...", "percentage": 10})
            
            if table_override:
                selected_table = table_override
                logger.info(f"Using table override: {selected_table}")
            elif report_level and report_level in REPORT_CONFIG:
                selected_table = REPORT_CONFIG[report_level]["table"]
                logger.info(f"Using report_level: {selected_table}")
            else:
                table_result = self.scope_detector.detect_report_level(user_prompt)
                selected_table = table_result.get("table", config.default_table)
                logger.info(f"LLM selected table: {selected_table} (reason: {table_result.get('reason')})")

            full_table_name = f"{self.bq_client.project}.{config.bq_dataset}.{selected_table}"
            yield block("progress", {"message": f"Using table: {selected_table}", "percentage": 15})

            bq_manager = BigQueryManager(self.bq_client, config.bq_dataset, selected_table)
            table_schema = bq_manager.get_table_schema()

            row_limit = REPORT_CONFIG.get(selected_table, {}).get("row_limit", 1000)

            yield block("progress", {"message": "Generating SQL query...", "percentage": 25})
            try:
                sql_query = self.gemini_manager.generate_sql_from_prompt(user_prompt, table_schema, full_table_name, row_limit)
            except Exception as e:
                yield block("error", {"message": "SQL generation failed", "details": str(e)})
                return

            if not sql_query or len(sql_query.strip()) < 10:
                yield block("error", {"message": "Failed to generate valid SQL query"})
                return

            yield block("code", {"language": "sql", "content": sql_query, "title": "Generated SQL Query"})
            yield block("progress", {"message": "Validating query with BigQuery...", "percentage": 40})

            # Use BigQuery's dry run for validation - it's more accurate than our regex parsing
            try:
                scanned_bytes = bq_manager.execute_query(sql_query, dry_run=True)
                scanned_mb = round(scanned_bytes / 1e6, 2)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Query validation failed: {error_msg}")
                
                # Check if it's a column/syntax error that we can retry
                if "Unrecognized name" in error_msg or "Column" in error_msg or "not found" in error_msg:
                    yield block("progress", {"message": "Fixing query based on BigQuery feedback...", "percentage": 35})
                    try:
                        # Extract the specific error and retry
                        enhanced_prompt = f"""{user_prompt}

IMPORTANT: Previous SQL query failed with error:
{error_msg}

Please generate a corrected query using ONLY the columns available in the schema."""
                        
                        sql_query = self.gemini_manager.generate_sql_from_prompt(enhanced_prompt, table_schema, full_table_name, row_limit)
                        yield block("code", {"language": "sql", "content": sql_query, "title": "Corrected SQL Query"})
                        
                        # Try validation again
                        try:
                            scanned_bytes = bq_manager.execute_query(sql_query, dry_run=True)
                            scanned_mb = round(scanned_bytes / 1e6, 2)
                        except Exception as retry_error:
                            yield block("error", {"message": "Query validation failed after retry", "details": str(retry_error)})
                            return
                    except Exception as regen_error:
                        yield block("error", {"message": "Failed to regenerate query", "details": str(regen_error)})
                        return
                else:
                    yield block("error", {"message": "Query validation failed", "details": error_msg})
                    return

            yield block("progress", {"message": f"Will scan ~{scanned_mb} MB", "percentage": 50})
            if scanned_bytes and scanned_bytes > config.max_scan_bytes:
                yield block("error", {"message": "⚠️ Query too expensive. Please refine your question to scan less data."})
                return

            yield block("progress", {"message": "Executing query...", "percentage": 60})
            try:
                results_df = bq_manager.execute_query(sql_query, dry_run=False)
            except Exception as e:
                yield block("error", {"message": "Query execution failed", "details": str(e)})
                return

            if results_df.empty:
                yield block("error", {"message": "No results found", "details": "The query returned no rows. Try adjusting your date range or criteria."})
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
                "Show trend over time for this data",
                "Break down by additional dimensions",
                "Compare with previous period"
            ]})
            yield block("progress", {"message": "Complete!", "percentage": 100})

        except Exception as e:
            logger.exception("Unexpected error during analysis stream")
            yield block("error", {"message": "Analysis failed", "details": str(e)})

# FastAPI App
app = FastAPI(title="DV360 Analytics API", version="2.0.0", description="Natural language analytics for DV360 data with intelligent scope detection")

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
        logger.info("Analytics service with intelligent scope detection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analytics service: {e}")
        analytics_service = None

@app.get("/")
async def root():
    return {"status": "healthy", "service": "DV360 Analytics API v2.0 - LLM-powered scope detection"}

@app.get("/health")
async def health_check():
    if analytics_service is None:
        return {"status": "degraded", "bigquery": "not-initialized", "gemini": "not-initialized", "timestamp": datetime.now().isoformat()}
    return {"status": "healthy", "bigquery": "connected", "gemini": "configured" if config.google_api_key else "not-configured", "scope_detection": "llm-powered", "timestamp": datetime.now().isoformat()}

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
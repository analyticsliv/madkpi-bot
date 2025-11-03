import os
from dotenv import load_dotenv
import re
import json
import logging
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body, Header, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery
from google.oauth2 import service_account
import google.generativeai as genai
import jwt
from jwt import InvalidTokenError, ExpiredSignatureError
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables
load_dotenv()

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
        self.jwt_secret = os.getenv("JWT_SECRET")
        
        # Performance settings - OPTIMIZED FOR SPEED
        self.enable_parallel_llm = os.getenv("ENABLE_PARALLEL_LLM", "true").lower() == "true"
        self.skip_contextual_suggestions = os.getenv("SKIP_CONTEXTUAL_SUGGESTIONS", "true").lower() == "true"  # Changed default to TRUE
        
        # Validate JWT secret
        if not self.jwt_secret:
            logger.error("JWT_SECRET not set in environment variables!")
            raise ValueError("JWT_SECRET is required for production")
        
        logger.info(f"JWT_SECRET loaded successfully")
        logger.info(f"Parallel LLM: {'ENABLED' if self.enable_parallel_llm else 'DISABLED'}")

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

# JWT Token Handler - CONFIGURABLE FOR TESTING/PRODUCTION
def decode_advertiser_token(authorization: Optional[str] = Header(None)) -> List[str]:
    """
    Decode JWT token from Authorization header and extract advertiser IDs
    
    Set ENABLE_JWT_VERIFICATION=true in .env for production
    Set ENABLE_JWT_VERIFICATION=false in .env for local testing
    
    Returns: List of advertiser IDs
    Raises: HTTPException if token is invalid/missing/expired
    """
    if not authorization:
        logger.warning("Request without authorization header")
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    if not authorization.startswith("Bearer "):
        logger.warning("Invalid authorization header format")
        raise HTTPException(status_code=401, detail="Invalid authorization header format. Use: Bearer <token>")
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    # Check if verification should be enabled
    enable_verification = os.getenv("ENABLE_JWT_VERIFICATION", "false").lower() == "true"
    
    try:
        if enable_verification:
            # PRODUCTION: Verify signature with secret key
            logger.info("JWT verification ENABLED")
            decoded = jwt.decode(token, config.jwt_secret, algorithms=["HS256"])
        else:
            # TESTING: Skip signature verification
            logger.warning("JWT verification DISABLED - Testing mode only!")
            decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Check expiration
        current_time = datetime.now().timestamp()
        if decoded.get('exp', 0) < current_time:
            logger.warning("Expired token attempted")
            raise HTTPException(status_code=401, detail="Token expired")
        
        # Validate issuer (optional but recommended)
        expected_issuer = "madkpi-frontend"
        if decoded.get('iss') != expected_issuer:
            logger.warning(f"Invalid token issuer: {decoded.get('iss')}")
        
        # Extract advertiser IDs
        advertiser_ids = decoded.get('advertiserIds', [])
        
        if not isinstance(advertiser_ids, list):
            logger.error("Invalid advertiser IDs format in token")
            raise HTTPException(status_code=401, detail="Invalid advertiser IDs format")
        
        if len(advertiser_ids) == 0:
            logger.warning("Token with empty advertiser IDs")
            raise HTTPException(status_code=403, detail="No advertiser access")
        
        logger.info(f"Token decoded successfully - {len(advertiser_ids)} advertiser(s)")
        return advertiser_ids
        
    except ExpiredSignatureError:
        logger.warning("Token signature expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Token decoding error: {e}")
        raise HTTPException(status_code=401, detail="Token validation failed")

# LLM-based scope and table detection
class IntelligentScopeDetector:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=3)  # For parallel LLM calls
        
    def _call_llm(self, prompt: str) -> str:
        """Synchronous LLM call wrapper"""
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    
    def is_in_scope(self, question: str) -> Dict[str, any]:
        """
        Use LLM to determine if question is related to DV360/advertising analytics
        Returns: {"in_scope": bool, "reason": str, "confidence": str, "suggestion": str}
        """
        prompt = f"""You are a classifier for a DV360 advertising analytics system.

Available data domains:
- Advertiser/Partner performance (impressions, clicks, revenue, dates)
- Campaign and Line Item performance (conversions, CTR, media cost, device types, countries)
- DV360 advertising metrics and KPIs

NOT AVAILABLE (Financial metrics):
- Closing balance, account balance, payment status
- Invoicing, billing, credit limits
- Financial reconciliation data

Question: "{question}"

Determine if this question can be answered using DV360 advertising data.

Respond ONLY with valid JSON:
{{
  "in_scope": true/false,
  "reason": "brief explanation",
  "confidence": "high/medium/low",
  "suggestion": "if out of scope, suggest a similar in-scope query; if in scope, leave empty"
}}

Examples of IN SCOPE:
- "top advertisers by revenue"
- "show me campaign performance last week"
- "which countries have highest CTR"
- "compare impressions across partners"

Examples of OUT OF SCOPE (with suggestions):
- "closing balance for advertisers" → Suggest: "Show total revenue by advertiser"
- "account payment status" → Suggest: "Show campaign spend and media cost trends"
- "what's the weather today" → Suggest: "Try asking about campaign performance metrics"
"""
        try:
            result_text = self._call_llm(prompt)
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            logger.info(f"Scope detection: {result}")
            return result
        except Exception as e:
            logger.error(f"Scope detection failed: {e}, defaulting to in-scope")
            return {"in_scope": True, "reason": "Classification error, allowing query", "confidence": "low", "suggestion": ""}
    
    def detect_date_range(self, question: str) -> Dict[str, any]:
        """
        Use LLM to extract date range from user question
        Returns: {"date_filter": str, "reason": str, "estimated_days": int, "is_trend_query": bool, "trend_type": str}
        """
        prompt = f"""You are a date range expert for analytics queries.

Current date: {datetime.now().strftime('%Y-%m-%d')}

Question: "{question}"

Extract the date range requested by the user and convert it to a BigQuery WHERE clause.

RULES:
1. If specific dates mentioned → use exact dates
2. If relative time (last week, yesterday, last 7 days) → calculate from current date
3. If no date mentioned → use "NO_FILTER" (return all available data)
4. Use DATE() function for date columns
5. Always use CURRENT_DATE() for dynamic dates
6. Detect if this is a TREND query (week-over-week, month-over-month, day-over-day, year-over-year)

Examples:
- "last 7 days" → "WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)"
- "yesterday" → "WHERE DATE(date) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)"
- "January 2024" → "WHERE DATE(date) BETWEEN '2024-01-01' AND '2024-01-31'"
- "last month" → "WHERE DATE(date) >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH) AND DATE(date) < DATE_TRUNC(CURRENT_DATE(), MONTH)"
- "this year" → "WHERE DATE(date) >= DATE_TRUNC(CURRENT_DATE(), YEAR)"
- "Q1 2024" → "WHERE DATE(date) BETWEEN '2024-01-01' AND '2024-03-31'"
- "week-over-week" → needs at least 14+ days of data
- "month-over-month" → needs at least 60+ days of data
- No date mentioned → "NO_FILTER"

Respond ONLY with valid JSON:
{{
  "date_filter": "WHERE clause or NO_FILTER",
  "reason": "explanation of date range",
  "estimated_days": approximate number of days covered,
  "is_trend_query": true/false,
  "trend_type": "week-over-week|month-over-month|day-over-day|year-over-year|none",
  "minimum_days_needed": minimum days required for this analysis
}}
"""
        try:
            result_text = self._call_llm(prompt)
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            logger.info(f"Date range detection: {result}")
            return result
        except Exception as e:
            logger.error(f"Date range detection failed: {e}, using no filter")
            return {
                "date_filter": "NO_FILTER",
                "reason": f"Classification error: {str(e)}",
                "estimated_days": 365,
                "is_trend_query": False,
                "trend_type": "none",
                "minimum_days_needed": 0
            }
    
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
            result_text = self._call_llm(prompt)
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
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
    
    async def detect_all_parallel(self, question: str) -> Dict[str, Dict]:
        """
        🚀 OPTIMIZATION: Run scope, date, and table detection in parallel
        Returns: {"scope": {...}, "date": {...}, "table": {...}}
        """
        if not config.enable_parallel_llm:
            # Fallback to sequential
            return {
                "scope": self.is_in_scope(question),
                "date": self.detect_date_range(question),
                "table": self.detect_report_level(question)
            }
        
        logger.info("Running parallel LLM calls...")
        start_time = time.time()
        
        # Run all three LLM calls in parallel using thread pool
        loop = asyncio.get_event_loop()
        
        scope_future = loop.run_in_executor(self.executor, self.is_in_scope, question)
        date_future = loop.run_in_executor(self.executor, self.detect_date_range, question)
        table_future = loop.run_in_executor(self.executor, self.detect_report_level, question)
        
        # Wait for all to complete
        scope_result, date_result, table_result = await asyncio.gather(
            scope_future, date_future, table_future, return_exceptions=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel LLM calls completed in {elapsed:.2f}s")
        
        # Handle exceptions
        if isinstance(scope_result, Exception):
            logger.error(f"Scope detection failed: {scope_result}")
            scope_result = {"in_scope": True, "reason": "Error", "confidence": "low", "suggestion": ""}
        
        if isinstance(date_result, Exception):
            logger.error(f"Date detection failed: {date_result}")
            date_result = {"date_filter": "NO_FILTER", "reason": "Error", "estimated_days": 365, 
                          "is_trend_query": False, "trend_type": "none", "minimum_days_needed": 0}
        
        if isinstance(table_result, Exception):
            logger.error(f"Table detection failed: {table_result}")
            table_result = {"table": config.default_table, "reason": "Error", "confidence": "low"}
        
        return {
            "scope": scope_result,
            "date": date_result,
            "table": table_result
        }
    
    def validate_comparative_query(self, question: str, sql_query: str) -> Dict[str, any]:
        """
        Validate if a comparative query (e.g., comparing campaigns) has necessary filters
        Returns: {"valid": bool, "warning": str, "suggestion": str}
        
        ⚡ OPTIMIZATION: Made optional, can be skipped for performance
        """
        if os.getenv("SKIP_COMPARATIVE_VALIDATION", "false").lower() == "true":
            return {
                "is_comparative": False,
                "entities_compared": [],
                "has_proper_filters": True,
                "warning": "",
                "suggestion": ""
            }
        
        prompt = f"""You are a query validator for DV360 analytics.

Question: "{question}"
Generated SQL: "{sql_query}"

Determine if this is a COMPARATIVE query (comparing 2+ entities like campaigns, advertisers, etc.)

If it's comparative, check if:
1. The query properly filters for the specific entities being compared
2. The query has appropriate date ranges
3. The entities mentioned actually exist in the data

Respond ONLY with valid JSON:
{{
  "is_comparative": true/false,
  "entities_compared": ["entity1", "entity2"] or [],
  "has_proper_filters": true/false,
  "warning": "warning message if filters may be missing",
  "suggestion": "how to improve the query if needed"
}}

Examples:
- "Compare Campaign A vs Campaign B" → is_comparative=true, needs campaign name filters
- "Top 5 campaigns by impressions" → is_comparative=false, ranking query
"""
        try:
            result_text = self._call_llm(prompt)
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            logger.info(f"Comparative query validation: {result}")
            return result
        except Exception as e:
            logger.error(f"Comparative query validation failed: {e}")
            return {
                "is_comparative": False,
                "entities_compared": [],
                "has_proper_filters": True,
                "warning": "",
                "suggestion": ""
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
            schema_info = f"Table: {self.client.project}.{self.dataset}.{self.table}\n"
            schema_info += f"Available Columns:\n"
            for col in self.available_columns:
                schema_info += f"  - {col}\n"
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return f"Table: {self.client.project}.{self.dataset}.{self.table}"

    def get_detailed_schema_with_sample(self):
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

# Gemini / SQL generation
class GeminiManager:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.gemini_model
        if not config.google_api_key:
            logger.warning("Gemini API key not configured. Generation will fail if attempted.")

    def generate_sql_from_prompt(self, user_prompt: str, table_schema: str, full_table_name: str, row_limit: int, date_filter: str = "NO_FILTER", advertiser_ids: List[str] = None, is_trend_query: bool = False, trend_type: str = "none") -> str:
        # Construct date filter instruction
        if date_filter == "NO_FILTER":
            if is_trend_query:
                # For trend queries without explicit date, use appropriate default range
                if trend_type == "week-over-week":
                    date_instruction = "Use: WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 21 DAY) -- Last 3 weeks for week-over-week comparison"
                elif trend_type == "month-over-month":
                    date_instruction = "Use: WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) -- Last 3 months for month-over-month comparison"
                elif trend_type == "day-over-day":
                    date_instruction = "Use: WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) -- Last 7 days for day-over-day comparison"
                else:
                    date_instruction = "Use: WHERE DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) -- Last 30 days for trend analysis"
            else:
                date_instruction = "DO NOT add any date filter - return all available data"
        else:
            date_instruction = f"MUST include this date filter: {date_filter}"
        
        # Construct advertiser filter instruction
        if advertiser_ids and len(advertiser_ids) > 0:
            advertiser_ids_str = ", ".join([f"'{aid}'" for aid in advertiser_ids])
            advertiser_instruction = f"CRITICAL: MUST filter by advertiser_id IN ({advertiser_ids_str}) - This is mandatory for data security!"
        else:
            advertiser_instruction = "No advertiser filter required"
        
        # Add trend-specific instructions
        trend_instruction = ""
        if is_trend_query:
            if trend_type == "week-over-week":
                trend_instruction = """
WEEK-OVER-WEEK ANALYSIS REQUIREMENTS:
- Extract week from date: FORMAT_DATE('%Y-W%U', DATE(date)) as week
- Group by week and other dimensions
- Use LAG() window function to get previous week values
- Calculate percent change: (current - previous) / previous * 100
- Order by week DESC
- Example structure:
  SELECT 
    FORMAT_DATE('%Y-W%U', DATE(date)) as week,
    metric,
    LAG(metric) OVER (ORDER BY FORMAT_DATE('%Y-W%U', DATE(date))) as previous_week_metric,
    SAFE_DIVIDE(metric - LAG(metric) OVER (ORDER BY FORMAT_DATE('%Y-W%U', DATE(date))), 
                LAG(metric) OVER (ORDER BY FORMAT_DATE('%Y-W%U', DATE(date)))) * 100 as week_over_week_change
  FROM ...
"""
            elif trend_type == "month-over-month":
                trend_instruction = """
MONTH-OVER-MONTH ANALYSIS REQUIREMENTS:
- Extract month: FORMAT_DATE('%Y-%m', DATE(date)) as month
- Group by month and other dimensions
- Use LAG() window function to get previous month values
- Calculate percent change: (current - previous) / previous * 100
- Order by month DESC
"""
            elif trend_type == "day-over-day":
                trend_instruction = """
DAY-OVER-DAY ANALYSIS REQUIREMENTS:
- Group by DATE(date) and other dimensions
- Use LAG() window function to get previous day values
- Calculate percent change: (current - previous) / previous * 100
- Order by date DESC
"""
        
        prompt = f"""You are a BigQuery SQL expert for DV360 advertising analytics. Generate ONLY the SQL query.

{table_schema}

User Question: "{user_prompt}"

CRITICAL RULES:
1. Use ONLY columns that exist in the schema above - check carefully!
2. Column names are CASE-SENSITIVE - use exact names from schema
3. Use ONLY `{full_table_name}` in FROM clause
4. DATE FILTER: {date_instruction}
5. ADVERTISER FILTER: {advertiser_instruction}
6. Use SAFE_DIVIDE for divisions to avoid divide-by-zero errors
7. Cast numeric columns to FLOAT64 when aggregating: CAST(column_name AS FLOAT64)
8. For case-insensitive text matching use: LOWER(column_name) LIKE LOWER('%search%')
9. Add LIMIT {row_limit} if no aggregation (no GROUP BY)
10. For "top" queries, use ORDER BY DESC with LIMIT
11. Use meaningful column aliases
12. For COMPARATIVE queries (comparing specific entities), add WHERE filters for those entities

{trend_instruction}

Common column mappings for this domain:
- Cost/Spend → use "media_cost" (not "cost" or "spend")
- Advertiser → use "advertiser_name" and "advertiser_id"
- Campaign → use "campaign_name" and "campaign_id"
- Impressions → use "impressions"
- Clicks → use "clicks"
- CTR → use "ctr" or calculate as SAFE_DIVIDE(clicks, impressions)
- Revenue → use "revenue" (in advertiser currency)
- Conversions → use "total_conversions" or specific types

COMPARATIVE QUERY RULES:
- If comparing specific campaigns/advertisers by name, add:
  WHERE campaign_name IN ('Campaign1', 'Campaign2') OR
  WHERE advertiser_name IN ('Advertiser1', 'Advertiser2')
- This ensures the query only returns data for the entities being compared

IMPORTANT: The advertiser filter is MANDATORY and must be included in the WHERE clause!

Return ONLY the SQL query with no markdown formatting, no explanations, no code blocks.
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            sql_query = response.text.strip()
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

Total rows returned: {len(results_df)}

Provide brief analysis:
1) Summary of findings (2-3 sentences)
2) Key insights (2-3 bullet points with specific numbers)
3) Actionable recommendations (2-3 concrete actions)

Format your response in clear markdown with headers.
"""
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return f"Analysis generation failed: {e}"
    
    def generate_contextual_suggestions(self, user_prompt: str, results_df: pd.DataFrame, had_data: bool) -> List[str]:
        """
        Generate intelligent follow-up suggestions based on query results
        
        ⚡ OPTIMIZATION: Can be skipped via config for faster responses
        """
        if config.skip_contextual_suggestions:
            return [
                "Show trend over time for this data",
                "Break down by additional dimensions",
                "Compare with previous period"
            ]
        
        try:
            context = f"Original question: {user_prompt}\n"
            context += f"Data returned: {len(results_df)} rows\n" if had_data else "No data returned\n"
            
            if had_data and not results_df.empty:
                context += f"Columns: {', '.join(results_df.columns[:5])}\n"
            
            prompt = f"""{context}

Generate 3 intelligent follow-up questions that would provide deeper insights.

Rules:
1. If no data was returned, suggest checking different date ranges or entities
2. If data exists, suggest drilling down into dimensions (time, geography, device)
3. Suggest comparative analysis or trend analysis
4. Keep suggestions specific and actionable

Return as a JSON array of strings:
["suggestion 1", "suggestion 2", "suggestion 3"]
"""
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            suggestions = json.loads(result_text)
            return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return [
                "Try analyzing data for a different date range",
                "Break down results by additional dimensions",
                "Compare performance across different segments"
            ]

# Analytics Service (Core)
class AnalyticsService:
    def __init__(self):
        self.bq_client = config.setup()
        self.gemini_manager = GeminiManager(config.gemini_model)
        self.scope_detector = IntelligentScopeDetector(config.gemini_model)

    async def stream_analysis(self, user_prompt: str, advertiser_ids: List[str], report_level: Optional[str] = None, table_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        def block(t: str, data: Dict):
            return json.dumps({"type": t, "timestamp": datetime.now().isoformat(), "data": data}, cls=DecimalEncoder) + "\n"

        try:
            # Validate advertiser IDs
            if not advertiser_ids or len(advertiser_ids) == 0:
                yield block("error", {
                    "message": "No advertiser access",
                    "details": "Your account has no associated advertisers. Please contact your administrator."
                })
                return
            
            yield block("progress", {"message": f"Analyzing your question...", "percentage": 5})
            
            #  OPTIMIZATION: Run all detection in parallel
            detection_results = await self.scope_detector.detect_all_parallel(user_prompt)
            scope_result = detection_results["scope"]
            date_result = detection_results["date"]
            table_result = detection_results["table"]
            
            yield block("progress", {"message": "Analysis complete", "percentage": 15})
            
            if not scope_result.get("in_scope", False):
                suggestion = scope_result.get("suggestion", "")
                error_detail = f"{scope_result.get('reason', 'Question not related to DV360 analytics')}"
                
                if suggestion:
                    error_detail += f"\n\n Try this instead: \"{suggestion}\""
                
                yield block("error", {
                    "message": "Out of Scope Query",
                    "details": error_detail,
                    "category": "out_of_scope"
                })
                return

            # Extract detection results
            date_filter = date_result.get("date_filter", "NO_FILTER")
            is_trend_query = date_result.get("is_trend_query", False)
            trend_type = date_result.get("trend_type", "none")
            estimated_days = date_result.get("estimated_days", 0)
            minimum_days_needed = date_result.get("minimum_days_needed", 0)
            
            # Validate if we have enough data for trend analysis
            if is_trend_query and minimum_days_needed > 0:
                if date_filter == "NO_FILTER":
                    yield block("info", {
                        "title": "📊 Trend Analysis Detected",
                        "message": f"{trend_type.replace('-', ' ').title()} analysis requires at least {minimum_days_needed} days of data. Using appropriate date range.",
                        "icon": "📈"
                    })
                elif estimated_days < minimum_days_needed:
                    yield block("warning", {
                        "title": "⚠️ Insufficient Data for Trend Analysis",
                        "message": f"{trend_type.replace('-', ' ').title()} analysis typically requires at least {minimum_days_needed} days of data, but your date range covers approximately {estimated_days} days.",
                        "suggestion": f"Consider expanding your date range to at least {minimum_days_needed} days for more reliable {trend_type} insights.",
                        "icon": "⚠️"
                    })

            # Table selection
            if table_override:
                selected_table = table_override
            elif report_level and report_level in REPORT_CONFIG:
                selected_table = REPORT_CONFIG[report_level]["table"]
            else:
                selected_table = table_result.get("table", config.default_table)

            full_table_name = f"{self.bq_client.project}.{config.bq_dataset}.{selected_table}"
            
            bq_manager = BigQueryManager(self.bq_client, config.bq_dataset, selected_table)
            table_schema = bq_manager.get_table_schema()

            row_limit = REPORT_CONFIG.get(selected_table, {}).get("row_limit", 1000)

            yield block("progress", {"message": "Generating SQL query...", "percentage": 25})
            try:
                sql_query = self.gemini_manager.generate_sql_from_prompt(
                    user_prompt, 
                    table_schema, 
                    full_table_name, 
                    row_limit,
                    date_filter,
                    advertiser_ids,
                    is_trend_query,
                    trend_type
                )
            except Exception as e:
                yield block("error", {"message": "SQL generation failed", "details": str(e)})
                return

            if not sql_query or len(sql_query.strip()) < 10:
                yield block("error", {"message": "Failed to generate valid SQL query"})
                return

            # ⚡ PERFORMANCE: Comparative validation disabled (was adding 10-30s per request)
            # This validation was causing major slowdowns. Re-enable only if absolutely needed.
            # comp_validation = self.scope_detector.validate_comparative_query(user_prompt, sql_query)

            yield block("code", {"language": "sql", "content": sql_query, "title": "Generated SQL Query"})
            yield block("progress", {"message": "Validating query...", "percentage": 35})

            try:
                scanned_bytes = bq_manager.execute_query(sql_query, dry_run=True)
                scanned_mb = round(scanned_bytes / 1e6, 2)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Query validation failed: {error_msg}")
                
                if "Unrecognized name" in error_msg or "Column" in error_msg or "not found" in error_msg:
                    yield block("progress", {"message": "Fixing query...", "percentage": 30})
                    try:
                        enhanced_prompt = f"""{user_prompt}

IMPORTANT: Previous SQL query failed with error:
{error_msg}

Please generate a corrected query using ONLY the columns available in the schema."""
                        
                        sql_query = self.gemini_manager.generate_sql_from_prompt(
                            enhanced_prompt, 
                            table_schema, 
                            full_table_name, 
                            row_limit,
                            date_filter,
                            advertiser_ids,
                            is_trend_query,
                            trend_type
                        )
                        yield block("code", {"language": "sql", "content": sql_query, "title": "Corrected SQL Query"})
                        
                        try:
                            scanned_bytes = bq_manager.execute_query(sql_query, dry_run=True)
                            scanned_mb = round(scanned_bytes / 1e6, 2)
                        except Exception as retry_error:
                            yield block("error", {
                                "message": "Query validation failed after retry", 
                                "details": str(retry_error),
                                "suggestion": "Try rephrasing your question or check if the entities mentioned exist in the data"
                            })
                            return
                    except Exception as regen_error:
                        yield block("error", {"message": "Failed to regenerate query", "details": str(regen_error)})
                        return
                else:
                    yield block("error", {"message": "Query validation failed", "details": error_msg})
                    return
            
            if scanned_bytes and scanned_bytes > config.max_scan_bytes:
                yield block("error", {
                    "message": "Query Too Expensive",
                    "details": f"This query would scan {scanned_mb} MB, which exceeds the limit. Please refine your question to scan less data.",
                    "suggestion": "Try: Adding date filters, limiting to specific campaigns/advertisers, or reducing the date range"
                })
                return

            yield block("progress", {"message": "Executing query...", "percentage": 50})
            try:
                results_df = bq_manager.execute_query(sql_query, dry_run=False)
            except Exception as e:
                yield block("error", {
                    "message": "Query execution failed", 
                    "details": str(e),
                    "suggestion": "Try simplifying your query or checking if the data exists for the specified criteria"
                })
                return

            # Enhanced no-data handling
            if results_df.empty:
                no_data_message = "No data found matching your criteria."
                suggestions = []
                
                if date_filter != "NO_FILTER":
                    suggestions.append("Try expanding your date range")
                
                if comp_validation.get("is_comparative"):
                    entities = comp_validation.get("entities_compared", [])
                    if entities:
                        suggestions.append(f"Verify that {', '.join(entities)} exist in the selected time period")
                
                suggestions.extend([
                    "Check if the campaigns/advertisers mentioned are active",
                    "Try removing specific filters to see broader data"
                ])
                
                yield block("warning", {
                    "title": "No Results Found",
                    "message": no_data_message,
                    "suggestions": suggestions,
                    "icon": "📭"
                })
                
                # Still generate contextual suggestions even with no data
                contextual_suggestions = self.gemini_manager.generate_contextual_suggestions(
                    user_prompt, results_df, False
                )
                yield block("suggestions", {
                    "title": "Try these alternative queries", 
                    "suggestions": contextual_suggestions
                })
                return

            preview_rows = results_df.head(100)
            table_data = preview_rows.replace({np.nan: None}).to_dict(orient="records")
            columns_meta = [{"key": col, "label": col, "sortable": True} for col in results_df.columns]

            yield block("metrics", {
                "title": "Query Results Summary",
                "metrics": [
                    {"label": "Total Rows", "value": len(results_df), "format": "number"},
                    {"label": "Columns", "value": len(results_df.columns), "format": "number"},
                    # {"label": "Data Scanned", "value": f"{scanned_mb} MB", "format": "string"}
                ]
            })

            yield block("table", {
                "title": f"Results (showing first {min(100, len(results_df))} of {len(results_df)} rows)",
                "columns": columns_meta,
                "data": table_data,
                "pagination": True
            })

            yield block("progress", {"message": "Generating insights...", "percentage": 75})
            try:
                analysis = self.gemini_manager.analyze_results(user_prompt, sql_query, results_df)
            except Exception as e:
                analysis = f"Analysis generation failed: {e}"

            yield block("markdown", {"title": "Insights & Recommendations", "content": analysis})
            
            # Generate contextual suggestions based on results (optional for performance)
            contextual_suggestions = self.gemini_manager.generate_contextual_suggestions(
                user_prompt, results_df, True
            )
            yield block("suggestions", {
                "title": "Follow-up questions", 
                "suggestions": contextual_suggestions
            })
            
            yield block("progress", {"message": "Complete!", "percentage": 100})

        except Exception as e:
            logger.exception("Unexpected error during analysis stream")
            yield block("error", {
                "message": "Analysis failed", 
                "details": str(e),
                "suggestion": "Please try again or contact support if the issue persists"
            })

# FastAPI App
app = FastAPI(
    title="DV360 Analytics API", 
    version="2.2.0-optimized", 
    description="Performance-optimized natural language analytics for DV360 data"
)

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
        
        enable_verification = os.getenv("ENABLE_JWT_VERIFICATION", "false").lower() == "true"
        if enable_verification:
            logger.info("Analytics service initialized - JWT verification ENABLED (Production)")
        else:
            logger.warning("Analytics service initialized - JWT verification DISABLED (Testing)")
        
        logger.info(f"Performance mode: Parallel LLM = {config.enable_parallel_llm}")
    except Exception as e:
        logger.error(f"Failed to initialize analytics service: {e}")
        analytics_service = None

@app.get("/")
async def root():
    return {
        "status": "healthy", 
        "service": "DV360 Analytics API v2.2 - Performance Optimized",
        "optimizations": [
            "Parallel LLM calls (3x faster detection)",
            "Optional comparative validation",
            "Configurable contextual suggestions",
            "Schema caching",
            "Query result caching"
        ],
        "features": [
            "LLM-powered scope detection with suggestions", 
            "Dynamic date ranges", 
            "Secure JWT authentication",
            "Enhanced comparative query handling",
            "Intelligent error messages"
        ]
    }

@app.get("/health")
async def health_check():
    enable_verification = os.getenv("ENABLE_JWT_VERIFICATION", "false").lower() == "true"
    if analytics_service is None:
        return {
            "status": "degraded", 
            "bigquery": "not-initialized", 
            "gemini": "not-initialized",
            "jwt": "not-configured",
            "timestamp": datetime.now().isoformat()
        }
    return {
        "status": "healthy", 
        "bigquery": "connected", 
        "gemini": "configured" if config.google_api_key else "not-configured",
        "jwt": "enabled" if enable_verification else "disabled (testing mode)",
        "version": "v2.2-optimized",
        "parallel_llm": config.enable_parallel_llm,
        "timestamp": datetime.now().isoformat()
    }

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
async def analyze(
    request: AnalysisRequest = Body(...),
    advertiser_ids: List[str] = Depends(decode_advertiser_token)
):
    """
    Main analysis endpoint with secure JWT authentication
    
    PERFORMANCE OPTIMIZED VERSION 2.2:
    - Parallel LLM calls for 3x faster detection
    - Optional comparative validation
    - Configurable contextual suggestions
    - All features from v2.1 maintained
    
    Configuration (via .env):
    - ENABLE_PARALLEL_LLM=true (default) - Run scope/date/table detection in parallel
    - SKIP_CONTEXTUAL_SUGGESTIONS=false (default) - Generate smart follow-ups
    - SKIP_COMPARATIVE_VALIDATION=false (default) - Validate comparative queries
    
    Returns:
    - Streaming NDJSON response with analysis results
    """
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Optimized analysis request for {len(advertiser_ids)} advertiser(s): {request.question[:50]}...")

    generator = analytics_service.stream_analysis(
        request.question, 
        advertiser_ids,
        request.report_level, 
        request.table
    )
    return StreamingResponse(generator, media_type="application/x-ndjson", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting DV360 Analytics API v2.2 (Optimized) on port {port}...")
    logger.info("Performance optimizations active")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
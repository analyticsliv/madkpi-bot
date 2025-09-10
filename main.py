import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
import pandas as pd
from decimal import Decimal
import numpy as np
from google.cloud import bigquery
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal objects."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if pd.isna(obj):
            return None
        return super(DecimalEncoder, self).default(obj)

class Config:
    """Configuration class for the application."""
    def __init__(self):
        self.bq_project = os.getenv("BQ_PROJECT", "dx-api-project")
        self.bq_dataset = os.getenv("BQ_DATASET", "madkpi")
        self.bq_table = os.getenv("BQ_TABLE", "madkpi_data_sep_to_jan")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.fallback_models = ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
    def setup_credentials(self):
        """Setup Google Cloud credentials."""
        try:
            # For Cloud Run, use default credentials or service account
            self.bq_client = bigquery.Client(project=self.bq_project)
            
            # Configure Gemini API
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=gemini_api_key)
            logger.info("Credentials setup successful!")
            return True
        except Exception as e:
            logger.error(f"Credential setup failed: {e}")
            return False

class BigQueryManager:
    """Manages BigQuery operations for DV360 data."""
    
    def __init__(self, client, dataset: str, table: str):
        self.client = client
        self.dataset = dataset
        self.table = table
        self.available_columns = []
        self._initialize_columns()
    
    def _initialize_columns(self):
        """Initialize table columns."""
        try:
            self.available_columns = self._get_table_columns()
        except Exception as e:
            logger.error(f"Failed to initialize columns: {e}")
    
    def _get_table_columns(self) -> List[str]:
        """Get all available columns from the table with types."""
        try:
            query = f"""
            SELECT column_name, data_type
            FROM {self.client.project}.{self.dataset}.INFORMATION_SCHEMA.COLUMNS
            WHERE table_name = '{self.table.split('.')[-1]}'
            """
            query_job = self.client.query(query)
            results = query_job.result()
            return [f"{row.column_name} ({row.data_type})" for row in results]
        except Exception as e:
            logger.error(f"Failed to get table columns: {e}")
            return []
    
    def _convert_decimals_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Decimal objects to float in the DataFrame."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains Decimal objects
                if not df[col].empty and isinstance(df[col].iloc[0], Decimal):
                    df[col] = df[col].astype(float)
                # Also handle string representations of numbers that might be Decimals
                elif df[col].dtype == 'object':
                    try:
                        # Try to convert to numeric, coercing errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
        return df
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            query_job = self.client.query(sql_query)
            results = query_job.result()

            # Try to import db-dtypes to handle NUMERIC/BIGNUMERIC
            try:
                import db_dtypes  # noqa: F401
                df = results.to_dataframe(create_bqstorage_client=False)
            except ImportError:
                logger.warning("db-dtypes not found — falling back to string conversion")
                df = results.to_dataframe(create_bqstorage_client=False)
                df = df.astype(str)

            # Convert Decimal objects to float for JSON serialization
            df = self._convert_decimals_to_float(df)
            
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

    
    def get_sample_data(self, limit: int = 5) -> pd.DataFrame:
        """Get sample data for user reference."""
        query = f"SELECT * FROM {self.client.project}.{self.dataset}.{self.table} LIMIT {limit}"
        return self.execute_query(query)
    
    def get_table_schema(self) -> str:
        """Get table schema as string for Gemini context."""
        try:
            sample_df = self.get_sample_data(3)
            schema_info = f"Table: {self.client.project}.{self.dataset}.{self.table}\nColumns: {', '.join(self.available_columns)}\n"
            schema_info += f"Sample data:\n{sample_df.to_string()}"
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return f"Table: {self.client.project}.{self.dataset}.{self.table}\nColumns: {', '.join(self.available_columns)}"

class GeminiManager:
    """Manages Gemini AI interactions."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
    
    def generate_sql_from_prompt(self, user_prompt: str, table_schema: str) -> str:
        """Generate SQL query from natural language prompt."""
        prompt = f"""
        You are a BigQuery SQL expert. Given the following table schema and sample data:
        
        {table_schema}
        
        User request: {user_prompt}
        
        Generate a valid BigQuery SQL query that answers the user's question.
        Return ONLY the SQL query without any explanations or markdown formatting.
        Use the fully qualified table name (project.dataset.table) in the FROM clause.
        Use appropriate aggregations and filters.
        
        IMPORTANT: For numeric columns (Revenue, Cost, Impressions, etc.):
        - Always use CAST(column AS FLOAT64) instead of BIGNUMERIC or NUMERIC
        - This prevents Decimal serialization issues in the API response
        - Example: SELECT CAST(SUM(CAST(Revenue AS FLOAT64)) AS FLOAT64) as total_revenue
        
        For date columns, use BigQuery functions such as FORMAT_DATE(), DATE_TRUNC(), or PARSE_DATE().
        For any calculations that involve division, use SAFE_DIVIDE() to avoid division by zero errors.
        Important:
        - If the user provides only a partial advertiser name or campaign name, use a LIKE condition or CONTAINS_SUBSTR() instead of exact matches.
        - Always make searches case-insensitive by wrapping both sides with LOWER().
        - For example: WHERE LOWER(Advertiser_Name) LIKE LOWER('%pilgrim_hom%')
        """
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            sql_query = response.text.strip()
            # Clean up SQL query
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            return sql_query
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")
    
    def analyze_results(self, user_prompt: str, sql_query: str, results_df: pd.DataFrame) -> str:
        """Analyze query results and provide insights."""
        # Convert DataFrame to string representation to avoid Decimal serialization in the prompt
        df_sample = results_df.head(10).copy()
        
        # Convert all values to strings for safe string representation
        for col in df_sample.columns:
            df_sample[col] = df_sample[col].astype(str)
        
        prompt = f"""
        User's original question: {user_prompt}
        
        SQL query used: {sql_query}
        
        Query results (first 10 rows):
        {df_sample.to_string()}
        
        Please provide:
        1. A clear summary of what the data shows
        2. Key insights and trends
        3. Business recommendations
        4. Any data quality observations
        
        Format the response in clear, business-friendly language.
        """
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return f"Analysis generation failed: {str(e)}"

class AnalyticsService:
    """Main service class for DV360 analytics."""
    
    def __init__(self):
        self.config = Config()
        self.setup_complete = self.config.setup_credentials()
        
        if self.setup_complete:
            self.bq_manager = BigQueryManager(
                self.config.bq_client, 
                self.config.bq_dataset, 
                self.config.bq_table
            )
            self.gemini_manager = GeminiManager(self.config.gemini_model)
        else:
            raise Exception("Failed to setup credentials")
    
    def _safe_json_serialize(self, data):
        """Safely serialize data to JSON, handling Decimals and other non-serializable types."""
        try:
            return json.loads(json.dumps(data, cls=DecimalEncoder))
        except Exception as e:
            logger.error(f"JSON serialization error: {e}")
            # Fallback: convert problematic values to strings
            if isinstance(data, dict):
                return {k: str(v) if isinstance(v, (Decimal, np.integer, np.floating)) else v for k, v in data.items()}
            elif isinstance(data, list):
                return [str(item) if isinstance(item, (Decimal, np.integer, np.floating)) else item for item in data]
            else:
                return str(data)
    
    async def stream_analysis(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """Stream analysis results in NDJSON format."""
        try:
            # Step 1: Progress indicator
            yield self._create_block("progress", {
                "message": "Generating SQL query...",
                "percentage": 10
            })
            
            # Step 2: Generate SQL
            table_schema = self.bq_manager.get_table_schema()
            sql_query = self.gemini_manager.generate_sql_from_prompt(user_prompt, table_schema)
            
            if not sql_query:
                yield self._create_block("error", {
                    "message": "Failed to generate SQL query",
                    "details": "Please try rephrasing your question"
                })
                return
            
            # Step 3: Show generated SQL
            yield self._create_block("code", {
                "language": "sql",
                "content": sql_query,
                "title": "Generated SQL Query"
            })
            
            yield self._create_block("progress", {
                "message": "Executing query...",
                "percentage": 40
            })
            
            # Step 4: Execute query
            results_df = self.bq_manager.execute_query(sql_query)
            print("results_df", results_df)
            
            if results_df.empty:
                yield self._create_block("error", {
                    "message": "No results found",
                    "details": "The query returned no data. Try adjusting your filters or question."
                })
                return
            
            yield self._create_block("progress", {
                "message": "Processing results...",
                "percentage": 60
            })
            
            # Step 5: Show metrics
            yield self._create_block("metrics", {
                "title": "Query Results Summary",
                "metrics": [
                    {"label": "Total Rows", "value": len(results_df), "format": "number"},
                    {"label": "Columns", "value": len(results_df.columns), "format": "number"}
                ]
            })
            
            # Step 6: Show data table (limit to first 100 rows for performance)
            # Convert DataFrame to dict with aggressive Decimal handling
            table_data = []
            for _, row in results_df.head(100).iterrows():
                row_data = {}
                for col, value in row.items():
                    # Aggressive conversion of any potential problematic types
                    if isinstance(value, Decimal):
                        row_data[col] = float(value)
                    elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                        row_data[col] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                        row_data[col] = float(value)
                    elif pd.isna(value) or str(value).lower() in ['nan', 'none', 'null']:
                        row_data[col] = None
                    elif hasattr(value, 'item'):  # NumPy scalar
                        try:
                            row_data[col] = value.item()
                        except:
                            row_data[col] = str(value)
                    else:
                        # Try to convert string representations of Decimals
                        try:
                            if isinstance(value, str) and '.' in value and value.replace('.', '').replace('-', '').isdigit():
                                row_data[col] = float(value)
                            else:
                                row_data[col] = value
                        except:
                            row_data[col] = str(value) if value is not None else None
                
                # Final safety check for the entire row
                try:
                    json.dumps(row_data, cls=DecimalEncoder)
                    table_data.append(row_data)
                except Exception as e:
                    logger.warning(f"Row serialization issue, converting to strings: {e}")
                    # Convert entire row to strings as fallback
                    safe_row = {k: str(v) if v is not None else None for k, v in row_data.items()}
                    table_data.append(safe_row)
            
            columns = [{"key": col, "label": col, "sortable": True} for col in results_df.columns]
            
            yield self._create_block("table", {
                "title": f"Results (showing first {min(100, len(results_df))} rows)",
                "columns": columns,
                "data": table_data,
                "pagination": True
            })
            
            yield self._create_block("progress", {
                "message": "Generating insights...",
                "percentage": 80
            })
            
            # Step 7: Generate analysis
            analysis = self.gemini_manager.analyze_results(user_prompt, sql_query, results_df)
            
            yield self._create_block("markdown", {
                "content": f"## AI Analysis\n\n{analysis}",
                "title": "Insights & Recommendations"
            })
            
            # Step 8: Suggest follow-up questions
            suggestions = self._generate_suggestions(user_prompt, results_df)
            yield self._create_block("suggestions", {
                "title": "Suggested follow-up questions",
                "suggestions": suggestions
            })
            
            yield self._create_block("progress", {
                "message": "Analysis complete!",
                "percentage": 100
            })
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            yield self._create_block("error", {
                "message": "Analysis failed",
                "details": str(e)
            })
    
    def _create_block(self, block_type: str, data: Dict) -> str:
        """Create a block in NDJSON format."""
        block = {
            "type": block_type,
            "timestamp": datetime.now().isoformat(),
            "data": self._safe_json_serialize(data)
        }
        return json.dumps(block, cls=DecimalEncoder) + "\n"
    
    def _generate_suggestions(self, original_prompt: str, results_df: pd.DataFrame) -> List[str]:
        """Generate follow-up question suggestions."""
        suggestions = [
            "Show me the trend over time for this data",
            "Break down these results by different dimensions",
            "What are the top performing segments?",
            "How does this compare to last period?",
            "Show me the detailed breakdown by campaign"
        ]
        return suggestions[:3]  # Limit to 3 suggestions

# FastAPI app setup
app = FastAPI(
    title="DV360 Analytics API",
    description="Natural language analytics for DV360 data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AnalysisRequest(BaseModel):
    question: str

# Global service instance
analytics_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize the analytics service on startup."""
    global analytics_service
    try:
        analytics_service = AnalyticsService()
        logger.info("Analytics service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analytics service: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "DV360 Analytics API"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "bigquery": "connected",
        "gemini": "connected",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Analyze data with streaming response."""
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    return StreamingResponse(
        analytics_service.stream_analysis(request.question),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.get("/schema")
async def get_schema():
    """Get table schema information."""
    if analytics_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        schema_info = analytics_service.bq_manager.get_table_schema()
        sample_data = analytics_service.bq_manager.get_sample_data(5)
        
        return {
            "schema": schema_info,
            "sample_data": sample_data.to_dict('records'),
            "columns": analytics_service.bq_manager.available_columns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
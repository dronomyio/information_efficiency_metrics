# python/api.py
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import redis
import psycopg2
from contextlib import asynccontextmanager
import os

# Import C++ extensions
import info_efficiency_cpp as cpp_lib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data with nanosecond precision"""
    symbol: str
    timestamps: np.ndarray  # int64 nanoseconds since epoch
    prices: np.ndarray
    volumes: np.ndarray
    returns: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(self.timestamps, unit='ns'),
            'price': self.prices,
            'volume': self.volumes
        })
        if self.returns is not None:
            df['returns'] = self.returns
        return df.set_index('timestamp')


class InfoEfficiencyAnalyzer:
    """Main API class for information efficiency analysis"""
    
    def __init__(
        self,
        polygon_api_key: str,
        num_gpus: Optional[int] = None,
        cache_enabled: bool = True,
        db_config: Optional[Dict] = None
    ):
        self.api_key = polygon_api_key
        self.cache_enabled = cache_enabled
        
        # Initialize C++ components
        self.vr_calculator = cpp_lib.create_variance_ratio_calculator()
        self.acf_processor = cpp_lib.create_autocorr_processor(os.cpu_count())
        self.data_reader = cpp_lib.create_polygon_reader(polygon_api_key)
        
        # Initialize cache
        if cache_enabled:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=6379,
                decode_responses=True
            )
        
        # Initialize database connection
        if db_config:
            self.db_conn = psycopg2.connect(**db_config)
        else:
            self.db_conn = None
    
    async def fetch_market_data(
        self,
        symbol: str,
        date: str,
        data_type: str = 'trades'
    ) -> MarketData:
        """Fetch market data from Polygon.io"""
        
        # Check cache first
        cache_key = f"{symbol}:{date}:{data_type}"
        if self.cache_enabled:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for {cache_key}")
                return self._deserialize_market_data(cached_data)
        
        # Fetch from Polygon
        logger.info(f"Fetching {data_type} data for {symbol} on {date}")
        
        if data_type == 'trades':
            prices, volumes, timestamps, count = cpp_lib.read_trades_data(
                self.data_reader, symbol, date
            )
        else:  # quotes
            prices, volumes, timestamps, count = cpp_lib.read_quotes_data(
                self.data_reader, symbol, date
            )
        
        market_data = MarketData(
            symbol=symbol,
            timestamps=np.array(timestamps[:count]),
            prices=np.array(prices[:count]),
            volumes=np.array(volumes[:count])
        )
        
        # Cache the data
        if self.cache_enabled:
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                self._serialize_market_data(market_data)
            )
        
        return market_data
    
    def compute_returns(
        self,
        market_data: MarketData,
        interval_ns: int = 1_000_000_000  # 1 second default
    ) -> np.ndarray:
        """Compute returns at specified nanosecond intervals"""
        
        returns = cpp_lib.compute_returns_from_trades(
            self.data_reader,
            market_data.symbol,
            market_data.timestamps[0],  # Use first timestamp as date proxy
            interval_ns
        )
        
        return np.array(returns)
    
    def calculate_variance_ratios(
        self,
        returns: np.ndarray,
        horizons: List[int]
    ) -> Dict[int, float]:
        """Calculate variance ratios for multiple horizons using multi-GPU"""
        
        logger.info(f"Computing variance ratios for {len(horizons)} horizons")
        
        vr_results = cpp_lib.compute_variance_ratios(
            self.vr_calculator,
            returns,
            horizons
        )
        
        return dict(zip(horizons, vr_results))
    
    def calculate_autocorrelations(
        self,
        returns: np.ndarray,
        max_lag: int = 100
    ) -> Tuple[np.ndarray, float, float]:
        """Calculate autocorrelations and decay parameters using SIMD"""
        
        logger.info(f"Computing autocorrelations up to lag {max_lag}")
        
        acf = cpp_lib.compute_autocorrelations(
            self.acf_processor,
            returns,
            max_lag
        )
        
        # Estimate decay parameters
        phi, half_life = self._estimate_decay_params(acf)
        
        return acf, phi, half_life
    
    def _estimate_decay_params(self, acf: np.ndarray) -> Tuple[float, float]:
        """Estimate exponential decay parameters from ACF"""
        
        # Fit exponential decay to log(|ACF|)
        lags = np.arange(1, len(acf))
        log_acf = np.log(np.abs(acf[1:]) + 1e-10)
        
        # Linear regression
        coeffs = np.polyfit(lags, log_acf, 1)
        phi = np.exp(coeffs[0])
        half_life = np.log(0.5) / coeffs[0] if coeffs[0] != 0 else np.inf
        
        return phi, half_life
    
    async def batch_analysis(
        self,
        symbols: List[str],
        dates: List[str],
        metrics: List[str] = ['vr', 'acf']
    ) -> pd.DataFrame:
        """Perform batch analysis across multiple symbols and dates"""
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                for date in dates:
                    tasks.append(
                        self._analyze_single(symbol, date, metrics, session)
                    )
            
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in completed:
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    logger.error(f"Analysis failed: {result}")
        
        return pd.DataFrame(results)
    
    async def _analyze_single(
        self,
        symbol: str,
        date: str,
        metrics: List[str],
        session: aiohttp.ClientSession
    ) -> Dict:
        """Analyze single symbol-date combination"""
        
        # Fetch data
        market_data = await self.fetch_market_data(symbol, date)
        
        # Compute returns
        returns = self.compute_returns(market_data)
        
        result = {
            'symbol': symbol,
            'date': date,
            'num_observations': len(returns)
        }
        
        # Compute requested metrics
        if 'vr' in metrics:
            horizons = [2, 5, 10, 20, 50, 100]
            vr_results = self.calculate_variance_ratios(returns, horizons)
            for h, vr in vr_results.items():
                result[f'vr_{h}'] = vr
        
        if 'acf' in metrics:
            acf, phi, half_life = self.calculate_autocorrelations(returns)
            result['acf_phi'] = phi
            result['acf_half_life'] = half_life
            result['acf_lag1'] = acf[1] if len(acf) > 1 else np.nan
        
        # Store results in database
        if self.db_conn:
            self._store_results(result)
        
        return result
    
    def _store_results(self, result: Dict):
        """Store analysis results in PostgreSQL"""
        
        if not self.db_conn:
            return
        
        cursor = self.db_conn.cursor()
        
        query = """
        INSERT INTO efficiency_metrics 
        (symbol, date, num_obs, vr_2, vr_5, vr_10, vr_20, vr_50, vr_100,
         acf_phi, acf_half_life, acf_lag1, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (symbol, date) DO UPDATE SET
            num_obs = EXCLUDED.num_obs,
            vr_2 = EXCLUDED.vr_2,
            vr_5 = EXCLUDED.vr_5,
            vr_10 = EXCLUDED.vr_10,
            vr_20 = EXCLUDED.vr_20,
            vr_50 = EXCLUDED.vr_50,
            vr_100 = EXCLUDED.vr_100,
            acf_phi = EXCLUDED.acf_phi,
            acf_half_life = EXCLUDED.acf_half_life,
            acf_lag1 = EXCLUDED.acf_lag1,
            created_at = NOW()
        """
        
        values = (
            result['symbol'],
            result['date'],
            result.get('num_observations', 0),
            result.get('vr_2'),
            result.get('vr_5'),
            result.get('vr_10'),
            result.get('vr_20'),
            result.get('vr_50'),
            result.get('vr_100'),
            result.get('acf_phi'),
            result.get('acf_half_life'),
            result.get('acf_lag1')
        )
        
        cursor.execute(query, values)
        self.db_conn.commit()
        cursor.close()
    
    def _serialize_market_data(self, data: MarketData) -> str:
        """Serialize market data for caching"""
        return json.dumps({
            'symbol': data.symbol,
            'timestamps': data.timestamps.tolist(),
            'prices': data.prices.tolist(),
            'volumes': data.volumes.tolist()
        })
    
    def _deserialize_market_data(self, data_str: str) -> MarketData:
        """Deserialize market data from cache"""
        data = json.loads(data_str)
        return MarketData(
            symbol=data['symbol'],
            timestamps=np.array(data['timestamps']),
            prices=np.array(data['prices']),
            volumes=np.array(data['volumes'])
        )
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'vr_calculator'):
            cpp_lib.destroy_variance_ratio_calculator(self.vr_calculator)
        if hasattr(self, 'acf_processor'):
            cpp_lib.destroy_autocorr_processor(self.acf_processor)
        if hasattr(self, 'data_reader'):
            cpp_lib.destroy_polygon_reader(self.data_reader)
        if self.db_conn:
            self.db_conn.close()


# FastAPI application
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Information Efficiency Analysis API")

# Global analyzer instance
analyzer = None


class AnalysisRequest(BaseModel):
    symbols: List[str]
    dates: List[str]
    metrics: List[str] = ['vr', 'acf']
    interval_ns: int = 1_000_000_000  # 1 second default


class AnalysisResponse(BaseModel):
    status: str
    results: Optional[Dict] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize analyzer on startup"""
    global analyzer
    analyzer = InfoEfficiencyAnalyzer(
        polygon_api_key=os.getenv('POLYGON_API_KEY'),
        cache_enabled=True,
        db_config={
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': 'info_efficiency',
            'user': 'admin',
            'password': os.getenv('DB_PASSWORD')
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Run efficiency analysis"""
    
    try:
        # Run analysis
        results = await analyzer.batch_analysis(
            symbols=request.symbols,
            dates=request.dates,
            metrics=request.metrics
        )
        
        return AnalysisResponse(
            status="success",
            results=results.to_dict()
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{symbol}/{date}")
async def get_results(symbol: str, date: str):
    """Retrieve cached results"""
    
    if not analyzer.db_conn:
        raise HTTPException(status_code=503, detail="Database not available")
    
    cursor = analyzer.db_conn.cursor()
    query = """
    SELECT * FROM efficiency_metrics
    WHERE symbol = %s AND date = %s
    """
    cursor.execute(query, (symbol, date))
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        return JSONResponse(content=dict(result))
    else:
        raise HTTPException(status_code=404, detail="Results not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

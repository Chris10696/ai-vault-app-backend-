"""
Pydantic models for differential privacy operations.
Defines data structures for privacy queries, responses, and user profiles.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator
import uuid
from enum import Enum

class PrivacyLevel(str, Enum):
    """Privacy protection levels."""
    MINIMAL = "minimal"      # ε = 1.0
    STANDARD = "standard"    # ε = 0.5  
    HIGH = "high"           # ε = 0.1
    MAXIMUM = "maximum"     # ε = 0.01

class QueryType(str, Enum):
    """Types of privacy-preserving queries."""
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    HISTOGRAM = "histogram"
    RANGE_QUERY = "range_query"

class PrivacyParameters(BaseModel):
    """Privacy parameters for differential privacy operations."""
    epsilon: float = Field(..., gt=0, le=10, description="Privacy budget (smaller = more private)")
    delta: float = Field(default=1e-5, ge=0, lt=1, description="Failure probability")
    sensitivity: float = Field(..., gt=0, description="Query sensitivity")
    
    @validator('epsilon')
    def validate_epsilon(cls, v):
        """Validate epsilon parameter."""
        if v <= 0 or v > 10:
            raise ValueError("Epsilon must be between 0 and 10")
        return v
    
    @validator('delta')
    def validate_delta(cls, v):
        """Validate delta parameter."""
        if v < 0 or v >= 1:
            raise ValueError("Delta must be between 0 and 1")
        return v

class PrivacyQuery(BaseModel):
    """Model for privacy-preserving query requests."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    query_type: QueryType
    dataset_id: str
    query_params: Dict[str, Any] = Field(default_factory=dict)
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    custom_epsilon: Optional[float] = None
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('custom_epsilon')
    def validate_custom_epsilon(cls, v):
        """Validate custom epsilon if provided."""
        if v is not None and (v <= 0 or v > 10):
            raise ValueError("Custom epsilon must be between 0 and 10")
        return v

class PrivacyResponse(BaseModel):
    """Model for privacy-preserving query responses."""
    query_id: str
    user_id: str
    result: Union[float, int, List[float], Dict[str, Any]]
    epsilon_used: float
    noise_added: float
    original_result: Optional[Union[float, int, List[float]]] = None  # Only for testing
    accuracy_loss: Optional[float] = None
    remaining_budget: float
    responded_at: datetime = Field(default_factory=datetime.utcnow)
    
class UserPrivacyProfile(BaseModel):
    """User privacy profile and preferences."""
    user_id: str
    preferred_privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    total_epsilon_budget: float = Field(default=10.0, gt=0)
    daily_epsilon_limit: float = Field(default=2.0, gt=0)
    epsilon_used_today: float = Field(default=0.0, ge=0)
    epsilon_used_total: float = Field(default=0.0, ge=0)
    last_reset_date: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('daily_epsilon_limit')
    def validate_daily_limit(cls, v, values):
        """Ensure daily limit doesn't exceed total budget."""
        total = values.get('total_epsilon_budget', 10.0)
        if v > total:
            raise ValueError("Daily limit cannot exceed total budget")
        return v

class DatasetMetadata(BaseModel):
    """Metadata for datasets used in privacy queries."""
    dataset_id: str
    name: str
    description: str
    record_count: int
    columns: List[str]
    data_types: Dict[str, str]
    sensitivity_levels: Dict[str, float]  # Column sensitivity mapping
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PrivacyAuditLog(BaseModel):
    """Audit log entry for privacy operations."""
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    operation_type: str
    query_id: Optional[str] = None
    epsilon_consumed: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VaultEntry(BaseModel):
    """Encrypted vault entry model."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    encrypted_data: str  # Base64 encoded encrypted JSON
    data_type: str = "generic"
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    
class VaultCreateRequest(BaseModel):
    """Request model for creating vault entries."""
    title: str = Field(..., min_length=1, max_length=200)
    data: Dict[str, Any]
    data_type: str = "generic"
    tags: List[str] = Field(default_factory=list)
    
class VaultUpdateRequest(BaseModel):
    """Request model for updating vault entries."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    data: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

"""
Privacy budget monitoring and alerting system.
Provides real-time monitoring and proactive alerts for budget management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

from .analytics import privacy_analytics, UsageStats
from .budget import budget_manager

logger = logging.getLogger(__name__)

class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Types of privacy alerts."""
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNUSUAL_USAGE = "unusual_usage"
    HIGH_EPSILON_REQUEST = "high_epsilon_request"
    RAPID_CONSUMPTION = "rapid_consumption"

@dataclass
class PrivacyAlert:
    """Privacy monitoring alert."""
    alert_id: str
    user_id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    metadata: Dict
    timestamp: datetime
    acknowledged: bool = False

class BudgetMonitor:
    """
    Real-time privacy budget monitoring system.
    Tracks usage patterns and generates alerts for unusual activity.
    """
    
    def __init__(self):
        self.alert_handlers: List[Callable] = []
        self.monitoring_active = False
        self.alert_thresholds = {
            "daily_budget_warning": 0.8,  # 80% of daily budget
            "daily_budget_critical": 0.95,  # 95% of daily budget
            "total_budget_warning": 0.7,   # 70% of total budget
            "total_budget_critical": 0.9,  # 90% of total budget
            "high_epsilon_threshold": 1.0,  # Epsilon > 1.0
            "rapid_consumption_window": 300,  # 5 minutes
            "rapid_consumption_threshold": 0.5  # 50% of daily budget in window
        }
        logger.info("BudgetMonitor initialized")
    
    def add_alert_handler(self, handler: Callable[[PrivacyAlert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    async def start_monitoring(self):
        """Start the monitoring loop."""
        self.monitoring_active = True
        logger.info("Budget monitoring started")
        
        # Run monitoring loop
        while self.monitoring_active:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.monitoring_active = False
        logger.info("Budget monitoring stopped")
    
    async def _monitoring_cycle(self):
        """Execute one monitoring cycle."""
        try:
            # Get all active users (users with recent activity)
            active_users = await self._get_active_users()
            
            for user_id in active_users:
                await self._check_user_budget_status(user_id)
                await self._check_usage_patterns(user_id)
                
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
    
    async def _get_active_users(self) -> List[str]:
        """Get list of users with recent activity."""
        try:
            # Users with queries in the last 24 hours
            if not privacy_analytics.supabase:
                return []
            
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            result = privacy_analytics.supabase.table("privacy_queries").select(
                "user_id"
            ).gte("requested_at", cutoff_time.isoformat()).execute()
            
            if result.data:
                return list(set(row["user_id"] for row in result.data))
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get active users: {e}")
            return []
    
    async def _check_user_budget_status(self, user_id: str):
        """Check budget status for a user and generate alerts if needed."""
        try:
            # Get current budget status
            status = await budget_manager.get_user_budget_status(user_id)
            
            if "error" in status:
                return
            
            daily_utilization = status["daily_budget"]["utilization"]
            total_utilization = status["total_budget"]["utilization"]
            
            # Check daily budget thresholds
            if daily_utilization >= self.alert_thresholds["daily_budget_critical"]:
                await self._create_alert(
                    user_id,
                    AlertType.BUDGET_EXHAUSTED,
                    AlertLevel.CRITICAL,
                    "Daily Privacy Budget Nearly Exhausted",
                    f"You have used {daily_utilization:.1f}% of your daily privacy budget. "
                    f"Remaining: {status['daily_budget']['remaining']:.4f} epsilon.",
                    {"budget_type": "daily", "utilization": daily_utilization}
                )
            elif daily_utilization >= self.alert_thresholds["daily_budget_warning"]:
                await self._create_alert(
                    user_id,
                    AlertType.BUDGET_WARNING,
                    AlertLevel.WARNING,
                    "Daily Privacy Budget Warning",
                    f"You have used {daily_utilization:.1f}% of your daily privacy budget. "
                    f"Consider optimizing your queries or increasing your limit.",
                    {"budget_type": "daily", "utilization": daily_utilization}
                )
            
            # Check total budget thresholds
            if total_utilization >= self.alert_thresholds["total_budget_critical"]:
                await self._create_alert(
                    user_id,
                    AlertType.BUDGET_EXHAUSTED,
                    AlertLevel.CRITICAL,
                    "Total Privacy Budget Nearly Exhausted",
                    f"You have used {total_utilization:.1f}% of your total privacy budget. "
                    f"Remaining: {status['total_budget']['remaining']:.4f} epsilon.",
                    {"budget_type": "total", "utilization": total_utilization}
                )
            elif total_utilization >= self.alert_thresholds["total_budget_warning"]:
                await self._create_alert(
                    user_id,
                    AlertType.BUDGET_WARNING,
                    AlertLevel.WARNING,
                    "Total Privacy Budget Warning",
                    f"You have used {total_utilization:.1f}% of your total privacy budget. "
                    f"Consider reviewing your privacy settings.",
                    {"budget_type": "total", "utilization": total_utilization}
                )
                
        except Exception as e:
            logger.error(f"Failed to check budget status for user {user_id}: {e}")
    
    async def _check_usage_patterns(self, user_id: str):
        """Check for unusual usage patterns."""
        try:
            # Check for rapid consumption in recent window
            rapid_consumption = await self._check_rapid_consumption(user_id)
            if rapid_consumption:
                await self._create_alert(
                    user_id,
                    AlertType.RAPID_CONSUMPTION,
                    AlertLevel.WARNING,
                    "Rapid Privacy Budget Consumption",
                    f"Unusual spike in privacy budget usage detected. "
                    f"You've consumed {rapid_consumption['epsilon_consumed']:.4f} epsilon "
                    f"in the last {rapid_consumption['window_minutes']} minutes.",
                    rapid_consumption
                )
            
            # Check recent high epsilon requests
            high_epsilon_requests = await self._check_high_epsilon_requests(user_id)
            if high_epsilon_requests:
                await self._create_alert(
                    user_id,
                    AlertType.HIGH_EPSILON_REQUEST,
                    AlertLevel.INFO,
                    "High Epsilon Value Requests",
                    f"You've made {len(high_epsilon_requests)} requests with high epsilon values "
                    f"(> {self.alert_thresholds['high_epsilon_threshold']}) in the last hour. "
                    f"Consider using lower privacy levels for better protection.",
                    {"high_epsilon_count": len(high_epsilon_requests)}
                )
                
        except Exception as e:
            logger.error(f"Failed to check usage patterns for user {user_id}: {e}")
    
    async def _check_rapid_consumption(self, user_id: str) -> Optional[Dict]:
        """Check for rapid budget consumption."""
        try:
            if not privacy_analytics.supabase:
                return None
            
            window_minutes = self.alert_thresholds["rapid_consumption_window"] // 60
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            
            # Get recent queries
            result = privacy_analytics.supabase.table("privacy_queries").select(
                "epsilon_used, requested_at"
            ).eq("user_id", user_id).gte(
                "requested_at", cutoff_time.isoformat()
            ).eq("status", "completed").execute()
            
            if not result.data or len(result.data) < 3:  # Need multiple queries for pattern
                return None
            
            total_epsilon = sum(float(q["epsilon_used"]) for q in result.data)
            
            # Get user's daily limit
            status = await budget_manager.get_user_budget_status(user_id)
            if "error" in status:
                return None
            
            daily_limit = status["daily_budget"]["limit"]
            consumption_rate = total_epsilon / daily_limit
            
            if consumption_rate >= self.alert_thresholds["rapid_consumption_threshold"]:
                return {
                    "epsilon_consumed": total_epsilon,
                    "window_minutes": window_minutes,
                    "consumption_rate": consumption_rate,
                    "query_count": len(result.data)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check rapid consumption for user {user_id}: {e}")
            return None
    
    async def _check_high_epsilon_requests(self, user_id: str) -> List[Dict]:
        """Check for recent high epsilon requests."""
        try:
            if not privacy_analytics.supabase:
                return []
            
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            result = privacy_analytics.supabase.table("privacy_queries").select(
                "query_id, epsilon_used, query_type, requested_at"
            ).eq("user_id", user_id).gte(
                "requested_at", cutoff_time.isoformat()
            ).gte(
                "epsilon_used", self.alert_thresholds["high_epsilon_threshold"]
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to check high epsilon requests for user {user_id}: {e}")
            return []
    
    async def _create_alert(
        self, 
        user_id: str, 
        alert_type: AlertType, 
        level: AlertLevel,
        title: str, 
        message: str, 
        metadata: Dict
    ):
        """Create and dispatch an alert."""
        try:
            alert = PrivacyAlert(
                alert_id=f"alert_{user_id}_{alert_type}_{int(datetime.utcnow().timestamp())}",
                user_id=user_id,
                alert_type=alert_type,
                level=level,
                title=title,
                message=message,
                metadata=metadata,
                timestamp=datetime.utcnow()
            )
            
            # Store alert in database
            await self._store_alert(alert)
            
            # Dispatch to handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            logger.info(f"Created alert: {alert.alert_type} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _store_alert(self, alert: PrivacyAlert):
        """Store alert in database."""
        try:
            if not privacy_analytics.supabase:
                return
            
            alert_data = {
                "alert_id": alert.alert_id,
                "user_id": alert.user_id,
                "alert_type": alert.alert_type,
                "level": alert.level,
                "title": alert.title,
                "message": alert.message,
                "metadata": alert.metadata,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged
            }
            
            privacy_analytics.supabase.table("privacy_alerts").insert(alert_data).execute()
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")

class AlertHandlers:
    """Collection of alert handler functions."""
    
    @staticmethod
    def console_handler(alert: PrivacyAlert):
        """Print alert to console."""
        print(f"[{alert.level.upper()}] {alert.title}")
        print(f"User: {alert.user_id}")
        print(f"Message: {alert.message}")
        print(f"Time: {alert.timestamp}")
        print("-" * 50)
    
    @staticmethod
    async def email_handler(alert: PrivacyAlert):
        """Send alert via email (if configured)."""
        try:
            # Email configuration from environment
            smtp_server = os.getenv("SMTP_SERVER")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_username = os.getenv("SMTP_USERNAME")
            smtp_password = os.getenv("SMTP_PASSWORD")
            from_email = os.getenv("ALERT_FROM_EMAIL")
            
            # Get user email (would need to be implemented)
            to_email = await AlertHandlers._get_user_email(alert.user_id)
            
            if not all([smtp_server, smtp_username, smtp_password, from_email, to_email]):
                logger.warning("Email configuration incomplete, skipping email alert")
                return
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = f"[AI Vault Privacy Alert] {alert.title}"
            
            body = f"""
            Privacy Alert for your AI Vault account:
            
            Alert Type: {alert.alert_type}
            Level: {alert.level.upper()}
            Time: {alert.timestamp}
            
            {alert.message}
            
            Please review your privacy settings in the AI Vault dashboard.
            
            Best regards,
            AI Vault Privacy Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    @staticmethod
    async def _get_user_email(user_id: str) -> Optional[str]:
        """Get user email address."""
        try:
            if not privacy_analytics.supabase:
                return None
            
            result = privacy_analytics.supabase.table("users").select("email").eq("id", user_id).execute()
            
            if result.data:
                return result.data[0]["email"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user email for {user_id}: {e}")
            return None

# Global monitor instance
budget_monitor = BudgetMonitor()

# Add default handlers
budget_monitor.add_alert_handler(AlertHandlers.console_handler)
budget_monitor.add_alert_handler(AlertHandlers.email_handler)

# Additional SQL for alerts table
ALERTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS privacy_alerts (
    alert_id VARCHAR(200) PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL CHECK (level IN ('info', 'warning', 'critical')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_privacy_alerts_user_timestamp ON privacy_alerts(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_privacy_alerts_level ON privacy_alerts(level);
CREATE INDEX IF NOT EXISTS idx_privacy_alerts_acknowledged ON privacy_alerts(acknowledged);
"""

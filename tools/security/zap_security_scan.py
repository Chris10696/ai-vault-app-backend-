"""
OWASP ZAP security scanning automation for AI Vault API.
Provides automated security testing as part of CI/CD pipeline.
"""

import time
import json
import logging
import sys
import os
from typing import Dict, List, Any
from zapv2 import ZAPv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZAPSecurityScanner:
    """
    OWASP ZAP security scanner for automated API security testing.
    Integrates with CI/CD pipeline for continuous security validation.
    """
    
    def __init__(self, target_url: str, api_key: str = None):
        """
        Initialize ZAP scanner.
        
        Args:
            target_url: Base URL of API to scan
            api_key: ZAP API key (optional)
        """
        self.target_url = target_url.rstrip('/')
        self.zap = ZAPv2(
            apikey=api_key,
            proxies={
                'http': 'http://localhost:8080',
                'https': 'http://localhost:8080'
            }
        )
        self.scan_results = {}
        
    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """
        Run comprehensive security scan including spider, passive, and active scans.
        
        Returns:
            Comprehensive scan results
        """
        logger.info(f"Starting comprehensive security scan for {self.target_url}")
        
        try:
            # Step 1: Access target URL
            self._access_target()
            
            # Step 2: Spider scan
            spider_results = self._run_spider_scan()
            
            # Step 3: Wait for passive scanning
            self._wait_for_passive_scan()
            
            # Step 4: Active scan
            active_results = self._run_active_scan()
            
            # Step 5: Get results
            alerts = self._get_alerts()
            
            # Compile results
            self.scan_results = {
                "target_url": self.target_url,
                "scan_timestamp": time.time(),
                "spider_results": spider_results,
                "active_results": active_results,
                "alerts": alerts,
                "summary": self._generate_summary(alerts)
            }
            
            logger.info("Security scan completed successfully")
            return self.scan_results
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            raise

    def _access_target(self):
        """Access target URL to initialize ZAP session."""
        logger.info("Accessing target URL...")
        self.zap.urlopen(self.target_url)
        time.sleep(2)

    def _run_spider_scan(self) -> Dict[str, Any]:
        """
        Run spider scan to discover all accessible URLs.
        
        Returns:
            Spider scan results
        """
        logger.info("Starting spider scan...")
        
        spider_id = self.zap.spider.scan(self.target_url)
        time.sleep(2)
        
        # Monitor spider progress
        while int(self.zap.spider.status(spider_id)) < 100:
            progress = self.zap.spider.status(spider_id)
            logger.info(f"Spider scan progress: {progress}%")
            time.sleep(3)
        
        logger.info("Spider scan completed")
        
        # Get discovered URLs
        urls = self.zap.spider.results(spider_id)
        
        return {
            "spider_id": spider_id,
            "discovered_urls": urls,
            "url_count": len(urls)
        }

    def _wait_for_passive_scan(self):
        """Wait for passive scanning to complete."""
        logger.info("Waiting for passive scan to complete...")
        
        while int(self.zap.pscan.records_to_scan) > 0:
            records_left = self.zap.pscan.records_to_scan
            logger.info(f"Passive scan records remaining: {records_left}")
            time.sleep(2)
        
        logger.info("Passive scan completed")

    def _run_active_scan(self) -> Dict[str, Any]:
        """
        Run active security scan for vulnerability detection.
        
        Returns:
            Active scan results
        """
        logger.info("Starting active scan...")
        
        scan_id = self.zap.ascan.scan(self.target_url)
        time.sleep(2)
        
        # Monitor active scan progress
        while int(self.zap.ascan.status(scan_id)) < 100:
            progress = self.zap.ascan.status(scan_id)
            logger.info(f"Active scan progress: {progress}%")
            time.sleep(5)
        
        logger.info("Active scan completed")
        
        return {
            "scan_id": scan_id,
            "status": "completed"
        }

    def _get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all security alerts found during scan.
        
        Returns:
            List of security alerts
        """
        logger.info("Retrieving security alerts...")
        
        alerts = self.zap.core.alerts()
        
        # Parse and categorize alerts
        processed_alerts = []
        for alert in alerts:
            processed_alert = {
                "alert_id": alert.get("id"),
                "name": alert.get("alert"),
                "risk": alert.get("risk"),
                "confidence": alert.get("confidence"),
                "url": alert.get("url"),
                "description": alert.get("description"),
                "solution": alert.get("solution"),
                "reference": alert.get("reference"),
                "cwe_id": alert.get("cweid"),
                "wasc_id": alert.get("wascid")
            }
            processed_alerts.append(processed_alert)
        
        logger.info(f"Found {len(processed_alerts)} security alerts")
        return processed_alerts

    def _generate_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary of scan results.
        
        Args:
            alerts: List of security alerts
            
        Returns:
            Scan summary
        """
        summary = {
            "total_alerts": len(alerts),
            "risk_breakdown": {"High": 0, "Medium": 0, "Low": 0, "Informational": 0},
            "critical_issues": [],
            "recommendations": []
        }
        
        # Count alerts by risk level
        for alert in alerts:
            risk = alert.get("risk", "Informational")
            if risk in summary["risk_breakdown"]:
                summary["risk_breakdown"][risk] += 1
        
        # Identify critical issues (High risk)
        critical_alerts = [alert for alert in alerts if alert.get("risk") == "High"]
        summary["critical_issues"] = critical_alerts[:5]  # Top 5 critical issues
        
        # Generate recommendations
        if summary["risk_breakdown"]["High"] > 0:
            summary["recommendations"].append("Address high-risk vulnerabilities immediately")
        
        if summary["risk_breakdown"]["Medium"] > 5:
            summary["recommendations"].append("Review and fix medium-risk vulnerabilities")
        
        if len(alerts) == 0:
            summary["recommendations"].append("No security issues found - good security posture")
        
        return summary

    def save_results(self, output_file: str = "zap_scan_results.json"):
        """
        Save scan results to JSON file.
        
        Args:
            output_file: Output file path
        """
        if not self.scan_results:
            logger.warning("No scan results to save")
            return
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.scan_results, f, indent=2)
            
            logger.info(f"Scan results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def generate_html_report(self, output_file: str = "zap_scan_report.html"):
        """Generate HTML report."""
        try:
            html_report = self.zap.core.htmlreport()
            
            with open(output_file, 'w') as f:
                f.write(html_report)
            
            logger.info(f"HTML report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")

def main():
    """Main entry point for security scanning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Vault Security Scanner")
    parser.add_argument("--target", required=True, help="Target URL to scan")
    parser.add_argument("--api-key", help="ZAP API key")
    parser.add_argument("--output", default="zap_results.json", help="Output file")
    parser.add_argument("--fail-on-high", action="store_true", help="Fail if high-risk vulnerabilities found")
    
    args = parser.parse_args()
    
    try:
        # Initialize scanner
        scanner = ZAPSecurityScanner(args.target, args.api_key)
        
        # Run scan
        results = scanner.run_comprehensive_scan()
        
        # Save results
        scanner.save_results(args.output)
        scanner.generate_html_report("scan_report.html")
        
        # Check for high-risk issues
        high_risk_count = results["summary"]["risk_breakdown"]["High"]
        
        if args.fail_on_high and high_risk_count > 0:
            logger.error(f"Found {high_risk_count} high-risk vulnerabilities")
            sys.exit(1)
        
        logger.info("Security scan completed successfully")
        
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

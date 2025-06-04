#!/usr/bin/env python3

"""
Google Cloud Project Manager Script with Binary Authorization Support
Handles creation, deletion, and checking of Google Cloud resources including
projects, VPCs, subnets, and GKE clusters with Binary Authorization enabled.

Enhanced with JSON output for tracking created resources.

Usage examples are provided in the help text (--help).
"""

import argparse
import configparser
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime


# Configuration constants
class Constants:
    """Application constants"""
    DEFAULT_API_ENABLE_WAIT = 15
    MIN_RANDOM_DIGITS = 5
    MAX_RANDOM_DIGITS = 10
    DEFAULT_LOG_FILE = "gcp_manager.log"
    DEFAULT_CONFIG_FILE = "gcp_manager_config.json"
    DEFAULT_OUTPUT_FILE = "gcp_resources.json"


@dataclass
class ResourceInfo:
    """Structured resource information for cleanup operations"""
    type: str
    id: str
    region: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreationResult:
    """Structured information about created resources"""
    timestamp: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    project_number: Optional[str] = None
    vpc_name: Optional[str] = None
    subnet_name: Optional[str] = None
    cluster_name: Optional[str] = None
    region: Optional[str] = None
    node_count: Optional[int] = None
    binary_authorization_enabled: bool = False
    operation_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_number": self.project_number,
            "vpc_name": self.vpc_name,
            "subnet_name": self.subnet_name,
            "cluster_name": self.cluster_name,
            "region": self.region,
            "node_count": self.node_count,
            "binary_authorization_enabled": self.binary_authorization_enabled,
            "operation_type": self.operation_type
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class Logger:
    """Enhanced logger with colored output and file logging capabilities"""
    
    def __init__(self, log_to_file: bool = False, log_file: str = None):
        """Initialize logger with optional file logging
        
        Args:
            log_to_file: Whether to log to a file
            log_file: Path to log file (defaults to gcp_manager.log in current directory)
        """
        self.log_to_file = log_to_file
        self.log_file = log_file or Constants.DEFAULT_LOG_FILE
        
        if self.log_to_file:
            # Configure file logging
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def info(self, message: str) -> None:
        """Log info message
        
        Args:
            message: Message to log
        """
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")
        if self.log_to_file:
            logging.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message
        
        Args:
            message: Message to log
        """
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
        if self.log_to_file:
            logging.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message
        
        Args:
            message: Message to log
        """
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
        if self.log_to_file:
            logging.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message
        
        Args:
            message: Message to log
        """
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} {message}")
        if self.log_to_file:
            logging.debug(message)


class ResourceTracker:
    """Manages tracking and JSON output of created resources"""
    
    def __init__(self, output_file: str = None):
        """Initialize resource tracker
        
        Args:
            output_file: Path to JSON output file
        """
        self.output_file = output_file or Constants.DEFAULT_OUTPUT_FILE
        self.creation_results: List[CreationResult] = []
    
    def add_result(self, result: CreationResult) -> None:
        """Add a creation result to tracking
        
        Args:
            result: CreationResult to add
        """
        self.creation_results.append(result)
    
    def save_to_file(self, append: bool = True) -> None:
        """Save results to JSON file
        
        Args:
            append: If True, append to existing file; if False, overwrite
        """
        try:
            existing_data = []
            if append and os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
            
            # Add new results
            all_data = existing_data + [result.to_dict() for result in self.creation_results]
            
            with open(self.output_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            print(f"\n📊 Resource information saved to: {self.output_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save resource information to file: {e}")
    
    def print_summary(self) -> None:
        """Print a summary of all creation results"""
        if not self.creation_results:
            return
        
        print(f"\n{Colors.GREEN}📋 RESOURCE CREATION SUMMARY{Colors.NC}")
        print("=" * 50)
        
        for result in self.creation_results:
            print(f"\n🕒 Timestamp: {result.timestamp}")
            print(f"🔧 Operation: {result.operation_type}")
            
            if result.org_id:
                print(f"🏢 Organization ID: {result.org_id}")
            if result.project_id:
                print(f"📁 Project ID: {result.project_id}")
            if result.project_name:
                print(f"📝 Project Name: {result.project_name}")
            if result.project_number:
                print(f"🔢 Project Number: {result.project_number}")
            if result.vpc_name:
                print(f"🌐 VPC Name: {result.vpc_name}")
            if result.subnet_name:
                print(f"🔗 Subnet Name: {result.subnet_name}")
            if result.cluster_name:
                print(f"⚙️  GKE Cluster: {result.cluster_name}")
            if result.region:
                print(f"🌍 Region: {result.region}")
            if result.node_count:
                print(f"🖥️  Node Count: {result.node_count}")
            if result.binary_authorization_enabled:
                print(f"🔒 Binary Authorization: ✅ ENABLED")
        
        print("\n" + "=" * 50)


class BaseGCloudManager:
    """Base class with shared functionality for all GCP managers"""
    
    # Default configuration shared across managers
    DEFAULT_CONFIG = {
        "project": {
            "prefix": "ctd-testing",
            "random_digits_min": 5,
            "random_digits_max": 10
        },
        "gke": {
            "vpc_name": "gke-vpc",
            "subnet_name": "gke-subnet",
            "pod_range_name": "pod-range",
            "service_range_name": "service-range",
            "primary_range": "10.0.0.0/16",
            "pod_range": "10.1.0.0/16",
            "service_range": "10.2.0.0/16",
            "master_range": "10.3.0.0/28",
            "machine_type": "e2-standard-2",
            "disk_size": "50GB",
            "min_nodes": 1,
            "max_nodes": 4,
            "cluster_prefix": "standard-cluster",
            # Security settings
            "--shielded-secure-boot": True,
            "--shielded-integrity-monitoring": True,
            "enable_private_nodes": True,
            "enable_master_authorized_networks": True,
            "enable_binary_authorization": True
        },
        "binary_authorization": {
            "default_policy_mode": "ALWAYS_ALLOW",  # Start permissive, can be changed to ALWAYS_DENY
            "enforcement_mode": "ENFORCED_BLOCK_AND_AUDIT_LOG",
            "create_default_attestor": True,
            "attestor_name": "build-attestor"
        },
        "regions": [
            'us-central1', 'us-east1', 'us-east4', 'us-west1', 'us-west2', 'us-west3', 'us-west4',
            'europe-west1', 'europe-west2', 'europe-west3', 'europe-west4', 'europe-west6',
            'europe-north1', 'europe-central2', 'asia-east1', 'asia-east2', 'asia-northeast1',
            'asia-northeast2', 'asia-northeast3', 'asia-south1', 'asia-southeast1', 'asia-southeast2',
            'australia-southeast1', 'southamerica-east1'
        ],
        "timeouts": {
            "api_enable_wait": 15,
            "cluster_create_timeout": 1800,  # 30 minutes
            "resource_delete_timeout": 600   # 10 minutes
        },
        "output": {
            "json_file": "gcp_resources.json",
            "append_mode": True
        }
    }
    
    def __init__(self, config_file: str = None, logger: Logger = None, tracker: ResourceTracker = None):
        """Initialize with optional config file, logger, and tracker
        
        Args:
            config_file: Path to config file (JSON or INI)
            logger: Logger instance
            tracker: ResourceTracker instance
        """
        self.logger = logger or Logger()
        self.tracker = tracker
        self.config = self._load_config(config_file)
        self._validate_config()
    
    def _load_config(self, config_file: str = None) -> Dict:
        """Load configuration from file or use defaults
        
        Args:
            config_file: Path to config file (JSON or INI)
            
        Returns:
            Configuration dictionary
        """
        config = self.DEFAULT_CONFIG.copy()
        
        if config_file:
            path = Path(config_file)
            if not path.exists():
                self.logger.warning(f"Config file {config_file} not found, using defaults")
                return config
                
            try:
                if path.suffix.lower() == '.json':
                    with open(path, 'r') as f:
                        user_config = json.load(f)
                        self._update_nested_dict(config, user_config)
                elif path.suffix.lower() in ['.ini', '.cfg']:
                    parser = configparser.ConfigParser()
                    parser.read(path)
                    for section in parser.sections():
                        if section not in config:
                            config[section] = {}
                        for key, value in parser.items(section):
                            # Try to convert string values to appropriate types
                            try:
                                if value.isdigit():
                                    config[section][key] = int(value)
                                elif value.lower() in ['true', 'false']:
                                    config[section][key] = value.lower() == 'true'
                                else:
                                    config[section][key] = value
                            except:
                                config[section][key] = value
                else:
                    self.logger.warning(f"Unsupported config file format: {path.suffix}")
            except Exception as e:
                self.logger.error(f"Error loading config file: {e}")
                
        return config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary with another dictionary
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _validate_config(self) -> None:
        """Validate configuration values against schema
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['project', 'gke', 'binary_authorization', 'regions', 'timeouts']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate project config
        project_config = self.config['project']
        if not isinstance(project_config.get('random_digits_min'), int) or project_config['random_digits_min'] < 1:
            raise ValueError("project.random_digits_min must be a positive integer")
        if not isinstance(project_config.get('random_digits_max'), int) or project_config['random_digits_max'] < project_config['random_digits_min']:
            raise ValueError("project.random_digits_max must be >= random_digits_min")
        
        # Validate regions
        if not isinstance(self.config['regions'], list) or not self.config['regions']:
            raise ValueError("regions must be a non-empty list")
        
        # Validate timeouts
        timeout_config = self.config['timeouts']
        for timeout_key in ['api_enable_wait', 'cluster_create_timeout', 'resource_delete_timeout']:
            if not isinstance(timeout_config.get(timeout_key), int) or timeout_config[timeout_key] < 1:
                raise ValueError(f"timeouts.{timeout_key} must be a positive integer")
    
    def _generate_random_digits(self) -> str:
        """Generate random digits for resource naming
        
        Returns:
            Random digits string
        """
        min_length = self.config["project"]["random_digits_min"]
        max_length = self.config["project"]["random_digits_max"]
        length = random.randint(min_length, max_length)
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])
    
    def _validate_region(self, region: str) -> bool:
        """Validate GCP region format
        
        Args:
            region: Region to validate
            
        Returns:
            True if valid, False otherwise
        """
        return region in self.config["regions"]
    
    def _mask_sensitive_data(self, data: str, show_chars: int = 3) -> str:
        """Safely mask sensitive data for logging
        
        Args:
            data: Sensitive data to mask
            show_chars: Number of characters to show at the beginning
            
        Returns:
            Masked string
        """
        if not data or len(data) <= show_chars:
            return "***"
        return data[:show_chars] + "*" * (len(data) - show_chars)


class GCloudCommandRunner(BaseGCloudManager):
    """Base class for running gcloud commands"""
    
    def run_command(self, cmd: List[str], check: bool = True, ignore_errors: bool = False, dry_run: bool = False) -> subprocess.CompletedProcess:
        """Execute a shell command and return the result
        
        Args:
            cmd: Command to execute as list of strings
            check: Whether to check return code
            ignore_errors: If True, don't log errors or raise exceptions on failure
            dry_run: If True, only log the command without executing
            
        Returns:
            CompletedProcess instance with command result
            
        Raises:
            subprocess.CalledProcessError: If command fails and check is True and ignore_errors is False
            FileNotFoundError: If command executable not found
        """
        try:
            self.logger.debug(f"Executing: {' '.join(cmd)}")
            
            if dry_run:
                self.logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
                # Return a mock successful result for dry runs
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                self.logger.error(f"Command failed: {' '.join(cmd)}")
                self.logger.error(f"Error: {e.stderr}")
            raise
        except FileNotFoundError:
            if not ignore_errors:
                self.logger.error("gcloud CLI not found. Please install Google Cloud SDK.")
            raise
    
    def check_gcloud_auth(self) -> bool:
        """Check if gcloud is authenticated
        
        Returns:
            True if authenticated, False otherwise
        """
        try:
            result = self.run_command(["gcloud", "auth", "list", "--format=value(account)"], ignore_errors=True)
            if result.stdout.strip():
                return True
            return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


class GCloudProjectManager(GCloudCommandRunner):
    """Manages Google Cloud project operations"""
    
    def _validate_org_id(self, org_id: str) -> bool:
        """Validate organization ID format
        
        Args:
            org_id: Organization ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        return re.match(r'^\d+$', org_id) is not None
    
    def _validate_billing_id(self, billing_id: str) -> bool:
        """Validate billing account ID format
        
        Args:
            billing_id: Billing account ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        return re.match(r'^[A-Z0-9]{6}-[A-Z0-9]{6}-[A-Z0-9]{6}$', billing_id) is not None
    
    def _validate_project_id(self, project_id: str) -> bool:
        """Validate project ID format
        
        Args:
            project_id: Project ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        return re.match(r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$', project_id) is not None

    def check_project_exists(self, project_id: str) -> bool:
        """Check if a project exists
        
        Args:
            project_id: Project ID to check
            
        Returns:
            True if project exists, False otherwise
        """
        try:
            self.run_command(["gcloud", "projects", "describe", project_id], ignore_errors=True)
            return True
        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            raise RuntimeError("gcloud CLI not found.")

    def _enable_project_apis(self, project_id: str, dry_run: bool = False) -> None:
        """Enable all required APIs for the project including Binary Authorization
        
        Args:
            project_id: Project ID
            dry_run: If True, only simulate the operation
            
        Raises:
            RuntimeError: If API enablement fails
        """
        required_apis = [
            "container.googleapis.com",
            "compute.googleapis.com",
            "binaryauthorization.googleapis.com",
            "containeranalysis.googleapis.com",
            "cloudkms.googleapis.com",  # For Binary Authorization attestation keys
            "cloudbuild.googleapis.com"  # For automated builds and attestations
        ]
        
        self.logger.info("Enabling required APIs for Binary Authorization and GKE...")
        
        if dry_run:
            self.logger.info("[DRY RUN] API enablement simulation completed")
            return
        
        try:
            self.run_command([
                "gcloud", "services", "enable", 
                *required_apis,
                "--project", project_id
            ])
            
            # Wait for APIs to be enabled
            api_wait_time = self.config["timeouts"]["api_enable_wait"]
            self.logger.info(f"Waiting {api_wait_time} seconds for APIs to be enabled...")
            time.sleep(api_wait_time)
        except Exception as e:
            self.logger.error(f"Failed to enable APIs: {e}")
            raise RuntimeError(f"API enablement failed: {str(e)}")

    def _setup_binary_authorization_policy(self, project_id: str, dry_run: bool = False) -> None:
        """Set up initial Binary Authorization policy
        
        Args:
            project_id: Project ID
            dry_run: If True, only simulate the operation
            
        Raises:
            RuntimeError: If policy setup fails
        """
        self.logger.info("Setting up Binary Authorization policy...")
        
        if dry_run:
            self.logger.info("[DRY RUN] Binary Authorization policy setup simulation completed")
            return
        
        try:
            # Create a basic policy that allows all images initially
            policy_mode = self.config["binary_authorization"]["default_policy_mode"]
            enforcement_mode = self.config["binary_authorization"]["enforcement_mode"]
            
            policy_yaml = f"""globalPolicyEvaluationMode: ENABLE
defaultAdmissionRule:
  evaluationMode: {policy_mode}
  enforcementMode: {enforcement_mode}
name: projects/{project_id}/policy
"""
            
            # Create temporary policy file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(policy_yaml)
                policy_file = f.name
            
            try:
                # Import the policy
                self.run_command([
                    "gcloud", "container", "binauthz", "policy", "import", policy_file,
                    "--project", project_id
                ])
                self.logger.info("Binary Authorization policy configured successfully")
                
                # Create default attestor if configured
                if self.config["binary_authorization"]["create_default_attestor"]:
                    self._create_default_attestor(project_id)
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(policy_file)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Failed to setup Binary Authorization policy: {e}")
            raise RuntimeError(f"Binary Authorization policy setup failed: {str(e)}")

    def _create_default_attestor(self, project_id: str) -> None:
        """Create a default attestor for Binary Authorization
        
        Args:
            project_id: Project ID
            
        Raises:
            RuntimeError: If attestor creation fails
        """
        attestor_name = self.config["binary_authorization"]["attestor_name"]
        note_id = f"{attestor_name}-note"
        
        self.logger.info(f"Creating default attestor: {attestor_name}")
        
        try:
            # Create a note for the attestor
            import tempfile
            note_json = {
                "attestation": {
                    "hint": {
                        "human_readable_name": f"Default attestor note for {project_id}"
                    }
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(note_json, f)
                note_file = f.name
            
            try:
                # Create the note using REST API
                self.run_command([
                    "curl", "-X", "POST",
                    "-H", "Content-Type: application/json",
                    "-H", f"Authorization: Bearer $(gcloud auth print-access-token)",
                    "--data-binary", f"@{note_file}",
                    f"https://containeranalysis.googleapis.com/v1/projects/{project_id}/notes?noteId={note_id}"
                ])
                
                # Create the attestor
                self.run_command([
                    "gcloud", "container", "binauthz", "attestors", "create", attestor_name,
                    f"--attestation-authority-note={note_id}",
                    f"--attestation-authority-note-project={project_id}",
                    "--project", project_id
                ])
                
                # Grant Binary Authorization service account access to the note
                project_number = self._get_project_number(project_id)
                service_account = f"service-{project_number}@gcp-sa-binaryauthorization.iam.gserviceaccount.com"
                
                self.run_command([
                    "curl", "-X", "POST",
                    "-H", "Content-Type: application/json",
                    "-H", f"Authorization: Bearer $(gcloud auth print-access-token)",
                    f"https://containeranalysis.googleapis.com/v1/projects/{project_id}/notes/{note_id}:setIamPolicy",
                    "--data", json.dumps({
                        "policy": {
                            "bindings": [{
                                "role": "roles/containeranalysis.notes.occurrences.viewer",
                                "members": [f"serviceAccount:{service_account}"]
                            }]
                        }
                    })
                ])
                
                self.logger.info(f"Default attestor '{attestor_name}' created successfully")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(note_file)
                except:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Failed to create default attestor (this is optional): {e}")

    def _get_project_number(self, project_id: str) -> str:
        """Get project number from project ID
        
        Args:
            project_id: Project ID
            
        Returns:
            Project number as string
        """
        result = self.run_command([
            "gcloud", "projects", "describe", project_id,
            "--format=value(projectNumber)"
        ])
        return result.stdout.strip()

    def _get_project_name(self, project_id: str) -> str:
        """Get project name from project ID
        
        Args:
            project_id: Project ID
            
        Returns:
            Project name as string
        """
        try:
            result = self.run_command([
                "gcloud", "projects", "describe", project_id,
                "--format=value(name)"
            ])
            return result.stdout.strip()
        except Exception:
            return ""

    def create_project(self, org_id: str, billing_id: str, dry_run: bool = False) -> CreationResult:
        """Create a new Google Cloud project with Binary Authorization enabled
        
        Args:
            org_id: Organization ID
            billing_id: Billing account ID
            dry_run: If True, only simulate the operation
            
        Returns:
            CreationResult with project information
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If project creation fails
        """
        # Validate inputs
        if not self._validate_org_id(org_id):
            raise ValueError("Invalid organization ID format. Should be numeric.")
        
        if not self._validate_billing_id(billing_id):
            self.logger.warning("Billing ID format may be invalid. Expected format: XXXXXX-XXXXXX-XXXXXX")
        
        # Check authentication
        if not dry_run and not self.check_gcloud_auth():
            raise RuntimeError("Not authenticated with gcloud. Run 'gcloud auth login' first.")
        
        # Generate project details
        random_digits = self._generate_random_digits()
        prefix = self.config["project"]["prefix"]
        project_id = f"{prefix}-{random_digits}"
        project_name = f"{prefix.upper()} - {random_digits}"
        
        # Create result object
        result = CreationResult(
            timestamp=datetime.now().isoformat(),
            org_id=org_id,
            project_id=project_id,
            project_name=project_name,
            binary_authorization_enabled=True,
            operation_type="create_project"
        )
        
        self.logger.info(f"Creating project: {project_id}")
        self.logger.info(f"Organization ID: {org_id}")
        # Securely mask billing ID - only show first 3 characters
        masked_billing = self._mask_sensitive_data(billing_id, 3)
        self.logger.info(f"Billing Account: {masked_billing}")
        
        if dry_run:
            self.logger.info("[DRY RUN] Project creation simulation completed successfully")
            return result
        
        created_resources = []
        try:
            # Create the project
            self.logger.info("Creating project...")
            self.run_command([
                "gcloud", "projects", "create", project_id,
                f"--organization={org_id}",
                f"--name={project_name}"
            ])
            created_resources.append(ResourceInfo(type="project", id=project_id))
            
            # Get project number
            result.project_number = self._get_project_number(project_id)
            
            # Link billing account
            self.logger.info("Linking billing account...")
            self.run_command([
                "gcloud", "billing", "projects", "link", project_id,
                f"--billing-account={billing_id}"
            ])
            
            # Set as active project
            self.logger.info("Setting as active project...")
            self.run_command([
                "gcloud", "config", "set", "project", project_id
            ])
            
            # Enable all required APIs including Binary Authorization
            self._enable_project_apis(project_id)
            
            # Set up Binary Authorization policy
            self._setup_binary_authorization_policy(project_id)
            
            self.logger.info("Project created successfully with Binary Authorization enabled!")
            self.logger.info(f"Project ID: {project_id}")
            self.logger.info(f"Project Name: {project_name}")
            
            # Track the result
            if self.tracker:
                self.tracker.add_result(result)
            
            # Display project details
            print("\nProject details:")
            self.run_command([
                "gcloud", "projects", "describe", project_id
            ])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            # Cleanup any created resources
            self._cleanup_resources(created_resources)
            raise RuntimeError(f"Project creation failed: {str(e)}")
    
    def _cleanup_resources(self, resources: List[ResourceInfo]) -> None:
        """Clean up resources after a failed operation
        
        Args:
            resources: List of ResourceInfo objects to clean up
        """
        self.logger.warning("Cleaning up resources after failed operation...")
        
        for resource in reversed(resources):
            try:
                if resource.type == "project":
                    self.logger.info(f"Deleting project {resource.id}...")
                    self.run_command([
                        "gcloud", "projects", "delete", resource.id, "--quiet"
                    ])
                # Add other resource types as needed
            except Exception as e:
                self.logger.error(f"Failed to clean up {resource.type} {resource.id}: {e}")
    
    def delete_project(self, project_id: str, force: bool = False, dry_run: bool = False) -> None:
        """Delete a Google Cloud project
        
        Args:
            project_id: Project ID to delete
            force: Skip confirmation prompt if True
            dry_run: If True, only simulate the operation
            
        Raises:
            ValueError: If project ID is invalid
            RuntimeError: If project deletion fails
        """
        # Validate project ID
        if not self._validate_project_id(project_id):
            self.logger.warning("Project ID format may be invalid.")
        
        # Check authentication
        if not dry_run and not self.check_gcloud_auth():
            raise RuntimeError("Not authenticated with gcloud. Run 'gcloud auth login' first.")
        
        # Check if project exists
        if not dry_run and not self.check_project_exists(project_id):
             raise RuntimeError(f"Project '{project_id}' not found or access denied.")
        
        self.logger.warning(f"This will permanently delete project: {project_id}")
        self.logger.warning("This action cannot be undone!")
        
        if dry_run:
            self.logger.info("[DRY RUN] Project deletion simulation completed")
            return
        
        # Confirmation prompt unless force is True
        if not force:
            confirmation = input("Are you sure you want to delete this project? (yes/no): ")
            
            if confirmation.lower() != 'yes':
                self.logger.info("Project deletion cancelled")
                return
        
        try:
            self.logger.info(f"Deleting project: {project_id}")
            self.run_command([
                "gcloud", "projects", "delete", project_id, "--quiet"
            ])
            
            self.logger.info("Project deleted successfully!")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to delete project: {e}")
            raise RuntimeError(f"Project deletion failed: {str(e)}")


class GCloudGKEManager(GCloudCommandRunner):
    """Manages Google Kubernetes Engine (GKE) operations with Binary Authorization"""
    
    def check_vpc_exists(self, project_id: str, vpc_name: str) -> bool:
        """Check if a VPC exists in a project
        
        Args:
            project_id: Project ID
            vpc_name: VPC name
            
        Returns:
            True if VPC exists, False otherwise
        """
        try:
            self.run_command(["gcloud", "compute", "networks", "describe", vpc_name, "--project", project_id], ignore_errors=True)
            return True
        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            raise RuntimeError("gcloud CLI not found.")

    def check_subnet_exists(self, project_id: str, subnet_name: str, region: str) -> bool:
        """Check if a subnet exists in a project and region
        
        Args:
            project_id: Project ID
            subnet_name: Subnet name
            region: GCP region
            
        Returns:
            True if subnet exists, False otherwise
        """
        try:
            self.run_command(["gcloud", "compute", "networks", "subnets", "describe", subnet_name, "--project", project_id, "--region", region], ignore_errors=True)
            return True
        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            raise RuntimeError("gcloud CLI not found.")

    def check_gke_cluster_exists(self, project_id: str, cluster_name: str, region: str) -> bool:
        """Check if a GKE cluster exists
        
        Args:
            project_id: Project ID
            cluster_name: Cluster name
            region: GCP region
            
        Returns:
            True if cluster exists, False otherwise
        """
        try:
            self.run_command(["gcloud", "container", "clusters", "describe", cluster_name, "--project", project_id, "--region", region], ignore_errors=True)
            return True
        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            raise RuntimeError("gcloud CLI not found.")

    def create_vpc_and_subnets(self, project_id: str, region: str = "us-central1", dry_run: bool = False) -> Tuple[str, str, str, str]:
        """Create VPC and subnets required for GKE cluster
        
        Args:
            project_id: Project ID
            region: GCP region
            dry_run: If True, only simulate the operation
            
        Returns:
            Tuple of (vpc_name, subnet_name, pod_range_name, service_range_name)
            
        Raises:
            ValueError: If region is invalid
            RuntimeError: If VPC/subnet creation fails
        """
        # Validate region
        if not self._validate_region(region):
            raise ValueError(f"Invalid region: {region}. Please use a valid GCP region.")
        
        # Get network configuration from config
        vpc_name = self.config["gke"]["vpc_name"]
        subnet_name = self.config["gke"]["subnet_name"]
        pod_range_name = self.config["gke"]["pod_range_name"]
        service_range_name = self.config["gke"]["service_range_name"]
        primary_range = self.config["gke"]["primary_range"]
        pod_range = self.config["gke"]["pod_range"]
        service_range = self.config["gke"]["service_range"]
        
        self.logger.info(f"Creating VPC and subnets for GKE in region: {region}")
        
        if dry_run:
            self.logger.info("[DRY RUN] VPC and subnets creation simulation completed")
            return vpc_name, subnet_name, pod_range_name, service_range_name
        
        created_resources = []
        try:
            # Create VPC
            self.logger.info(f"Creating VPC: {vpc_name}")
            self.run_command([
                "gcloud", "compute", "networks", "create", vpc_name,
                "--subnet-mode", "custom",
                "--project", project_id
            ])
            created_resources.append(ResourceInfo(type="vpc", id=vpc_name))
            
            # Create subnet with secondary ranges for pods and services
            self.logger.info(f"Creating subnet: {subnet_name}")
            self.run_command([
                "gcloud", "compute", "networks", "subnets", "create", subnet_name,
                "--network", vpc_name,
                "--range", primary_range,
                "--secondary-range", f"{pod_range_name}={pod_range}",
                "--secondary-range", f"{service_range_name}={service_range}",
                "--region", region,
                "--project", project_id
            ])
            created_resources.append(ResourceInfo(type="subnet", id=subnet_name, region=region))
            
            self.logger.info("VPC and subnets created successfully!")
            return vpc_name, subnet_name, pod_range_name, service_range_name
            
        except Exception as e:
            self.logger.error(f"Failed to create VPC/subnets: {e}")
            # Cleanup any created resources
            self._cleanup_resources(project_id, created_resources)
            raise RuntimeError(f"VPC/subnet creation failed: {str(e)}")
    
    def _cleanup_resources(self, project_id: str, resources: List[ResourceInfo]) -> None:
        """Clean up resources after a failed operation
        
        Args:
            project_id: Project ID
            resources: List of ResourceInfo objects to clean up
        """
        self.logger.warning("Cleaning up resources after failed operation...")
        
        for resource in reversed(resources):
            try:
                if resource.type == "vpc":
                    self.logger.info(f"Deleting VPC {resource.id}...")
                    self.run_command([
                        "gcloud", "compute", "networks", "delete", resource.id,
                        "--project", project_id, "--quiet"
                    ])
                elif resource.type == "subnet":
                    self.logger.info(f"Deleting subnet {resource.id}...")
                    self.run_command([
                        "gcloud", "compute", "networks", "subnets", "delete", resource.id,
                        "--project", project_id, "--region", resource.region, "--quiet"
                    ])
                elif resource.type == "cluster":
                    if resource.region:
                        self.logger.info(f"Deleting GKE cluster {resource.id}...")
                        self.run_command([
                            "gcloud", "container", "clusters", "delete", resource.id,
                            "--region", resource.region, "--project", project_id, "--quiet"
                        ])
            except Exception as e:
                self.logger.error(f"Failed to clean up {resource.type} {resource.id}: {e}")
    
    def create_gke_cluster(self, project_id: str, vpc_name: str, subnet_name: str, 
                          cluster_name: Optional[str] = None, 
                          region: str = "us-central1", node_count: int = 1, 
                          dry_run: bool = False) -> CreationResult:
        """Create a secure GKE cluster with Binary Authorization enabled
        
        Args:
            project_id: Project ID
            vpc_name: VPC name
            subnet_name: Subnet name
            cluster_name: Cluster name (auto-generated if None)
            region: GCP region
            node_count: Number of nodes
            dry_run: If True, only simulate the operation
            
        Returns:
            CreationResult with cluster information
            
        Raises:
            ValueError: If region is invalid
            RuntimeError: If cluster creation fails
        """
        # Validate region
        if not self._validate_region(region):
            raise ValueError(f"Invalid region: {region}. Please use a valid GCP region.")
        
        # Generate cluster name if not provided
        if not cluster_name:
            random_digits = self._generate_random_digits()
            prefix = self.config["gke"]["cluster_prefix"]
            cluster_name = f"{prefix}-{random_digits}"
        
        # Create result object
        result = CreationResult(
            timestamp=datetime.now().isoformat(),
            project_id=project_id,
            vpc_name=vpc_name,
            subnet_name=subnet_name,
            cluster_name=cluster_name,
            region=region,
            node_count=node_count,
            binary_authorization_enabled=True,
            operation_type="create_gke_cluster"
        )
        
        self.logger.info(f"Creating secure GKE cluster with Binary Authorization: {cluster_name}")
        self.logger.info(f"Region: {region}")
        self.logger.info(f"Node count: {node_count}")
        self.logger.info("Security features: Secure Boot, Shielded VMs, Binary Authorization")
        
        if dry_run:
            self.logger.info("[DRY RUN] GKE cluster creation simulation completed")
            return result
        
        created_resources = []
        try:
            # Create secure GKE cluster with Binary Authorization
            self._create_cluster(
                project_id=project_id,
                cluster_name=cluster_name,
                region=region,
                node_count=node_count,
                vpc_name=vpc_name,
                subnet_name=subnet_name
            )
            created_resources.append(ResourceInfo(type="cluster", id=cluster_name, region=region))
            
            # Get cluster credentials
            self._get_cluster_credentials(project_id, cluster_name, region)
            
            # Verify Binary Authorization is enabled
            self._verify_binary_authorization(project_id, cluster_name, region)
            
            # Track the result
            if self.tracker:
                self.tracker.add_result(result)
            
            self.logger.info("Secure GKE cluster created successfully!")
            self.logger.info(f"Cluster name: {cluster_name}")
            self.logger.info(f"VPC: {vpc_name}")
            self.logger.info(f"Subnet: {subnet_name}")
            self.logger.info("✅ Binary Authorization: ENABLED")
            self.logger.info("✅ Secure Boot: ENABLED")
            self.logger.info("✅ Shielded VMs: ENABLED")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create GKE cluster: {e}")
            # Cleanup any created resources
            self._cleanup_resources(project_id, created_resources)
            raise RuntimeError(f"GKE cluster creation failed: {str(e)}")
    
    def _create_cluster(self, project_id: str, cluster_name: str, region: str, 
                       node_count: int, vpc_name: str, subnet_name: str) -> None:
        """Create GKE cluster with all security features enabled
        
        Args:
            project_id: Project ID
            cluster_name: Cluster name
            region: GCP region
            node_count: Number of nodes
            vpc_name: VPC name
            subnet_name: Subnet name
            
        Raises:
            RuntimeError: If cluster creation fails
        """
        # Get configuration
        gke_config = self.config["gke"]
        
        self.logger.info("Creating secure GKE cluster (this may take 5-15 minutes)...")
        try:
            cmd = [
                "gcloud", "container", "clusters", "create", cluster_name,
                "--region", region,
                "--num-nodes", str(node_count),
                "--machine-type", gke_config["machine_type"],
                "--disk-size", gke_config["disk_size"],
                "--network", vpc_name,
                "--subnetwork", subnet_name,
                "--cluster-secondary-range-name", gke_config["pod_range_name"],
                "--services-secondary-range-name", gke_config["service_range_name"],
                "--enable-ip-alias",
                "--enable-shielded-nodes",
                "--enable-autoupgrade",
                "--enable-autorepair",
                "--enable-autoscaling",
                "--min-nodes", str(gke_config["min_nodes"]),
                "--max-nodes", str(gke_config["max_nodes"]),
                "--project", project_id,
                "--quiet"
            ]
            
            # Add security features based on configuration
            if gke_config.get("enable_shielded_secure_boot", True):
                cmd.append("--shielded-secure-boot")
                
            if gke_config.get("enable_shielded_integrity_monitoring", True):
                cmd.append("--shielded-integrity-monitoring")
                
            if gke_config.get("enable_private_nodes", True):
                cmd.extend([
                    "--enable-private-nodes",
                    "--master-ipv4-cidr", gke_config["master_range"]
                ])
                
            if gke_config.get("enable_master_authorized_networks", True):
                cmd.append("--enable-master-authorized-networks")
                
            # Enable Binary Authorization
            if gke_config.get("enable_binary_authorization", True):
                cmd.append("--binauthz-evaluation-mode=PROJECT_SINGLETON_POLICY_ENFORCE")
            
            self.run_command(cmd)
            
        except Exception as e:
            self.logger.error(f"Failed to create GKE cluster: {e}")
            raise RuntimeError(f"GKE cluster creation failed: {str(e)}")
    
    def _get_cluster_credentials(self, project_id: str, cluster_name: str, region: str) -> None:
        """Get credentials for GKE cluster
        
        Args:
            project_id: Project ID
            cluster_name: Cluster name
            region: GCP region
            
        Raises:
            RuntimeError: If getting credentials fails
        """
        self.logger.info("Getting cluster credentials...")
        try:
            self.run_command([
                "gcloud", "container", "clusters", "get-credentials", cluster_name,
                "--region", region,
                "--project", project_id
            ])
        except Exception as e:
            self.logger.error(f"Failed to get cluster credentials: {e}")
            raise RuntimeError(f"Getting cluster credentials failed: {str(e)}")
    
    def _verify_binary_authorization(self, project_id: str, cluster_name: str, region: str) -> None:
        """Verify Binary Authorization is enabled on the cluster
        
        Args:
            project_id: Project ID
            cluster_name: Cluster name
            region: GCP region
        """
        try:
            self.logger.info("Verifying Binary Authorization configuration...")
            result = self.run_command([
                "gcloud", "container", "clusters", "describe", cluster_name,
                "--region", region, "--project", project_id,
                "--format=json(binaryAuthorization)"
            ])
            
            import json
            cluster_info = json.loads(result.stdout)
            binary_auth = cluster_info.get("binaryAuthorization", {})
            evaluation_mode = binary_auth.get("evaluationMode", "DISABLED")
            
            if evaluation_mode == "PROJECT_SINGLETON_POLICY_ENFORCE":
                self.logger.info("✅ Binary Authorization is properly enabled")
            else:
                self.logger.warning(f"⚠️ Binary Authorization evaluation mode: {evaluation_mode}")
                
        except Exception as e:
            self.logger.warning(f"Could not verify Binary Authorization status: {e}")
    
    def delete_gke_cluster(self, project_id: str, cluster_name: str, region: str = "us-central1", 
                          force: bool = False, dry_run: bool = False) -> None:
        """Delete a GKE cluster
        
        Args:
            project_id: Project ID
            cluster_name: Cluster name
            region: GCP region
            force: Skip confirmation prompt if True
            dry_run: If True, only simulate the operation
            
        Raises:
            ValueError: If region is invalid
            RuntimeError: If cluster deletion fails
        """
        # Validate region
        if not self._validate_region(region):
            raise ValueError(f"Invalid region: {region}. Please use a valid GCP region.")
        
        # Check if cluster exists
        if not dry_run and not self.check_gke_cluster_exists(project_id, cluster_name, region):
            raise RuntimeError(f"GKE cluster '{cluster_name}' not found in project '{project_id}' region '{region}'.")

        self.logger.warning(f"This will delete GKE cluster: {cluster_name}")
        
        if dry_run:
            self.logger.info("[DRY RUN] GKE cluster deletion simulation completed")
            return
        
        # Confirmation prompt unless force is True
        if not force:
            confirmation = input(f"Are you sure you want to delete GKE cluster '{cluster_name}'? (yes/no): ")
            if confirmation.lower() != 'yes':
                self.logger.info("GKE cluster deletion cancelled")
                return

        try:
            self.logger.info(f"Deleting GKE cluster: {cluster_name}")
            self.run_command([
                "gcloud", "container", "clusters", "delete", cluster_name,
                "--region", region,
                "--project", project_id,
                "--quiet"
            ])
            
            self.logger.info("GKE cluster deleted successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to delete GKE cluster: {e}")
            raise RuntimeError(f"GKE cluster deletion failed: {str(e)}")

    def delete_vpc(self, project_id: str, vpc_name: str, force: bool = False, dry_run: bool = False) -> None:
        """Delete a VPC network
        
        Args:
            project_id: Project ID
            vpc_name: VPC name
            force: Skip confirmation prompt if True
            dry_run: If True, only simulate the operation
            
        Raises:
            RuntimeError: If VPC deletion fails
        """
        # Check if VPC exists
        if not dry_run and not self.check_vpc_exists(project_id, vpc_name):
            raise RuntimeError(f"VPC '{vpc_name}' not found in project '{project_id}'.")

        self.logger.warning(f"This will delete VPC network: {vpc_name}")
        
        if dry_run:
            self.logger.info("[DRY RUN] VPC deletion simulation completed")
            return
        
        # Confirmation prompt unless force is True
        if not force:
            confirmation = input(f"Are you sure you want to delete VPC '{vpc_name}'? (yes/no): ")
            if confirmation.lower() != 'yes':
                self.logger.info("VPC deletion cancelled")
                return

        try:
            self.logger.info(f"Deleting VPC: {vpc_name}")
            self.run_command([
                "gcloud", "compute", "networks", "delete", vpc_name,
                "--project", project_id,
                "--quiet"
            ])
            
            self.logger.info("VPC deleted successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to delete VPC: {e}")
            raise RuntimeError(f"VPC deletion failed: {str(e)}")

    def delete_subnet(self, project_id: str, subnet_name: str, region: str, 
                     force: bool = False, dry_run: bool = False) -> None:
        """Delete a subnet
        
        Args:
            project_id: Project ID
            subnet_name: Subnet name
            region: GCP region
            force: Skip confirmation prompt if True
            dry_run: If True, only simulate the operation
            
        Raises:
            ValueError: If region is invalid
            RuntimeError: If subnet deletion fails
        """
        # Validate region
        if not self._validate_region(region):
            raise ValueError(f"Invalid region: {region}. Please use a valid GCP region.")

        # Check if subnet exists
        if not dry_run and not self.check_subnet_exists(project_id, subnet_name, region):
            raise RuntimeError(f"Subnet '{subnet_name}' not found in project '{project_id}' region '{region}'.")

        self.logger.warning(f"This will delete subnet: {subnet_name}")
        
        if dry_run:
            self.logger.info("[DRY RUN] Subnet deletion simulation completed")
            return
        
        # Confirmation prompt unless force is True
        if not force:
            confirmation = input(f"Are you sure you want to delete subnet '{subnet_name}'? (yes/no): ")
            if confirmation.lower() != 'yes':
                self.logger.info("Subnet deletion cancelled")
                return

        try:
            self.logger.info(f"Deleting subnet: {subnet_name}")
            self.run_command([
                "gcloud", "compute", "networks", "subnets", "delete", subnet_name,
                "--project", project_id,
                "--region", region,
                "--quiet"
            ])
            
            self.logger.info("Subnet deleted successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to delete subnet: {e}")
            raise RuntimeError(f"Subnet deletion failed: {str(e)}")

    def update_binary_authorization_policy(self, project_id: str, policy_mode: str = "ALWAYS_DENY", 
                                         cluster_name: str = None, region: str = None, 
                                         dry_run: bool = False) -> None:
        """Update Binary Authorization policy for enhanced security
        
        Args:
            project_id: Project ID
            policy_mode: Policy mode (ALWAYS_ALLOW, ALWAYS_DENY, REQUIRE_ATTESTATION)
            cluster_name: Specific cluster name for cluster-specific rules
            region: Region for cluster-specific rules
            dry_run: If True, only simulate the operation
            
        Raises:
            RuntimeError: If policy update fails
        """
        self.logger.info(f"Updating Binary Authorization policy to {policy_mode}...")
        
        if dry_run:
            self.logger.info("[DRY RUN] Binary Authorization policy update simulation completed")
            return
        
        try:
            enforcement_mode = self.config["binary_authorization"]["enforcement_mode"]
            
            if cluster_name and region:
                # Create cluster-specific policy
                policy_yaml = f"""globalPolicyEvaluationMode: ENABLE
defaultAdmissionRule:
  evaluationMode: ALWAYS_DENY
  enforcementMode: {enforcement_mode}
clusterAdmissionRules:
  {region}.{cluster_name}:
    evaluationMode: {policy_mode}
    enforcementMode: {enforcement_mode}
name: projects/{project_id}/policy
"""
            else:
                # Create project-wide policy
                policy_yaml = f"""globalPolicyEvaluationMode: ENABLE
defaultAdmissionRule:
  evaluationMode: {policy_mode}
  enforcementMode: {enforcement_mode}
name: projects/{project_id}/policy
"""
            
            # Create temporary policy file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(policy_yaml)
                policy_file = f.name
            
            try:
                # Import the updated policy
                self.run_command([
                    "gcloud", "container", "binauthz", "policy", "import", policy_file,
                    "--project", project_id
                ])
                
                if cluster_name:
                    self.logger.info(f"Binary Authorization policy updated for cluster {cluster_name}")
                else:
                    self.logger.info("Binary Authorization policy updated project-wide")
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(policy_file)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Failed to update Binary Authorization policy: {e}")
            raise RuntimeError(f"Binary Authorization policy update failed: {str(e)}")


def create_default_config(config_path: str = None) -> None:
    """Create default configuration file with Binary Authorization settings
    
    Args:
        config_path: Path to save configuration file (defaults to DEFAULT_CONFIG_FILE)
    """
    if config_path is None:
        config_path = Constants.DEFAULT_CONFIG_FILE
    
    # Use default config from base class
    default_config = BaseGCloudManager.DEFAULT_CONFIG
    
    try:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default configuration saved to {config_path}")
        print("\nConfiguration includes:")
        print("✅ Binary Authorization support")
        print("✅ Secure Boot and Shielded VMs")
        print("✅ Private GKE clusters")
        print("✅ Enhanced security settings")
        print("✅ JSON output configuration")
    except Exception as e:
        print(f"Failed to create default configuration: {e}")


def main():
    """Main function to parse arguments and execute actions"""
    
    parser = argparse.ArgumentParser(
        description="Google Cloud Project Manager with Binary Authorization - Manage GCP Projects, VPCs, Subnets, and Secure GKE Clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new GCP project with Binary Authorization
  python %(prog)s -cp -o ORG_ID -b BILLING_ID

  # Create a full secure setup (Project, VPC, Subnet, Secure GKE with Binary Auth)
  python %(prog)s -cg -o ORG_ID -b BILLING_ID -r REGION

  # Create VPC and secure GKE in an existing project
  python %(prog)s -cvg -p PROJECT_ID -r REGION

  # Create secure GKE in an existing project and VPC/Subnet
  python %(prog)s -cgke -p PROJECT_ID --vpc VPC_NAME --subnet SUBNET_NAME -r REGION

  # Update Binary Authorization policy to be more restrictive
  python %(prog)s --update-binauthz -p PROJECT_ID --policy-mode ALWAYS_DENY

  # Update Binary Authorization policy for specific cluster
  python %(prog)s --update-binauthz -p PROJECT_ID -n CLUSTER_NAME -r REGION --policy-mode REQUIRE_ATTESTATION

  # Delete a GKE cluster
  python %(prog)s -dg -p PROJECT_ID -n CLUSTER_NAME -r REGION

  # Delete a VPC
  python %(prog)s -dv -p PROJECT_ID --vpc VPC_NAME

  # Delete a Subnet
  python %(prog)s -ds -p PROJECT_ID --subnet SUBNET_NAME -r REGION

  # Delete a Project (and all its resources)
  python %(prog)s -dp -p PROJECT_ID

  # Check if a project exists
  python %(prog)s --check-project -p PROJECT_ID

  # Check if a VPC exists
  python %(prog)s --check-vpc -p PROJECT_ID --vpc VPC_NAME

  # Check if a Subnet exists
  python %(prog)s --check-subnet -p PROJECT_ID --subnet SUBNET_NAME -r REGION

  # Check if a GKE cluster exists
  python %(prog)s --check-gke -p PROJECT_ID -n CLUSTER_NAME -r REGION

  # Create default configuration file with Binary Authorization settings
  python %(prog)s --init-config

  # Dry run mode (simulate operations without executing)
  python %(prog)s -cp -o ORG_ID -b BILLING_ID --dry-run

  # Specify custom output file for JSON results
  python %(prog)s -cg -o ORG_ID -b BILLING_ID -r REGION --output-file custom_resources.json
        """
    )
    
    # --- Action Group (Mutually Exclusive) ---
    action_group = parser.add_mutually_exclusive_group(required=True)
    
    # Create Actions
    action_group.add_argument(
        '-cp', '--create-project',
        action='store_true',
        help='Create a new GCP project with Binary Authorization enabled. Requires -o and -b.'
    )
    action_group.add_argument(
        '-cg', '--create-gke-full',
        action='store_true',
        help='Create a full secure setup: Project, VPC, Subnet, and Secure GKE Cluster with Binary Authorization. Requires -o, -b, -r.'
    )
    action_group.add_argument(
        '-cvg', '--create-vpc-gke',
        action='store_true',
        help='Create VPC and secure GKE cluster with Binary Authorization in an existing project. Requires -p, -r.'
    )
    action_group.add_argument(
        '-cgke', '--create-gke-only',
        action='store_true',
        help='Create secure GKE cluster with Binary Authorization in an existing project and VPC/Subnet. Requires -p, --vpc, --subnet, -r.'
    )
    
    # Delete Actions
    action_group.add_argument(
        '-dp', '--delete-project',
        action='store_true',
        help='Delete an existing project and all its resources. Requires -p.'
    )
    action_group.add_argument(
        '-dg', '--delete-gke',
        action='store_true',
        help='Delete an existing GKE cluster. Requires -p, -n, -r.'
    )
    action_group.add_argument(
        '-dv', '--delete-vpc',
        action='store_true',
        help='Delete an existing VPC network. Requires -p, --vpc.'
    )
    action_group.add_argument(
        '-ds', '--delete-subnet',
        action='store_true',
        help='Delete an existing subnet. Requires -p, --subnet, -r.'
    )

    # Binary Authorization Actions
    action_group.add_argument(
        '--update-binauthz',
        action='store_true',
        help='Update Binary Authorization policy. Requires -p. Optional: -n, -r, --policy-mode.'
    )

    # Check Actions
    action_group.add_argument(
        '--check-project',
        action='store_true',
        help='Check if a GCP project exists. Requires -p.'
    )
    action_group.add_argument(
        '--check-vpc',
        action='store_true',
        help='Check if a VPC network exists. Requires -p, --vpc.'
    )
    action_group.add_argument(
        '--check-subnet',
        action='store_true',
        help='Check if a subnet exists. Requires -p, --subnet, -r.'
    )
    action_group.add_argument(
        '--check-gke',
        action='store_true',
        help='Check if a GKE cluster exists. Requires -p, -n, -r.'
    )

    # Utility Action
    action_group.add_argument(
        '--init-config',
        action='store_true',
        help='Create default configuration file with Binary Authorization settings (gcp_manager_config.json).'
    )
    
    # --- Required Arguments for Actions ---
    parser.add_argument(
        '-p', '--project-id',
        type=str,
        help='Project ID (required for most operations).'
    )
    parser.add_argument(
        '-o', '--org-id',
        type=str,
        help='Organization ID (required for creating new projects).'
    )
    parser.add_argument(
        '-b', '--billing-id',
        type=str,
        help='Billing account ID (required for creating new projects).'
    )
    parser.add_argument(
        '-r', '--region',
        type=str,
        help='GCP region (required for GKE, VPC, Subnet operations).'
    )
    parser.add_argument(
        '-n', '--cluster-name',
        type=str,
        help='GKE cluster name (required for GKE operations, auto-generated if creating).'
    )
    parser.add_argument(
        '--vpc',
        type=str,
        help='VPC network name (required for some GKE/VPC operations).'
    )
    parser.add_argument(
        '--subnet',
        type=str,
        help='Subnet name (required for some GKE/Subnet operations).'
    )
    parser.add_argument(
        '--policy-mode',
        type=str,
        choices=['ALWAYS_ALLOW', 'ALWAYS_DENY', 'REQUIRE_ATTESTATION'],
        default='ALWAYS_DENY',
        help='Binary Authorization policy mode (default: ALWAYS_DENY).'
    )

    # --- Optional Arguments ---
    parser.add_argument(
        '-k', '--node-count',
        type=int,
        default=1,
        help='Number of nodes in GKE cluster (default: 1).'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (currently logs DEBUG level).'
    )
    parser.add_argument(
        '--log-file',
        action='store_true',
        help='Enable logging to file (gcp_manager.log).'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON or INI).'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to JSON output file for resource information (default: gcp_resources.json).'
    )
    parser.add_argument(
        '--no-json-output',
        action='store_true',
        help='Disable JSON output to file.'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Skip confirmation prompts for delete operations.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate operations without executing them (for testing).'
    )
    
    args = parser.parse_args()
    
    # --- Argument Validation Logic ---
    if args.init_config:
        create_default_config()
        return

    # Initialize logger
    logger = Logger(log_to_file=args.log_file)
    if args.verbose:
        # In a real scenario, you might set the logging level here
        logger.info("Verbose mode enabled (DEBUG logging)")
    
    # Initialize resource tracker
    tracker = None
    if not args.no_json_output:
        output_file = args.output_file or Constants.DEFAULT_OUTPUT_FILE
        tracker = ResourceTracker(output_file)
    
    if args.dry_run:
        logger.info("DRY RUN MODE: No actual resources will be created or modified")

    # Check gcloud authentication early (skip for dry run)
    if not args.dry_run:
        runner = GCloudCommandRunner(logger=logger)
        if not runner.check_gcloud_auth():
            logger.error("Not authenticated with gcloud. Run 'gcloud auth login' first.")
            sys.exit(1)

    # Validate required arguments based on action
    try:
        if args.create_project or args.create_gke_full:
            if not args.org_id or not args.billing_id:
                parser.error("-cp/--create-project and -cg/--create-gke-full require -o/--org-id and -b/--billing-id.")
        if args.create_gke_full or args.create_vpc_gke or args.create_gke_only or args.delete_gke or args.delete_subnet or args.check_subnet or args.check_gke or args.update_binauthz:
             if not args.region and (args.create_gke_full or args.create_vpc_gke or args.create_gke_only or args.delete_gke or args.delete_subnet or args.check_subnet or args.check_gke):
                 parser.error("Operations involving GKE, VPC, or Subnets require -r/--region.")
        if args.create_vpc_gke or args.create_gke_only or args.delete_project or args.delete_gke or args.delete_vpc or args.delete_subnet or args.check_project or args.check_vpc or args.check_subnet or args.check_gke or args.update_binauthz:
            if not args.project_id:
                parser.error("Most operations require an existing -p/--project-id.")
        if args.create_gke_only or args.delete_vpc or args.check_vpc:
            if not args.vpc:
                parser.error("This operation requires --vpc.")
        if args.create_gke_only or args.delete_subnet or args.check_subnet:
             if not args.subnet:
                 parser.error("This operation requires --subnet.")
        if args.delete_gke or args.check_gke:
            if not args.cluster_name:
                parser.error("This operation requires -n/--cluster-name.")
        if args.update_binauthz and args.policy_mode == 'REQUIRE_ATTESTATION':
            if not args.cluster_name or not args.region:
                parser.error("REQUIRE_ATTESTATION policy mode requires -n/--cluster-name and -r/--region for cluster-specific rules.")

    except argparse.ArgumentError as e:
        logger.error(f"Argument error: {e}")
        sys.exit(1)

    # Initialize managers
    try:
        project_manager = GCloudProjectManager(config_file=args.config, logger=logger, tracker=tracker)
        gke_manager = GCloudGKEManager(config_file=args.config, logger=logger, tracker=tracker)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    try:
        # --- Execute Actions ---
        if args.create_project:
            result = project_manager.create_project(args.org_id, args.billing_id, dry_run=args.dry_run)
            print(f"\n✅ Project '{result.project_id}' created successfully with Binary Authorization enabled!")

        elif args.create_gke_full:
            # 1. Create Project with Binary Authorization
            project_result = project_manager.create_project(args.org_id, args.billing_id, dry_run=args.dry_run)
            print(f"\n✅ Project '{project_result.project_id}' created successfully with Binary Authorization!")
            
            # 2. Create VPC and Subnets
            vpc_name, subnet_name, _, _ = gke_manager.create_vpc_and_subnets(project_result.project_id, args.region, dry_run=args.dry_run)
            print(f"✅ VPC '{vpc_name}' and Subnet '{subnet_name}' created successfully!")
            
            # 3. Create Secure GKE Cluster with Binary Authorization
            cluster_result = gke_manager.create_gke_cluster(
                project_result.project_id, vpc_name, subnet_name, args.cluster_name, args.region, args.node_count, dry_run=args.dry_run
            )
            print(f"✅ Secure GKE cluster '{cluster_result.cluster_name}' created successfully with Binary Authorization!")
            
            # Update project result with VPC and cluster info
            project_result.vpc_name = vpc_name
            project_result.subnet_name = subnet_name
            project_result.cluster_name = cluster_result.cluster_name
            project_result.region = args.region
            project_result.node_count = args.node_count
            project_result.operation_type = "create_gke_full"
            
            # Update tracker with consolidated result
            if tracker:
                tracker.creation_results = [project_result]  # Replace separate results with consolidated one

        elif args.create_vpc_gke:
            # 1. Check Project Exists
            if not args.dry_run and not project_manager.check_project_exists(args.project_id):
                 raise RuntimeError(f"Project '{args.project_id}' not found. Please create it first or verify the ID.")
            
            # 2. Create VPC and Subnets
            vpc_name, subnet_name, _, _ = gke_manager.create_vpc_and_subnets(args.project_id, args.region, dry_run=args.dry_run)
            print(f"✅ VPC '{vpc_name}' and Subnet '{subnet_name}' created successfully!")
            
            # 3. Create Secure GKE Cluster with Binary Authorization
            cluster_result = gke_manager.create_gke_cluster(
                args.project_id, vpc_name, subnet_name, args.cluster_name, args.region, args.node_count, dry_run=args.dry_run
            )
            print(f"✅ Secure GKE cluster '{cluster_result.cluster_name}' created successfully with Binary Authorization!")
            
            # Update result with VPC info
            cluster_result.vpc_name = vpc_name
            cluster_result.subnet_name = subnet_name
            cluster_result.operation_type = "create_vpc_gke"

        elif args.create_gke_only:
            # 1. Check Project Exists
            if not args.dry_run and not project_manager.check_project_exists(args.project_id):
                 raise RuntimeError(f"Project '{args.project_id}' not found. Please create it first or verify the ID.")
            # 2. Check VPC Exists
            if not args.dry_run and not gke_manager.check_vpc_exists(args.project_id, args.vpc):
                 raise RuntimeError(f"VPC '{args.vpc}' not found in project '{args.project_id}'. Please create it first or verify the name.")
            # 3. Check Subnet Exists
            if not args.dry_run and not gke_manager.check_subnet_exists(args.project_id, args.subnet, args.region):
                 raise RuntimeError(f"Subnet '{args.subnet}' not found in project '{args.project_id}' region '{args.region}'. Please create it first or verify the name.")
            # 4. Create Secure GKE Cluster with Binary Authorization
            cluster_result = gke_manager.create_gke_cluster(
                args.project_id, args.vpc, args.subnet, args.cluster_name, args.region, args.node_count, dry_run=args.dry_run
            )
            print(f"✅ Secure GKE cluster '{cluster_result.cluster_name}' created successfully with Binary Authorization!")

        elif args.update_binauthz:
            gke_manager.update_binary_authorization_policy(
                args.project_id, args.policy_mode, args.cluster_name, args.region, dry_run=args.dry_run
            )
            if args.cluster_name:
                print(f"\n✅ Binary Authorization policy updated for cluster '{args.cluster_name}' to {args.policy_mode}!")
            else:
                print(f"\n✅ Binary Authorization policy updated project-wide to {args.policy_mode}!")

        elif args.delete_project:
            project_manager.delete_project(args.project_id, force=args.force, dry_run=args.dry_run)
            print(f"\n✅ Project '{args.project_id}' deletion initiated successfully!")

        elif args.delete_gke:
            gke_manager.delete_gke_cluster(args.project_id, args.cluster_name, args.region, force=args.force, dry_run=args.dry_run)
            print(f"\n✅ GKE cluster '{args.cluster_name}' deletion initiated successfully!")

        elif args.delete_vpc:
            gke_manager.delete_vpc(args.project_id, args.vpc, force=args.force, dry_run=args.dry_run)
            print(f"\n✅ VPC '{args.vpc}' deletion initiated successfully!")

        elif args.delete_subnet:
            gke_manager.delete_subnet(args.project_id, args.subnet, args.region, force=args.force, dry_run=args.dry_run)
            print(f"\n✅ Subnet '{args.subnet}' deletion initiated successfully!")

        elif args.check_project:
            exists = project_manager.check_project_exists(args.project_id)
            status = "exists" if exists else "does not exist"
            print(f"Project '{args.project_id}' {status}.")

        elif args.check_vpc:
            exists = gke_manager.check_vpc_exists(args.project_id, args.vpc)
            status = "exists" if exists else "does not exist"
            print(f"VPC '{args.vpc}' in project '{args.project_id}' {status}.")

        elif args.check_subnet:
            exists = gke_manager.check_subnet_exists(args.project_id, args.subnet, args.region)
            status = "exists" if exists else "does not exist"
            print(f"Subnet '{args.subnet}' in project '{args.project_id}' region '{args.region}' {status}.")

        elif args.check_gke:
            exists = gke_manager.check_gke_cluster_exists(args.project_id, args.cluster_name, args.region)
            status = "exists" if exists else "does not exist"
            print(f"GKE cluster '{args.cluster_name}' in project '{args.project_id}' region '{args.region}' {status}.")

        # Save JSON output and print summary for creation operations
        if tracker and tracker.creation_results and not args.dry_run:
            tracker.print_summary()
            tracker.save_to_file(append=True)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[INFO]{Colors.NC} Operation cancelled by user")
        sys.exit(1)
    except (ValueError, RuntimeError, argparse.ArgumentError) as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

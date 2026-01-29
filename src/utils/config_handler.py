"""
Configuration Handler
Manages system configuration from files and environment
"""

import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigHandler:
    """Manages configuration from YAML/JSON files"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config handler
        
        Args:
            config_file: Path to config file (YAML or JSON)
        """
        self.config: Dict[str, Any] = {}
        
        if config_file and Path(config_file).exists():
            self.load(config_file)
    
    def load(self, config_file: str) -> bool:
        """
        Load configuration from file
        
        Args:
            config_file: Path to config file
            
        Returns:
            Success status
        """
        try:
            path = Path(config_file)
            
            if path.suffix == '.yaml' or path.suffix == '.yml':
                with open(path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.error(f"Unsupported config format: {path.suffix}")
                return False
            
            logger.info(f"Config loaded from {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def save(self, config_file: str, fmt: str = 'yaml') -> bool:
        """
        Save configuration to file
        
        Args:
            config_file: Path to config file
            fmt: Format ('yaml' or 'json')
            
        Returns:
            Success status
        """
        try:
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            if fmt == 'yaml':
                with open(config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif fmt == 'json':
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.error(f"Unsupported format: {fmt}")
                return False
            
            logger.info(f"Config saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-notation key
        
        Args:
            key: Key path (e.g., 'inference.model_path')
            default: Default value if key not found
            
        Returns:
            Config value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set config value by dot-notation key
        
        Args:
            key: Key path
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __repr__(self) -> str:
        return json.dumps(self.config, indent=2)

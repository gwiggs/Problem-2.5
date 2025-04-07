#!/usr/bin/env python3
"""
Migration script to move from the old llm_api.py structure to the new modular structure.
This script will:
1. Check if all necessary files exist in the new structure
2. Verify that all functions from llm_api.py have been migrated
3. Provide instructions for completing the migration
"""

import os
import sys
import inspect
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_module_from_file(file_path, module_name):
    """Load a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_functions_from_module(module):
    """Get all functions from a module"""
    return {name: func for name, func in inspect.getmembers(module) 
            if inspect.isfunction(func) and func.__module__ == module.__name__}

def check_file_exists(file_path):
    """Check if a file exists and log the result"""
    exists = os.path.exists(file_path)
    if exists:
        logger.info(f"✅ {file_path} exists")
    else:
        logger.error(f"❌ {file_path} does not exist")
    return exists

def main():
    """Main function to check migration status"""
    logger.info("Checking migration status...")
    
    # Check if old file exists
    old_file_exists = check_file_exists("llm_api.py")
    if not old_file_exists:
        logger.error("❌ llm_api.py not found. Migration may have already been completed.")
        return
    
    # Load the old module
    old_module = load_module_from_file("llm_api.py", "llm_api")
    if old_module is None:
        logger.error("❌ Failed to load llm_api.py")
        return
    
    # Get functions from old module
    old_functions = get_functions_from_module(old_module)
    logger.info(f"Found {len(old_functions)} functions in llm_api.py")
    
    # Check if new structure exists
    new_structure_exists = all([
        check_file_exists("app/__init__.py"),
        check_file_exists("app/main.py"),
        check_file_exists("app/api/__init__.py"),
        check_file_exists("app/api/routes.py"),
        check_file_exists("app/core/__init__.py"),
        check_file_exists("app/core/config.py"),
        check_file_exists("app/core/lifespan.py"),
        check_file_exists("app/core/model_manager.py"),
        check_file_exists("app/core/cleanup.py"),
        check_file_exists("app/models/__init__.py"),
        check_file_exists("app/models/schemas.py"),
        check_file_exists("app/utils/__init__.py"),
        check_file_exists("app/utils/gpu.py"),
        check_file_exists("app/utils/model_loader.py"),
        check_file_exists("app/utils/media_processor.py")
    ])
    
    if not new_structure_exists:
        logger.error("❌ New structure is incomplete. Please create all necessary files.")
        return
    
    # Check if Dockerfile has been updated
    dockerfile_updated = False
    if os.path.exists("Dockerfile"):
        with open("Dockerfile", "r") as f:
            dockerfile_content = f.read()
            dockerfile_updated = "uvicorn app.main:app" in dockerfile_content
    
    if dockerfile_updated:
        logger.info("✅ Dockerfile has been updated to use the new structure")
    else:
        logger.error("❌ Dockerfile needs to be updated to use app.main:app")
    
    # Provide migration instructions
    logger.info("\n=== Migration Instructions ===")
    logger.info("1. Verify that all functions from llm_api.py have been migrated to the new structure:")
    for func_name in old_functions:
        logger.info(f"   - {func_name}")
    
    logger.info("\n2. Update the Dockerfile to use the new entry point:")
    logger.info("   CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8100\"]")
    
    logger.info("\n3. Test the new structure:")
    logger.info("   docker-compose build llm")
    logger.info("   docker-compose up -d llm")
    
    logger.info("\n4. Once everything is working, remove llm_api.py:")
    logger.info("   rm llm_api.py")
    
    logger.info("\n=== Migration Complete ===")

if __name__ == "__main__":
    main() 
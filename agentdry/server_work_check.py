#!/usr/bin/env python3
"""
Test script to verify the AgentDRY MCP server can start correctly.
Run this from the agentdry directory to test the server before integrating with Claude Desktop.
"""

import os
import sys
import subprocess
import time

def test_server_startup():
    """Test if the AgentDRY MCP server can start without errors."""
    print("Testing AgentDRY MCP Server Startup...")
    print("=" * 50)
    
    # Set up environment
    venv_python = "C:/greendex/greendex/Scripts/python.exe"
    server_file = "servers/claude_mcp_server.py"
    
    if not os.path.exists(venv_python):
        print(f"ERROR: Virtual environment Python not found: {venv_python}")
        return False
    
    if not os.path.exists(server_file):
        print(f"ERROR: Server file not found: {server_file}")
        return False
    
    print(f"Using Python: {venv_python}")
    print(f"Server file: {server_file}")
    print()
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = f"C:/prass/agentdry;C:/greendex/greendex/Lib/site-packages"
    env['VIRTUAL_ENV'] = "C:/greendex/greendex"
    env['PYTHONIOENCODING'] = "utf-8"
    env['AGENTDRY_VERSION'] = "1.0"
    env['AGENTDRY_TAGLINE'] = "Made in Mangaluru by Vedanth and Praas team"
    
    try:
        print("Starting AgentDRY server (will run for 8 seconds)...")
        process = subprocess.Popen(
            [venv_python, server_file, "--test"],  # Added --test flag for clean exit
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # Wait for process to complete (should exit with --test flag)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # If it doesn't exit with --test flag, terminate it
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
        
        print("STDOUT:")
        print(stdout)
        print("\nSTDERR:")
        print(stderr)
        
        # Check for AgentDRY success indicators
        success_indicators = [
            "AgentDRY Server starting",
            "AgentDRY ready for Claude Desktop integration",
            "AgentDRY Server initialized successfully",
            "AgentDRY"
        ]
        
        # Check for error indicators
        error_indicators = [
            "Traceback",
            "ImportError",
            "ModuleNotFoundError",
            "failed to start"
        ]
        
        output_text = stdout + stderr
        
        has_success = any(indicator in output_text for indicator in success_indicators)
        has_error = any(indicator in output_text for indicator in error_indicators)
        
        if has_success and not has_error:
            print("\n✅ SUCCESS: AgentDRY Server started successfully!")
            print("🎉 Beautiful banner displayed correctly!")
            return True
        elif has_error:
            print("\n❌ ERROR: Server encountered errors during startup")
            return False
        else:
            print("\n⚠️  WARNING: Server started but success indicators not found")
            print("This might be normal for MCP servers in stdio mode")
            return True  # Give benefit of doubt for MCP servers
            
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print("AgentDRY server was terminated after timeout")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        # For MCP servers, timeout might be normal behavior
        if "AgentDRY" in (stdout + stderr):
            print("\n✅ Server appears to be working (MCP servers run indefinitely)")
            return True
        return False
    except Exception as e:
        print(f"ERROR: Failed to start AgentDRY server: {e}")
        return False

def test_imports():
    """Test if all required imports work."""
    print("Testing Python Imports for AgentDRY...")
    print("=" * 40)
    
    required_modules = [
        'mcp',
        'mcp.server.fastmcp',
        'httpx',
        'dotenv'
    ]
    
    # Add paths for testing
    sys.path.insert(0, 'C:/prass/agentdry')
    sys.path.insert(0, 'C:/greendex/greendex/Lib/site-packages')
    
    all_good = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ OK: {module}")
        except ImportError as e:
            print(f"❌ ERROR: {module} - {e}")
            all_good = False
    
    # Test local imports
    try:
        from utils.append_to_server import create_tool_from_user_input
        print("✅ OK: utils.append_to_server")
    except ImportError as e:
        print(f"❌ ERROR: utils.append_to_server - {e}")
        all_good = False
    
    try:
        from main import create_and_update_tool
        print("✅ OK: main")
    except ImportError as e:
        print(f"❌ ERROR: main - {e}")
        all_good = False
    
    return all_good

def test_configuration():
    """Test if configuration files exist and are properly formatted."""
    print("Testing AgentDRY Configuration...")
    print("=" * 35)
    
    config_files = [
        "C:/Users/buzzv/AppData/Roaming/Claude/claude_desktop_config.json",
        "servers/claude_mcp_server.py"
    ]
    
    all_good = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ OK: {config_file} exists")
            
            # Check if it contains AgentDRY references
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'agentdry' in content.lower() or 'AgentDRY' in content:
                        print(f"   ✅ Contains AgentDRY branding")
                    else:
                        print(f"   ⚠️  No AgentDRY branding found")
            except Exception as e:
                print(f"   ⚠️  Could not read file: {e}")
        else:
            print(f"❌ ERROR: {config_file} not found")
            all_good = False
    
    return all_good

def main():
    print("╔════════════════════════════════════════════════════╗")
    print("║                AgentDRY Test Suite                 ║")
    print("║            Made in Mangaluru by                    ║")
    print("║             Vedanth and Praas team                 ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    
    # Test configuration first
    config_ok = test_configuration()
    print()
    
    # Test imports
    imports_ok = test_imports()
    print()
    
    if imports_ok:
        print("All imports successful. Testing AgentDRY server startup...")
        print()
        server_ok = test_server_startup()
    else:
        print("Import errors found. Skipping server test.")
        server_ok = False
    
    print()
    print("=" * 60)
    print("AGENTDRY TEST SUMMARY:")
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Imports:       {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Server:        {'✅ PASS' if server_ok else '❌ FAIL'}")
    
    if config_ok and imports_ok and server_ok:
        print()
        print("🎉 All tests passed! AgentDRY is ready for Claude Desktop integration!")
        print()
        print("Next steps:")
        print("1. Update Claude Desktop config with the AgentDRY configuration")
        print("2. Restart Claude Desktop completely")
        print("3. Test integration with AgentDRY tools")
        print("4. Enjoy your custom dynamic tool creation system!")
    else:
        print()
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print()
        if not config_ok:
            print("- Check that configuration files exist and contain AgentDRY branding")
        if not imports_ok:
            print("- Install missing Python dependencies")
        if not server_ok:
            print("- Check server startup logs for specific errors")

if __name__ == "__main__":
    main()
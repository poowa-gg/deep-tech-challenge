#!/usr/bin/env python3
"""
AgriMind Edge - Easy Web Demo Starter
Simple script to start the web demonstration
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_flask():
    """Check if Flask is installed"""
    try:
        import flask
        return True
    except ImportError:
        return False

def install_flask():
    """Install Flask if not present"""
    print("📦 Installing Flask...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("\n" + "="*60)
    print("🌱 AGRIMIND EDGE - WEB DEMO LAUNCHER")
    print("="*60)
    print("🏆 Africa Deep Tech Challenge 2025 Submission")
    print("🚀 Breakthrough Agricultural Swarm Intelligence System")
    print("="*60)
    
    # Check Flask installation
    if not check_flask():
        print("❌ Flask not found. Installing...")
        if not install_flask():
            print("❌ Failed to install Flask. Please install manually:")
            print("   pip install flask")
            return
        print("✅ Flask installed successfully!")
    
    print("🔧 Starting AgriMind Edge Web Demo...")
    print("📱 The demo will open in your browser automatically")
    print("🌐 URL: http://localhost:5000")
    print("\n💡 Demo Features:")
    print("   • Ultra-low power sensor monitoring")
    print("   • AI crop disease detection (<1MB models)")
    print("   • 🚀 BREAKTHROUGH: Swarm Intelligence")
    print("   • Quantum-inspired resource optimization")
    print("   • Real-time pest migration tracking")
    print("   • Power management (6+ months battery)")
    print("   • Performance metrics dashboard")
    print("\n" + "="*60)
    
    # Start the web demo
    try:
        # Wait a moment then open browser
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Import and run the web demo
        from web_demo import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing web demo: {e}")
        print("\n💡 Alternative: Run directly with:")
        print("   python web_demo.py")
    except Exception as e:
        print(f"❌ Error starting web demo: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you're in the project directory")
        print("   2. Check that all files are present")
        print("   3. Try running: python web_demo.py")

if __name__ == "__main__":
    main()
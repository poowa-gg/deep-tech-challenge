#!/usr/bin/env python3
"""
Quick launcher for web demo from agrimind_edge directory
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to parent directory
os.chdir(parent_dir)

print("🌱 Starting AgriMind Edge Web Demo...")
print("📱 Opening browser to: http://localhost:5000")
print("🚀 Don't miss the BREAKTHROUGH Swarm Intelligence demo!")

# Import and run the web demo
try:
    from web_demo import app
    app.run(debug=False, host='0.0.0.0', port=5000)
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Please run from main directory:")
    print("   cd ..")
    print("   python web_demo.py")
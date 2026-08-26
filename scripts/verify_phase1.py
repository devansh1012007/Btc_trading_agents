#!/usr/bin/env python
"""
PHASE 1 Complete - Quick Start Guide
Execute this to verify everything is working
"""

import subprocess


def verify_phase1():
    """Verify PHASE 1 is complete and working."""
    print("\n" + "="*80)
    print("PHASE 1: FOUNDATION & DATA PIPELINE - VERIFICATION")
    print("="*80 + "\n")

    steps = [
        ("1. Check dependencies installed", "pip list | grep -E 'pandas|numpy|ccxt|lightgbm'"),
        ("2. Run unit tests", "python -m pytest tests/test_phase1.py -v --tb=short"),
        ("3. Run integration test", "python test_integration.py"),
    ]

    for step_name, command in steps:
        print(f"\n{step_name}...")
        print(f"Command: {command}\n")
        result = subprocess.run(command, shell=True, capture_output=False)
        if result.returncode != 0:
            print(f"❌ {step_name} FAILED")
            return False
        print(f"✅ {step_name} OK")

    print("\n" + "="*80)
    print("✅ PHASE 1 VERIFICATION COMPLETE")
    print("="*80)
    print("\nWhat's Next?")
    print("1. Review: PHASE1_COMPLETE.md")
    print("2. Review: README_PHASE1.md")
    print("3. Read: src/state/contracts.py (state schema)")
    print("4. Read: src/layers/feature_engine.py (features)")
    print("\nReady to start PHASE 2:")
    print("- Dynamic Structure Engine (zones)")
    print("- Regime Detection (HMM + classifier)")
    print("\n")

if __name__ == "__main__":
    verify_phase1()

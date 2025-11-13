"""
Quick test script to verify the Flask app and Google Drive integration are set up correctly
"""

import sys
import importlib.util

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...\n")
    
    tests = [
        ("Flask", "flask"),
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("ReportLab", "reportlab"),
        ("Google Auth", "google.auth"),
        ("Google API Client", "googleapiclient"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module in tests:
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                print(f"✓ {name:20} - OK")
                passed += 1
            else:
                print(f"✗ {name:20} - Not found")
                failed += 1
        except (ImportError, ModuleNotFoundError):
            print(f"✗ {name:20} - Not installed")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

def test_file_structure():
    """Test that required files exist"""
    import os
    print("\n\nChecking file structure...\n")
    
    required_files = [
        "flask_app.py",
        "requirements.txt",
        "README.md",
        "GOOGLE_DRIVE_SETUP.md",
        "utils/google_drive.py",
        "templates/report.html",
        "templates/index.html",
        ".gitignore",
    ]
    
    passed = 0
    failed = 0
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path:30} - Found")
            passed += 1
        else:
            print(f"✗ {file_path:30} - Missing")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

def test_flask_syntax():
    """Test Flask app syntax without running it"""
    print("\n\nTesting Flask app syntax...\n")
    
    try:
        import py_compile
        py_compile.compile('flask_app.py', doraise=True)
        print("✓ flask_app.py - Syntax OK")
        
        py_compile.compile('utils/google_drive.py', doraise=True)
        print("✓ utils/google_drive.py - Syntax OK")
        
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

def check_gitignore():
    """Verify sensitive files are in .gitignore"""
    print("\n\nChecking .gitignore security...\n")
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    required_entries = [
        'credentials.json',
        'token.pickle',
    ]
    
    all_present = True
    for entry in required_entries:
        if entry in content:
            print(f"✓ {entry:25} - Protected")
        else:
            print(f"✗ {entry:25} - NOT in .gitignore (SECURITY RISK!)")
            all_present = False
    
    return all_present

def main():
    """Run all tests"""
    print("="*60)
    print("ECG-DeepNet System Verification")
    print("="*60)
    
    results = []
    
    results.append(("File Structure", test_file_structure()))
    results.append(("Python Syntax", test_flask_syntax()))
    results.append(("Security (.gitignore)", check_gitignore()))
    results.append(("Module Imports", test_imports()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Follow GOOGLE_DRIVE_SETUP.md for Google Drive integration")
        print("  3. Run the app: python flask_app.py")
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

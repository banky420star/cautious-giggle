"""Double-click this file to start Cautious Giggle (no console window)."""
import runpy, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = [os.path.join(os.path.dirname(__file__), "launcher.py")]
runpy.run_path("launcher.py", run_name="__main__")

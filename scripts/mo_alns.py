"""MO-ALNS entrypoint for the current explicit-budget comparison protocol."""
import sys
from scripts.comparison_suite import main
if __name__=='__main__':
    if '--method' not in sys.argv: sys.argv.extend(['--method','mo_alns'])
    main()

"""Select one schema-13 ablation arm, then use the shared budgeted entrypoint."""
import argparse,json,sys
from pathlib import Path
from scripts.comparison_suite import main
if __name__=='__main__':
    arms=json.loads(Path('configs/factorial27_ablations.json').read_text(encoding='utf-8'))['arms']
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--arm',choices=tuple(arms),default='main')
    args,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest,'--config',arms[args.arm]]
    main()

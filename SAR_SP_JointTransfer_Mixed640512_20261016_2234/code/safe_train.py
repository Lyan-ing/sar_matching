"""Guard the frozen trainer against accidental reuse of an existing run directory."""
import argparse,sys,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(add_help=False);p.add_argument('--run',required=True);p.add_argument('--resume');a,rest=p.parse_known_args();folder=root/'runs'/a.run
if folder.exists() and any(folder.iterdir()):
 if not a.resume:raise SystemExit('Refusing to overwrite existing run: '+str(folder))
 for path in folder.glob('status_*.json'):
  if json.loads(path.read_text()).get('state')=='completed':raise SystemExit('This run already completed; do not repeat or extend the frozen experiment.')
raise SystemExit(subprocess.call([sys.executable,'-B',str(root/'code/joint_train.py'),*sys.argv[1:]]))

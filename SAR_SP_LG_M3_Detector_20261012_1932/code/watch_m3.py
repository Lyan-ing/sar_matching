"""Independent lightweight CPU status watcher; never restarts jobs or evaluates benchmarks."""
import json,time,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];run=root/'runs/m3'
while True:
 try:
  r=json.loads((run/'status.json').read_text());p=Path(f'/proc/{r["pid"]}/stat');alive=p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']
  age=(datetime.now(ZoneInfo('Asia/Shanghai'))-datetime.fromisoformat(r['heartbeat'])).total_seconds()
  state=dict(updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),state=r['state'],process_alive=alive,global_step=r['global_step'],heartbeat_age_seconds=age,checkpoint_exists=(run/'last.ckpt').exists(),warning='no progress >15min' if age>900 and r['state']=='training' else '',never_restarts=True)
  tmp=run/('monitor_status.json.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,run/'monitor_status.json');print(json.dumps(state),flush=True)
  if r['state'] in ['completed','failed','stopped']:break
 except FileNotFoundError:pass
 time.sleep(60)

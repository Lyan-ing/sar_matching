from common import *
def main():
 for v in ['joint','control','frozen']:
  lock=json.loads(output('selection_locked',folder=ROOT/'runs'/v).read_text());assert lock['final_step']==5000 and sha(ROOT/'runs'/v/'best.ckpt')==lock['best_sha']
 assert not output('final').exists(),'Final run is one-time and frozen'
 from evaluate import main as evaluate
 sys.argv.append('--final');evaluate()
if __name__=='__main__':main()

#!/usr/bin/env python3
import argparse,json
from causal_se2_occ.checkpoint import migrate_legacy_checkpoint
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();ck=migrate_legacy_checkpoint(a.source,a.output);print(json.dumps({'output':a.output,'epoch':ck.get('epoch'),'global_step':ck.get('global_step'),'mapping':'identity strict'},indent=2))

#!/usr/bin/env python3
"""Recompute run-wise Figure 2 endpoints from the released complex table."""
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'reproducibility/source_data/figure2/figure2_canonical_complex_table.csv');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.input.open() as f:rows=list(csv.DictReader(f))
 groups=defaultdict(list)
 for row in rows:groups[(row['method'],row['benchmark_set'],row['run_id'])].append(row)
 runs=[];endpoints={'top1':'rank1_rmsd_A','top3':'best_top3_rmsd_A','top5':'best_top5_rmsd_A','oracle':'best_oracle_rmsd_A'}
 for (method,benchmark,run),items in sorted(groups.items()):
  if len({r['complex_id'] for r in items})!=len(items):raise ValueError('Duplicate complex in a run/benchmark')
  for endpoint,field in endpoints.items():
   values=[]
   for row in items:
    try:value=float(row[field])
    except (ValueError,TypeError):value=math.inf
    values.append(value if math.isfinite(value) else math.inf)
   successes=sum(v<2 for v in values)
   runs.append(dict(method=method,benchmark=benchmark,run=run,endpoint=endpoint,n_complexes=len(items),success_count=successes,success_percent=100*successes/len(items)))
 summary=[];pooled=defaultdict(list)
 for row in runs:pooled[(row['method'],row['benchmark'],row['endpoint'])].append(row)
 for (method,benchmark,endpoint),items in sorted(pooled.items()):
  values=[r['success_percent'] for r in items]
  summary.append(dict(method=method,benchmark=benchmark,endpoint=endpoint,repeats=len(values),mean_percent=statistics.mean(values),sample_sd_percent=statistics.stdev(values) if len(values)>1 else None))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump({'strict_threshold_A':2,'failed_slots_retained':True,'run_wise':runs,'summary':summary},f,indent=2)
 print(a.output)
if __name__=='__main__':main()

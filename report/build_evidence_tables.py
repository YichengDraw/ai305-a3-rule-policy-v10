#!/usr/bin/env python3
"""Generate appendix numbers from the released native-history summary."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=Path(__file__).resolve().parent
h=json.loads((ROOT/'results/history/formal_history_summary.json').read_text())
rounds=['13000','15000','16000','17000','19000','24000']
S=[]
for i,rnd in enumerate(rounds,5):
    v=f'v{i}'
    item=next(x for x in h['versions'] if x['round']==rnd and x['version']==v)
    if len(item['cells'])!=6:raise ValueError('missing cells')
    sub=[]
    for task in ('2x5','2x10'):
        a=[x for x in item['cells'] if x['task']==task]
        sub.append((sum(x['completion'] for x in a)/3,sum(x['successes'] for x in a)))
    S.append((i,rnd,item,sub))
rows=[]
for i,rnd,s,sub in S:
    c1,n1=sub[0];c2,n2=sub[1]
    rows.append(f'V{i} & {rnd} & {c1*100:.2f}\\% & {n1}/60 & {c2*100:.2f}\\% & {n2}/60 & {s["completion"]*100:.2f}\\% & {s["successes"]}/120 \\\\')
content=r'''\begin{table*}[t]
\centering\small
\caption{Frozen version milestones: results by task (all three settings combined). Each row covers 20 seeds times six settings. The seed set changes from one milestone to another; matched improvements are listed separately in Table~\ref{tab:paired}.}
\label{tab:milestonesfull}
\begin{tabular}{llrrrrrr}
\toprule
& & \multicolumn{2}{c}{$2\times5$} & \multicolumn{2}{c}{$2\times10$} & \multicolumn{2}{c}{Overall}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Version & Test seeds & Completion & Success & Completion & Success & Completion & Success\\
\midrule
'''+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table*}
'''
out=R/'sections/07_generated_history_table.tex';out.write_text(content)
assert len(rows)==6
print(out, 'items',len(rows))
print('\n'.join(rows))

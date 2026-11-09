#!/usr/bin/env python3
"""Draw explanatory figures from frozen milestone summaries (no simulator)."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R=Path(__file__).resolve().parent
F=R/'figures';F.mkdir(exist_ok=True)
FONT=next((p for p in ('/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') if Path(p).is_file()),'DejaVuSans.ttf')
BOLD=next((p for p in ('/System/Library/Fonts/Supplemental/Arial Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf') if Path(p).is_file()),'DejaVuSans-Bold.ttf')
def ft(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
BG=(250,251,254);NAVY=(26,45,74);BLUE=(40,107,176);GREEN=(14,132,111);BORDER=(196,208,219);MUTED=(82,95,112)
def center(d,xy,text,font,fill=NAVY):
 x,y=xy;b=d.textbbox((0,0),text,font=font);d.text((x-(b[2]-b[0])/2,y-(b[3]-b[1])/2),text,font=font,fill=fill)
def block(d,x0,y0,x1,y1,heading,body,accent=BLUE):
 d.rounded_rectangle((x0,y0,x1,y1),22,fill=(255,255,255),outline=BORDER,width=3)
 d.rounded_rectangle((x0,y0,x1,y0+12),7,fill=accent)
 center(d,((x0+x1)/2,y0+67),heading,ft(43,True),NAVY)
 for i,line in enumerate(body):center(d,((x0+x1)/2,y0+142+47*i),line,ft(31),MUTED)
def arrow(d,x0,y0,x1,y1,color=BLUE,width=10):
 d.line((x0,y0,x1,y1),fill=color,width=width)
 import math
 angle=math.atan2(y1-y0,x1-x0);length=26;spread=.45
 a=(x1-length*math.cos(angle-spread),y1-length*math.sin(angle-spread))
 b=(x1-length*math.cos(angle+spread),y1-length*math.sin(angle+spread))
 d.polygon(((x1,y1),a,b),fill=color)

im=Image.new('RGB',(2750,580),BG);d=ImageDraw.Draw(im)
blocks=[('AI advisor',['Study failures','Propose a test']),('Human relay',['Set priorities','Pass evidence']),('Coding executor',['Implement rule','Run comparison']),('Results',['Full episodes','Success and errors'])]
for i,(head,lines) in enumerate(blocks):
 x=65+i*675;block(d,x,55,x+555,350,head,lines,[BLUE,GREEN,BLUE,GREEN][i]);
 if i<3:arrow(d,x+558,208,x+655,208)
d.line((2420,383,2420,456,330,456,330,383),fill=GREEN,width=8)
arrow(d,330,456,330,389,color=GREEN,width=8)
center(d,(1375,515),'Repeat with evidence; keep changes that improve the full task',ft(34,True),GREEN)
im.save(F/'advisor_executor_loop.png',optimize=True)

im=Image.new('RGB',(2750,570),BG);d=ImageDraw.Draw(im)
blocks=[('Observe',['3 camera images','Robot state, touch']),('Estimate',['Box and cookie','positions']),('Decide',['Current step','and safe path']),('Act',['Move, grasp,','place or recover'])]
for i,(head,lines) in enumerate(blocks):
 x=65+i*675;block(d,x,55,x+555,345,head,lines,[BLUE,GREEN,BLUE,GREEN][i]);
 if i<3:arrow(d,x+558,206,x+655,206)
d.line((2400,374,2400,451,346,451,346,374),fill=BLUE,width=8)
arrow(d,346,451,346,386,color=BLUE,width=8)
center(d,(1380,515),'Check the outcome and revise the next motion',ft(34,True),BLUE)
im.save(F/'feedback_controller.png',optimize=True)

history=json.loads((R.parent/'results/history/formal_history_summary.json').read_text())
rounds=['13000','15000','16000','17000','19000','24000']
sequence=[next(x for x in history['versions'] if x['version']==f'v{i+5}' and x['round']==rounds[i]) for i in range(6)]
assert [round(x['completion']*100,1) for x in sequence]==[46.8,74.6,67.4,81.2,88.1,98.4]
W,H=2740,820
im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
left=150;right=W-100;top=120;bottom=H-175
for yv in (0,20,40,60,80,100):
 yy=bottom-(yv/100)*(bottom-top)
 d.line((left,yy,right,yy),fill=(224,230,237),width=2)
 d.text((55,yy-25),f'{yv}%',font=ft(32),fill=MUTED)
xys=[left+125 +i*(right-left-250)/5 for i in range(6)]
for i,x in enumerate(xys):center(d,(x,bottom+58),f'V{i+5}',ft(38,True))
for kind,color,label in [('completion',BLUE,'Completion'),('success_rate',GREEN,'Success')]:
 points=[]
 for i,x in enumerate(xys):
  v=sequence[i][kind]*100;y=bottom-v/100*(bottom-top);points.append((x,y))
 d.line(points,fill=color,width=10,joint='curve')
 for (x,y),s in zip(points,sequence):
  d.ellipse((x-12,y-12,x+12,y+12),fill=color,outline='white',width=3)
  off=-55 if kind=='completion' else +34
  if kind=='success_rate' and abs(y-points[0][1])<35:off=35
  center(d,(x,y+off),f'{s[kind]*100:.1f}',ft(30,True),color)
center(d,(W/2,43),'Frozen milestones: policy improvement across V5 to V10',ft(47,True))
d.rounded_rectangle((W-670,80,W-110,150),12,fill=(246,247,250),outline=BORDER,width=2)
d.line((W-630,114,W-566,114),fill=BLUE,width=9);d.text((W-550,91),'Completion',font=ft(31),fill=NAVY)
d.line((W-370,114,W-306,114),fill=GREEN,width=9);d.text((W-290,91),'Success',font=ft(31),fill=NAVY)
im.save(F/'milestones.png',optimize=True)
print('Created',[(p.name,p.stat().st_size) for p in sorted(F.glob('*.png'))])

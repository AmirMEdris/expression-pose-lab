"""Atomic, local progress shared by preparation, training, and the monitor."""
import json
import os
import time
from pathlib import Path


class Progress:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'progress.json'
        try:self.state=json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError,ValueError):self.state={'started_at':time.time(),'history':[]}
    def update(self,**values):
        self.state.update(values);self.state.update(updated_at=time.time(),pid=os.getpid())
        temp=self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.state,indent=2),encoding='utf-8')
        for _ in range(10):
            try:temp.replace(self.path);break
            except PermissionError:time.sleep(.05)
        else:raise PermissionError('Progress file is locked')
    def log(self,message):
        print(message,flush=True)
        with (self.folder/'events.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%H:%M:%S')+' '+message+'\n')


def monitor(folder):
    import tkinter as tk
    from tkinter import ttk
    folder=Path(folder).resolve();root=tk.Tk();root.title('Expression Lab · RGB training progress');root.geometry('1000x790');root.minsize(840,680)
    bg='#0b1120';panel='#111c30';fg='#e2e8f0';muted='#94a3b8';mint='#5eead4';violet='#a78bfa'
    root.configure(bg=bg);style=ttk.Style();style.theme_use('clam');style.configure('Horizontal.TProgressbar',troughcolor=panel,background=mint)
    title=tk.StringVar(value='Preparing custom RGB training');detail=tk.StringVar();numbers=tk.StringVar();timing=tk.StringVar();note=tk.StringVar()
    for variable,size,color in [(title,23,fg),(detail,11,muted),(numbers,14,mint),(timing,10,muted)]:
        tk.Label(root,textvariable=variable,bg=bg,fg=color,font=('Segoe UI',size),anchor='w',justify='left',wraplength=930).pack(fill='x',padx=28,pady=(14,0))
    bar=ttk.Progressbar(root,maximum=1);bar.pack(fill='x',padx=28,pady=18)
    charts=tk.Frame(root,bg=bg);charts.pack(fill='both',expand=True,padx=24)
    plots=[]
    for caption in ['Loss · lower is better','Validation accuracy / macro F1 · higher is better']:
        frame=tk.Frame(charts,bg=panel);frame.pack(side='left',fill='both',expand=True,padx=4)
        tk.Label(frame,text=caption,bg=panel,fg=fg,font=('Segoe UI',10)).pack(pady=12)
        canvas=tk.Canvas(frame,bg=panel,highlightthickness=0,height=220);canvas.pack(fill='both',expand=True);plots.append(canvas)
    tk.Label(root,text='Loss: mint = training, violet = validation. Scores: mint = accuracy, violet = macro F1.',bg=bg,fg=muted,font=('Segoe UI',9)).pack(pady=8)
    logs=tk.Text(root,height=7,bg=panel,fg=muted,font=('Consolas',9),relief='flat',wrap='word');logs.pack(fill='x',padx=28,pady=8)
    tk.Label(root,textvariable=note,bg=bg,fg=muted,font=('Segoe UI',9),anchor='w').pack(fill='x',padx=28)
    buttons=tk.Frame(root,bg=bg);buttons.pack(fill='x',padx=28,pady=16)
    ttk.Button(buttons,text='Open experiment folder',command=lambda:os.startfile(str(folder))).pack(side='left')
    def open_artifact(name):
        path=folder/name
        if path.exists():os.startfile(str(path))
        else:note.set('This appears after the first Dream preview (epoch 10), or after final evaluation for results.')
    ttk.Button(buttons,text='View evolving Dreams',command=lambda:open_artifact('dream-progress.html')).pack(side='left',padx=8)
    ttk.Button(buttons,text='Final results',command=lambda:open_artifact('report.html')).pack(side='left')
    tk.Label(buttons,text='Closing this window leaves training running.',bg=bg,fg=muted,font=('Segoe UI',9)).pack(side='right')
    def plot(canvas,history,keys,maximum=None):
        canvas.delete('all');w=max(300,canvas.winfo_width());h=max(150,canvas.winfo_height());left=42;bottom=h-28
        values=[float(row.get(k,0)) for row in history for k in keys]
        top=maximum or max(1.,max(values,default=1)*1.1)
        canvas.create_line(left,12,left,bottom,w-14,bottom,fill='#334155')
        canvas.create_text(left-5,14,text=f'{top:.2g}',fill=muted,anchor='e');canvas.create_text(left-5,bottom,text='0',fill=muted,anchor='e')
        for key,color in zip(keys,[mint,violet]):
            points=[]
            for i,row in enumerate(history):points.extend([left+i/max(1,len(history)-1)*(w-left-20),bottom-float(row.get(key,0))/top*(bottom-18)])
            if len(points)>=4:canvas.create_line(*points,fill=color,width=2)
            elif points:canvas.create_oval(points[0]-3,points[1]-3,points[0]+3,points[1]+3,fill=color,outline='')
        if history:canvas.create_text(w-16,h-12,text=f'Epoch {len(history)}',fill=muted,anchor='e')
        else:canvas.create_text(w/2,h/2,text='Curves appear after the first epoch',fill=muted)
    def refresh():
        try:
            state=json.loads((folder/'progress.json').read_text(encoding='utf-8'))
            phase=state.get('phase','Preparing');title.set(phase)
            detail.set('Evaluating validation faces with dropout disabled and no weight updates. Validation selects the checkpoint.' if phase=='Validating' else state.get('message',''))
            done=state.get('done',0);total=state.get('total',0);bar['value']=done/max(1,total)
            epoch=state.get('epoch',0);epochs=state.get('epochs',0)
            text=f'Epoch {epoch}/{epochs}  ·  ' if epochs else ''
            text+=state.get('progress_text',f'{done:,} / {total:,}' if total else 'Working…')
            if 'val_accuracy' in state:text+=f'  ·  Validation {state["val_accuracy"]:.1%}'
            numbers.set(text)
            elapsed=max(0,time.time()-state.get('started_at',time.time()));stale=time.time()-state.get('updated_at',0)
            timing.set(f'Elapsed {int(elapsed//3600)}h {int(elapsed%3600//60)}m  ·  '+state.get('device','GPU training starts after data preparation')+('  ·  GPU memory '+state['gpu_memory'] if state.get('gpu_memory') else ''))
            history=state.get('history',[]);plot(plots[0],history,['train_loss','val_loss']);plot(plots[1],history,['val_accuracy','val_macro_f1'],1.)
            note.set('No recent update; check the log or worker process.' if stale>120 and state.get('phase') not in ('Complete','Failed','Paused') else 'Validation selects checkpoints. Final test results appear only after training.')
            if (folder/'events.log').exists():
                text='\n'.join((folder/'events.log').read_text(encoding='utf-8').splitlines()[-10:]);logs.configure(state='normal');logs.delete('1.0','end');logs.insert('1.0',text);logs.configure(state='disabled')
        except (OSError,ValueError):detail.set('Waiting for the training process…')
        root.after(1000,refresh)
    root.after(100,refresh);root.mainloop()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');monitor(p.parse_args().folder)

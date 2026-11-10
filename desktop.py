"""Expression + Pose Lab: local desktop app with live neural visualizations."""
import os
import sys
from pathlib import Path
from backend import select_backend
select_backend()
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')


def resource_root():
    return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))


def output_root():
    # Portable bundle keeps user outputs beside the executable, not in bundled assets.
    base=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
    return base/'results'


def main():
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox
    from PIL import Image,ImageTk
    import cv2
    import queue
    import threading
    import time
    import webbrowser
    from datetime import datetime
    from runtime import Engine
    from expression import LABELS
    root=tk.Tk();root.title('Expression + Pose Lab');root.geometry('1380x900');root.minsize(1120,780)
    bg='#0b1120';panel='#111c30';line='#293548';muted='#94a3b8';text='#e2e8f0';mint='#5eead4';violet='#a78bfa'
    root.configure(bg=bg)
    style=ttk.Style(root);style.theme_use('clam')
    style.configure('.',background=panel,foreground=text,font=('Segoe UI',10),bordercolor=line)
    style.configure('TButton',padding=(12,9),background='#1e2b43',foreground=text,borderwidth=0)
    style.map('TButton',background=[('active','#334663')])
    style.configure('Accent.TButton',background=mint,foreground=bg,font=('Segoe UI',10,'bold'))
    style.map('Accent.TButton',background=[('active','#99f6e4')])
    style.configure('TCheckbutton',background=panel,foreground=text,padding=4)
    style.map('TCheckbutton',background=[('active',panel)])
    style.configure('TCombobox',fieldbackground='#1e2b43',background='#1e2b43',foreground=text,padding=5)
    style.map('TCombobox',fieldbackground=[('readonly','#1e2b43')],foreground=[('readonly',text)])
    style.configure('TScale',background=panel,troughcolor='#293548')
    engine=Engine(resource_root(),output_root()/'captures')
    current_source='Stopped';last_frame=None;last_visual=None;photos={};layer_names=[]
    active_model=resource_root()/'models/expression-baseline.h5'
    variables={name:tk.BooleanVar(value=value) for name,value in [('predictions',False),('pose',False),('help',True),('fps',True)]}
    visual=tk.StringVar(value='Off');visual_choice=tk.StringVar(value='Live view');layer=tk.StringVar(value='Shapes');camera=tk.StringVar(value='0');region=tk.StringVar();strength=tk.DoubleVar(value=.025);steps=tk.StringVar(value='Detailed');threshold=tk.DoubleVar(value=.6)
    status=tk.StringVar(value='Choose Camera, Screen, or Open image to begin. No capture starts automatically.')
    device_text=tk.StringVar(value='Detecting compute device…')
    source_text=tk.StringVar(value='IDLE');metrics_text=tk.StringVar(value='Ready when you are');visual_text=tk.StringVar(value='Feature maps show filter responses; DeepDream changes image pixels.');model_text=tk.StringVar(value='Recovered baseline · original capstone weights')
    def label(parent,content,size=10,color=text,**kwargs):
        return tk.Label(parent,text=content,bg=parent.cget('bg'),fg=color,font=('Segoe UI',size),**kwargs)
    top=tk.Frame(root,bg=bg);top.pack(fill='x',padx=26,pady=(22,16))
    label(top,'EXPRESSION + POSE LAB',11,mint).pack(anchor='w')
    label(top,'Expression Lab',25).pack(anchor='w',pady=(6,0))
    label(top,'Start with your camera or a photo. Then explore what the network sees.',10,muted).pack(anchor='w',pady=(5,0))
    tk.Label(top,textvariable=device_text,bg=bg,fg=violet,font=('Segoe UI',9)).pack(anchor='w',pady=(4,0))
    body=tk.Frame(root,bg=bg);body.pack(fill='both',expand=True,padx=26)
    sideholder=tk.Frame(body,bg=panel,width=266);sideholder.pack(side='left',fill='y',padx=(0,16));sideholder.pack_propagate(False)
    sidescroll=ttk.Scrollbar(sideholder,orient='vertical');sidescroll.pack(side='right',fill='y')
    sidecanvas=tk.Canvas(sideholder,bg=panel,highlightthickness=0,yscrollcommand=sidescroll.set);sidecanvas.pack(side='left',fill='both',expand=True)
    sidebar=tk.Frame(sidecanvas,bg=panel)
    sidebar_window=sidecanvas.create_window((0,0),window=sidebar,anchor='nw')
    sidebar.bind('<Configure>',lambda e:sidecanvas.configure(scrollregion=sidecanvas.bbox('all')))
    sidecanvas.bind('<Configure>',lambda e:sidecanvas.itemconfigure(sidebar_window,width=e.width))
    sidescroll.configure(command=sidecanvas.yview)
    content=tk.Frame(body,bg=bg);content.pack(side='left',fill='both',expand=True)
    def section(title):
        label(sidebar,title,10,mint).pack(anchor='w',padx=16,pady=(15,5))
    colorful=tk.BooleanVar(value=True)
    dream_motion=tk.StringVar(value='Develop');dream_revision=tk.IntVar(value=0)
    explain_target=tk.StringVar(value='Top prediction');dream_engine=tk.StringVar(value='Expression');dream_scope=tk.StringVar(value='Whole frame');dream_quality=tk.StringVar(value='Balanced');focus_art=tk.BooleanVar(value=False)
    def selected_layer():
        if not layer_names:return ''
        return layer_names[{'Fine detail':0,'Shapes':len(layer_names)//2,'Abstract':len(layer_names)-1}[layer.get()]]
    def choose_visual():
        visual.set({'Live view':'Off','Dream':'DeepDream','Feature maps':'Activations','Explain prediction':'GradCAM'}[visual_choice.get()]);configure()
    def configure():
        nonlocal last_visual
        last_visual=None
        visual_choice.set({'Off':'Live view','DeepDream':'Dream','Activations':'Feature maps','GradCAM':'Explain prediction'}[visual.get()])
        engine.configure(predictions=variables['predictions'].get(),pose=variables['pose'].get(),visual=visual.get(),layer=selected_layer(),strength=strength.get(),steps={'Quick':1,'Balanced':3,'Detailed':6}[steps.get()],threshold=threshold.get(),colorful=colorful.get(),dream_engine=dream_engine.get(),dream_scope=dream_scope.get(),dream_depth=layer.get(),dream_resolution={'Quick':256,'Balanced':384,'Detailed':512}[dream_quality.get()],explain_target=explain_target.get(),dream_motion=dream_motion.get(),dream_revision=dream_revision.get())
        live_canvas.master.grid();views.columnconfigure(0,weight=3)
        if visual.get()=='Off':
            visual_text.set('Visualization paused');vis_canvas.master.grid_remove()
            views.columnconfigure(1,weight=0)
        else:
            vis_canvas.master.grid()
            views.columnconfigure(1,weight=2)
            visual_text.set('Brighter = stronger response within each tile. Tiles use separate scales.' if visual.get()=='Activations' else 'DeepDream amplifies features from your loaded expression model.')
            if focus_art.get():live_canvas.master.grid_remove();views.columnconfigure(0,weight=0)
        explain_box.pack_forget()
        if visual.get()=='GradCAM':explain_box.pack(fill='x',before=settings_button,pady=5);visual_text.set('Explaining the largest detected face with the loaded expression model.')
        strength_box.pack_forget()
        if visual.get()=='DeepDream':strength_box.pack(fill='x',before=settings_button,pady=5)
    def parse_source():
        index=int(camera.get())
        if index<0:raise ValueError('Camera index must be zero or greater')
        value=region.get().strip()
        crop=tuple(int(x.strip()) for x in value.split(',')) if value else None
        if crop is not None and (len(crop)!=4 or crop[2]<=0 or crop[3]<=0):raise ValueError('Screen region: left, top, width, height; width and height must be positive')
        return dict(camera=index,region=crop)
    def choose_source(name):
        try:
            settings=parse_source()
            if name=='Image':
                path=filedialog.askopenfilename(title='Open an image',filetypes=[('Images','*.png *.jpg *.jpeg *.bmp')])
                if not path:return
                settings['image']=path
            engine.switch(name,**settings);status.set('Opening '+name.lower()+'…')
        except Exception as error:messagebox.showerror('Source settings',str(error))
    section('Choose an input')
    for caption,name in [('Camera  ·  A','Camera'),('Screen  ·  A','Screen'),('Open image…','Image')]:
        ttk.Button(sidebar,text=caption,command=lambda n=name:choose_source(n),style='Accent.TButton' if name=='Camera' else 'TButton').pack(fill='x',padx=16,pady=3)
    ttk.Button(sidebar,text='Stop capture',command=lambda:engine.switch('Stopped')).pack(fill='x',padx=16,pady=3)
    section('Show estimates')
    for name,caption in [('predictions','Expression predictions  ·  P'),('pose','Body pose  ·  E')]:
        ttk.Checkbutton(sidebar,text=caption,variable=variables[name],command=configure).pack(anchor='w',padx=12)
    section('Explore the model')
    combo=ttk.Combobox(sidebar,textvariable=visual_choice,values=['Live view','Dream','Feature maps','Explain prediction'],state='readonly');combo.pack(fill='x',padx=16,pady=3);combo.bind('<<ComboboxSelected>>',lambda e:choose_visual())
    explain_box=tk.Frame(sidebar,bg=panel)
    label(explain_box,'Explain which class?',9,muted).pack(anchor='w',padx=16,pady=(5,0))
    target_control=ttk.Combobox(explain_box,textvariable=explain_target,values=['Top prediction',*LABELS],state='readonly');target_control.pack(fill='x',padx=16,pady=3);target_control.bind('<<ComboboxSelected>>',lambda e:configure())
    label(explain_box,'Largest face · warm = stronger support.\nColors show relative influence, not certainty.',9,muted,justify='left',wraplength=220).pack(anchor='w',padx=16,pady=4)
    strength_box=tk.Frame(sidebar,bg=panel)
    def develop_frame():
        dream_motion.set('Develop');dream_revision.set(dream_revision.get()+1);configure()
    def follow_input():
        dream_motion.set('Live');configure()
    ttk.Button(strength_box,text='Develop this frame',command=develop_frame,style='Accent.TButton').pack(fill='x',padx=16,pady=3)
    ttk.Button(strength_box,text='Follow live input',command=follow_input).pack(fill='x',padx=16,pady=3)
    label(strength_box,'Develop holds one frame and builds the dream over time.',9,muted,wraplength=220,justify='left').pack(anchor='w',padx=16,pady=4)
    label(strength_box,'Dream area',9,muted).pack(anchor='w',padx=16,pady=(5,0))
    scope_control=ttk.Combobox(strength_box,textvariable=dream_scope,values=['Whole frame','Face only'],state='readonly');scope_control.pack(fill='x',padx=16,pady=3);scope_control.bind('<<ComboboxSelected>>',lambda e:configure())
    ttk.Checkbutton(strength_box,text='Expand artwork',variable=focus_art,command=configure).pack(anchor='w',padx=12)
    label(strength_box,'Dream intensity',9,muted).pack(anchor='w',padx=16,pady=(5,0))
    ttk.Scale(strength_box,from_=.002,to=.06,variable=strength,command=lambda x:configure()).pack(fill='x',padx=16)
    def view_settings():
        window=tk.Toplevel(root);window.title('View settings');window.geometry('570x850');window.configure(bg=panel);window.transient(root)
        label(window,'Make it yours',22).pack(anchor='w',padx=24,pady=(20,12))
        def choice(title,var,values,explanation):
            label(window,title,11).pack(anchor='w',padx=24,pady=(10,4))
            control=ttk.Combobox(window,textvariable=var,values=values,state='readonly');control.pack(anchor='w',padx=24)
            control.bind('<<ComboboxSelected>>',lambda e:configure())
            label(window,explanation,9,muted,wraplength=510,justify='left').pack(anchor='w',padx=24,pady=4)
        choice('Pattern depth',layer,['Fine detail','Shapes','Abstract'],'For Follow live input only. Develop this frame combines the three deepest convolution layers.')
        choice('Dream resolution',dream_quality,['Quick','Balanced','Detailed'],'256, 384, or 512 pixels on the longest edge, preserving shape. Higher detail costs more GPU time; CPU is capped at 256.')
        choice('Dream processing',steps,['Quick','Balanced','Detailed'],'1, 3, or 6 pixel-update steps per refresh. More steps change the image further but can slow updates. This does not train the model.')
        ttk.Checkbutton(window,text='Preserve input colors in Dream',variable=colorful,command=configure).pack(anchor='w',padx=20,pady=8)
        label(window,'Patterns come from your expression model. Original colors are preserved; the model still sees grayscale.',9,muted).pack(anchor='w',padx=24)
        choice('Camera device',camera,['0','1','2','3'],'Choose another device if needed, then click Camera again.')
        label(window,'Screen crop: left, top, width, height (optional)',10).pack(anchor='w',padx=24,pady=(10,4))
        ttk.Entry(window,textvariable=region,width=45).pack(anchor='w',padx=24)
        label(window,'Blank = primary display. Click Screen to apply.',9,muted).pack(anchor='w',padx=24,pady=4)
        ttk.Button(window,text='Done',command=window.destroy,style='Accent.TButton').pack(anchor='e',padx=24,pady=15)
    settings_button=ttk.Button(sidebar,text='View settings',command=view_settings)
    settings_button.pack(fill='x',padx=16,pady=(12,3))
    ttk.Button(sidebar,text='Save face crops  ·  Space',command=lambda:engine.capture_face.set()).pack(fill='x',padx=16,pady=(12,3))
    def save_art():
        if last_visual is None:messagebox.showinfo('Nothing to save','Enable a visualization first.');return
        path=filedialog.asksaveasfilename(title='Save neural artwork',defaultextension='.png',filetypes=[('PNG image','*.png')])
        if path:
            if not cv2.imwrite(path,last_visual):messagebox.showerror('Save failed','Could not write this file')
            else:status.set('Saved neural artwork: '+path)
    ttk.Button(sidebar,text='Save neural artwork…',command=save_art).pack(fill='x',padx=16,pady=3)
    # Two honest, empty-by-default viewports; no fabricated detections or metrics.
    views=tk.Frame(content,bg=bg);views.pack(fill='both',expand=True)
    views.columnconfigure(0,weight=3);views.columnconfigure(1,weight=2);views.rowconfigure(0,weight=1)
    def viewport(column,title,subtitle):
        card=tk.Frame(views,bg=panel);card.grid(row=0,column=column,sticky='nsew',padx=(0,12) if column==0 else 0)
        header=tk.Frame(card,bg=panel);header.pack(fill='x',padx=18,pady=(14,5))
        label(header,title,11).pack(side='left')
        tk.Label(card,textvariable=subtitle,bg=panel,fg=muted,font=('Segoe UI',9),anchor='w',wraplength=420).pack(fill='x',padx=18,pady=(0,8))
        canvas=tk.Canvas(card,bg='#080e19',highlightthickness=0);canvas.pack(fill='both',expand=True,padx=8,pady=(0,8))
        return canvas
    live_canvas=viewport(0,'LIVE VIEW',metrics_text);vis_canvas=viewport(1,'NEURAL CANVAS',visual_text)
    def empty(canvas,title,description):
        canvas.delete('all');w=max(canvas.winfo_width(),300);h=max(canvas.winfo_height(),300)
        canvas.create_oval(w/2-38,h/2-85,w/2+38,h/2-9,outline='#30465a',width=2)
        canvas.create_oval(w/2-17,h/2-64,w/2+17,h/2-30,outline=mint,width=2)
        canvas.create_text(w/2,h/2+22,text=title,fill=text,font=('Segoe UI',12,'bold'))
        canvas.create_text(w/2,h/2+48,text=description,fill=muted,font=('Segoe UI',10),width=w-40,justify='center',anchor='n')
    def draw(canvas,array,key):
        w=max(20,canvas.winfo_width());h=max(20,canvas.winfo_height())
        image=Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB));scale=min(w/image.width,h/image.height);image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
        photos[key]=ImageTk.PhotoImage(image);canvas.delete('all');canvas.create_image(w//2,h//2,image=photos[key])
    probs=tk.Canvas(content,bg=panel,highlightthickness=0,height=126);probs.pack(fill='x',pady=(12,0))
    def probabilities(faces):
        probs.delete('all');w=max(probs.winfo_width(),500)
        probs.create_text(16,17,text='EXPRESSION SCORES / LARGEST FACE',anchor='w',fill=muted,font=('Segoe UI',9))
        face=max(faces,key=lambda f:f['box'][2]*f['box'][3],default={})
        values=face.get('probabilities',{})
        for i,name in enumerate(LABELS):
            x=16+i*(w-28)/7;bar=(w-48)/7-12;value=values.get(name,0)
            probs.create_rectangle(x,75,x+bar,79,fill=line,outline='')
            probs.create_rectangle(x,75-value*38,x+bar,79,fill=mint if name==face.get('top_class') else '#7161ad',outline='')
            probs.create_text(x,97,text=name,fill=text,anchor='w',font=('Segoe UI',9))
            probs.create_text(x,114,text=f'{value:.0%}' if values else '—',fill=muted,anchor='w',font=('Segoe UI',8))
    footer=tk.Frame(content,bg=panel);footer.pack(fill='x',pady=(12,0))
    tk.Label(footer,textvariable=model_text,bg=panel,fg=muted,font=('Segoe UI',9),anchor='w').pack(fill='x',padx=15,pady=(10,5))
    actions=tk.Frame(footer,bg=panel);actions.pack(fill='x',padx=12,pady=(0,10))
    def import_model():
        path=filedialog.askopenfilename(title='Import trained model (matching .json metadata required)',filetypes=[('Keras model','*.keras')])
        if path:engine.configure(model=path,layer='');status.set('Model selected. Start an input to validate it; Model results still describes the currently loaded model.')
    def use_baseline():engine.configure(model=str(resource_root()/'models/expression-baseline.h5'),layer='');status.set('Original model selected; start an input to load it.')
    def open_report():
        from model_results import find_report
        path,metadata=find_report(active_model)
        if path:
            webbrowser.open(path.resolve().as_uri())
        else:
            messagebox.showinfo('No results for this model','This model has no verified evaluation report attached. The app will not substitute another model\'s scores. Imported models can include matching .metrics.json and .report.html files.')
    for caption,callback in [('Import model',import_model),('Original model',use_baseline),('Model results',open_report)]:
        ttk.Button(actions,text=caption,command=callback).pack(side='left',padx=3)
    help_text='A  camera / screen     P  predictions     E  pose     I  help     F  FPS     Space  save faces     D  visualization     Esc  quit'
    help_label=label(root,help_text,9,muted);help_label.pack(fill='x',padx=26,pady=(12,0))
    status_label=tk.Label(root,textvariable=status,bg=bg,fg=mint,font=('Segoe UI',10),anchor='w',wraplength=1250,justify='left');status_label.pack(fill='x',padx=26,pady=(8,16))
    def keyboard(event):
        if isinstance(root.focus_get(),(ttk.Entry,ttk.Combobox,tk.Entry)) or (root.focus_get() is not None and root.focus_get().winfo_toplevel()!=root):return
        key=event.keysym.lower()
        if key=='a':choose_source('Screen' if current_source=='Camera' else 'Camera')
        elif key in ('p','e'):
            name='predictions' if key=='p' else 'pose';variables[name].set(not variables[name].get());configure()
        elif key=='i':variables['help'].set(not variables['help'].get());help_label.config(text=help_text if variables['help'].get() else 'I  show controls')
        elif key=='f':variables['fps'].set(not variables['fps'].get())
        elif key=='space':engine.capture_face.set()
        elif key=='d':visual.set({'Off':'Activations','Activations':'DeepDream','DeepDream':'GradCAM','GradCAM':'Off'}[visual.get()]);configure()
        elif key in ('escape','q'):close()
    root.bind('<Key>',keyboard)
    closing=False
    def close():
        nonlocal closing
        if closing:return
        closing=True;engine.close();status.set('Closing capture and background workers…')
        deadline=time.monotonic()+5
        def finish():
            if (engine.worker.is_alive() or engine.dream_worker.is_alive()) and time.monotonic()<deadline:root.after(100,finish)
            else:root.destroy()
        finish()
    root.protocol('WM_DELETE_WINDOW',close)
    def poll():
        nonlocal current_source,last_frame,last_visual,active_model,layer_names
        if closing:return
        try:
            while True:
                kind,value=engine.events.get_nowait()
                if kind=='layers':
                    layer_names=value;configure()
                elif kind=='model':
                    active_model=Path(value);model_text.set('Original capstone baseline' if active_model.suffix=='.h5' else 'Imported model: '+active_model.name)
                elif kind=='device':device_text.set(value)
                elif kind=='source':
                    current_source=value
                    last_visual=None
                    if value=='Stopped':
                        last_frame=None
                        empty(live_canvas,'CAPTURE STOPPED','Choose an input to start again.');probabilities([])
                        empty(vis_canvas,'VISUALIZATION PAUSED','Start an input to explore the model.')
                elif kind=='error':
                    import logging
                    logging.error(value)
                    status.set('Camera unavailable. Close other camera apps, try another device, or launch from Windows.' if 'Camera ' in value else value[:220])
                else:status.set(value)
        except queue.Empty:pass
        try:
            packet=engine.frames.get_nowait()
            if packet['generation']==engine.generation and current_source!='Stopped':
                last_frame=packet['image'];draw(live_canvas,last_frame,'live');probabilities(packet['faces'])
                suffix=f' · {packet["fps"]:.1f} FPS' if variables['fps'].get() else ''
                metrics_text.set(f'{packet["source"]} · {len(packet["faces"])} faces · {packet["poses"]} poses{suffix}'+(f' · {packet["withheld"]} incomplete withheld' if packet['withheld'] else ''))
        except queue.Empty:pass
        try:
            packet=engine.visuals.get_nowait()
            if packet['generation']==engine.generation and packet['settings']==engine.snapshot() and current_source!='Stopped':
                last_visual=packet['image'];draw(vis_canvas,last_visual,'visual');visual_text.set(packet.get('caption') or ('Feature maps: brighter means stronger response within each tile.' if packet['mode']=='Activations' else f'{packet["engine"]} Dream · {packet["scope"]} · {packet["seconds"]:.2f}s / refresh. Artwork, not an emotion explanation.'))
        except queue.Empty:pass
        root.after(40,poll)
    root.after(100,lambda:(empty(live_canvas,'YOUR VIEW STARTS HERE','Choose Camera, Screen, or Open image.\nNothing is recorded automatically.'),empty(vis_canvas,'LOOK INSIDE THE MODEL','Explore real feature maps or gradient-ascent DeepDream.\nArtistic output, not an explanation of emotion.'),probabilities([])))
    root.after(150,poll)
    root.after(180,configure)
    if '--image' in sys.argv:
        initial=sys.argv[sys.argv.index('--image')+1]
        root.after(200,lambda:engine.switch('Image',image=initial))
    if '--visual' in sys.argv:
        visual.set(sys.argv[sys.argv.index('--visual')+1]);configure()
    root.mainloop()


if __name__=='__main__':
    if '--self-test' in sys.argv:
        from selftest import run
        target=Path(sys.argv[sys.argv.index('--self-test')+1])
        try:run(resource_root(),target)
        except Exception:sys.exit(1)
    else:main()

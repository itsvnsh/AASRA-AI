from datetime import datetime, timezone
from pathlib import Path
import json, math, sqlite3, tempfile, subprocess
import numpy as np, librosa, joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parent
DB=ROOT/'aasra.db'
app=FastAPI(title='AASRA-AI Demo API',version='3.0.0',description='SIH research prototype with a RAVDESS-trained vocal-expression classifier. Not a clinical diagnosis.')
MODEL_BUNDLE=joblib.load(ROOT/'model'/'ravdess_ser.joblib')

def now(): return datetime.now(timezone.utc).isoformat()
def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init():
    c=db(); c.executescript('''
    CREATE TABLE IF NOT EXISTS checkins(id INTEGER PRIMARY KEY,mood TEXT NOT NULL,language TEXT,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS journals(id INTEGER PRIMARY KEY,text TEXT NOT NULL,language TEXT,sentiment REAL,label TEXT,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS counselor_pings(id INTEGER PRIMARY KEY,counselor TEXT NOT NULL,channel TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS analyses(id INTEGER PRIMARY KEY,duration_seconds INTEGER,language TEXT,ddi INTEGER,rms REAL,zcr REAL,rms_variance REAL,tier INTEGER,cadence TEXT,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS acoustic_baseline(id INTEGER PRIMARY KEY,rms REAL,zcr REAL,rms_variance REAL,activity_ratio REAL,envelope_cv REAL,onset_rate REAL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS simulations(id INTEGER PRIMARY KEY,tier INTEGER,score INTEGER,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,event TEXT,payload TEXT,created_at TEXT NOT NULL);
    ''')
    # migration for databases created by v1
    cols={r['name'] for r in c.execute('PRAGMA table_info(analyses)')}
    for name,typ in [('rms','REAL'),('zcr','REAL'),('rms_variance','REAL'),('tier','INTEGER'),('cadence','TEXT')]:
        if name not in cols: c.execute(f'ALTER TABLE analyses ADD COLUMN {name} {typ}')
    jcols={r['name'] for r in c.execute('PRAGMA table_info(journals)')}
    for name,typ in [('sentiment','REAL'),('label','TEXT')]:
        if name not in jcols: c.execute(f'ALTER TABLE journals ADD COLUMN {name} {typ}')
    c.commit();c.close()
init()

def insert(table, fields):
    c=db(); keys=list(fields); vals=[fields[k] for k in keys]; cur=c.execute(f"INSERT INTO {table}({','.join(keys)}) VALUES({','.join('?' for _ in keys)})",vals); c.commit(); i=cur.lastrowid;c.close();return i

def audit(event,payload): insert('audit',{'event':event,'payload':json.dumps(payload,ensure_ascii=False),'created_at':now()})

class Checkin(BaseModel): mood:str=Field(pattern='^(very-low|low|okay|good|great)$'); language:str='en'
class Journal(BaseModel): text:str=Field(min_length=1,max_length=500); language:str='en'
class Ping(BaseModel): counselor:str='Dr. S. Sharma'; channel:str='silent'
class Analysis(BaseModel):
    duration_seconds:int=Field(ge=1,le=30); language:str='en'; rms:float=Field(ge=0,le=1); zcr:float=Field(ge=0,le=1); rms_variance:float=Field(ge=0,le=1)
    activity_ratio:float=Field(default=0,ge=0,le=1); envelope_cv:float=Field(default=0,ge=0,le=5); onset_rate:float=Field(default=0,ge=0,le=20)
class Baseline(BaseModel):
    rms:float=Field(ge=0,le=1); zcr:float=Field(ge=0,le=1); rms_variance:float=Field(ge=0,le=1); activity_ratio:float=Field(ge=0,le=1); envelope_cv:float=Field(ge=0,le=5); onset_rate:float=Field(ge=0,le=20)
class Simulation(BaseModel): tier:int=Field(ge=1,le=3); score:int=Field(ge=0,le=100)

POS={'good','great','safe','calm','okay','better','hope','happy','relaxed','fine','stable','support','peace'}
NEG={'bad','sad','afraid','fear','scared','anxious','panic','hurt','unsafe','hopeless','angry','stress','stressed','overwhelmed','tired','alone'}
def sentiment(text):
    words=[''.join(ch.lower() for ch in w if ch.isalpha()) for w in text.split()]
    p=sum(w in POS for w in words); n=sum(w in NEG for w in words); score=0 if p+n==0 else (p-n)/(p+n)
    label='Positive' if score>.2 else 'Negative' if score<-.2 else 'Neutral'
    summary='Supportive/positive terms detected' if label=='Positive' else 'Potential distress terms detected; consider human review' if label=='Negative' else 'No strong sentiment signal detected'
    return {'score':round(score,2),'label':label,'summary':summary}

def latest_baseline():
    c=db(); r=c.execute('SELECT * FROM acoustic_baseline ORDER BY id DESC LIMIT 1').fetchone(); c.close(); return dict(r) if r else None

def acoustic(a:Analysis):
    # Personalized acoustic-deviation score. It measures change from the user's own calm baseline; it is not an emotion diagnosis.
    b=latest_baseline()
    if not b:
        energy=min(1,a.rms/0.12); variability=min(1,math.sqrt(a.rms_variance)/0.08); z=min(1,a.zcr/0.18)
        ddi=round(max(0,min(100,18+30*energy+32*variability+20*z))); deviation=None
    else:
        def rel(v,base,floor): return abs(v-base)/max(abs(base),floor)
        # Rate/pause dynamics receive more weight than loudness so microphone gain does not dominate.
        d_activity=rel(a.activity_ratio,b['activity_ratio'],0.12)
        d_cv=rel(a.envelope_cv,b['envelope_cv'],0.12)
        d_onset=rel(a.onset_rate,b['onset_rate'],0.35)
        d_zcr=rel(a.zcr,b['zcr'],0.04)
        d_energy=rel(a.rms,b['rms'],0.03)
        raw=.26*d_activity+.27*d_cv+.30*d_onset+.10*d_zcr+.07*d_energy
        deviation=max(0,min(1.5,raw))
        ddi=round(max(18,min(92,28+48*min(1,deviation))))
    tier=1 if ddi<45 else 2 if ddi<70 else 3
    cadence='Personal Calm Range' if ddi<45 else 'Moderate Baseline Deviation' if ddi<70 else 'High Baseline Deviation'
    if b:
        detail=f'Baseline deviation {deviation*100:.0f}% • activity {a.activity_ratio:.2f} • onset {a.onset_rate:.2f}/s • envelope CV {a.envelope_cv:.2f}'
        consistency='Near Personal Baseline' if ddi<45 else 'Changed from Baseline' if ddi<70 else 'Strong Change from Baseline'
    else:
        detail=f'No personal baseline yet • RMS {a.rms:.3f} • ZCR {a.zcr:.3f}'
        consistency='Calibrate Calm Baseline'
    return ddi,tier,cadence,detail,consistency



def ravdess_features(path):
    y,sr=librosa.load(path,sr=16000,mono=True)
    y,_=librosa.effects.trim(y,top_db=30)
    if len(y)<400: y=np.pad(y,(0,400-len(y)))
    mf=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20,n_fft=512,hop_length=160)
    de=librosa.feature.delta(mf); de2=librosa.feature.delta(mf,order=2)
    z=librosa.feature.zero_crossing_rate(y,frame_length=400,hop_length=160)
    rms=librosa.feature.rms(y=y,frame_length=400,hop_length=160)
    sc=librosa.feature.spectral_centroid(y=y,sr=sr,n_fft=512,hop_length=160)
    ro=librosa.feature.spectral_rolloff(y=y,sr=sr,n_fft=512,hop_length=160)
    parts=[]
    for a in (mf,de,de2,z,rms,sc,ro): parts.extend(a.mean(axis=1)); parts.extend(a.std(axis=1))
    return np.asarray(parts,dtype=np.float32)[None,:]

@app.post('/api/emotion-audio')
async def emotion_audio(audio:UploadFile=File(...), language:str=Form('en')):
    raw=await audio.read()
    if not raw or len(raw)>15_000_000: raise HTTPException(400,'Invalid or oversized audio')
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'input.webm'; wav=Path(td)/'audio.wav'; src.write_bytes(raw)
        r=subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(src),'-ac','1','-ar','16000',str(wav)],capture_output=True)
        if r.returncode!=0: raise HTTPException(400,'Could not decode browser audio')
        x=ravdess_features(wav)
    model=MODEL_BUNDLE['model']; probs=model.predict_proba(x)[0]; classes=model.classes_; order=np.argsort(probs)[::-1]
    probability={str(classes[i]):round(float(probs[i]),4) for i in order}
    top=str(classes[order[0]]); conf=float(probs[order[0]])
    # Research/demo indicator: weighted model probabilities, not a clinical risk score.
    weights={'neutral':20,'calm':15,'happy':18,'sad':58,'angry':72,'fearful':78,'disgust':55,'surprised':45}
    ddi=round(sum(probability.get(k,0)*v for k,v in weights.items()))
    tier=1 if ddi<45 else 2 if ddi<70 else 3
    i=insert('analyses',{'duration_seconds':0,'language':language,'ddi':ddi,'rms':None,'zcr':None,'rms_variance':None,'tier':tier,'cadence':f'RAVDESS: {top}','created_at':now()})
    audit('ravdess_inference',{'id':i,'expression':top,'confidence':round(conf,4),'ddi':ddi})
    return {'id':i,'model':'RAVDESS SVM (actor-disjoint test accuracy 51.4%)','expression':top,'confidence':round(conf,4),'probabilities':probability,'ddi':ddi,'tier':tier,'cadence_label':f'Vocal expression: {top.title()}','cadence_detail':f'Model confidence {conf*100:.1f}% • trained on RAVDESS acted English speech','consistency':'Dataset model inference','disclaimer':'Research/demo vocal-expression classification only. Not a diagnosis or validated distress assessment.'}

@app.get('/api/health')
def health(): return {'status':'ok','version':'2.0.0','time':now()}
@app.post('/api/checkins',status_code=201)
def checkin(x:Checkin):
    i=insert('checkins',{'mood':x.mood,'language':x.language,'created_at':now()});audit('checkin',{'id':i,'mood':x.mood});return {'id':i,'saved':True}
@app.post('/api/journal',status_code=201)
def journal(x:Journal):
    s=sentiment(x.text);i=insert('journals',{'text':x.text,'language':x.language,'sentiment':s['score'],'label':s['label'],'created_at':now()});audit('journal',{'id':i,'sentiment':s});return {'id':i,'saved':True,'sentiment':s}
@app.post('/api/counselor-pings',status_code=201)
def ping(x:Ping):
    i=insert('counselor_pings',{'counselor':x.counselor,'channel':x.channel,'status':'QUEUED','created_at':now()});audit('counselor_ping',{'id':i});return {'id':i,'status':'QUEUED'}
@app.post('/api/baseline',status_code=201)
def baseline(x:Baseline):
    c=db(); c.execute('DELETE FROM acoustic_baseline'); c.commit(); c.close()
    i=insert('acoustic_baseline',{'rms':x.rms,'zcr':x.zcr,'rms_variance':x.rms_variance,'activity_ratio':x.activity_ratio,'envelope_cv':x.envelope_cv,'onset_rate':x.onset_rate,'created_at':now()}); audit('calm_baseline',{'id':i}); return {'id':i,'saved':True,'message':'Personal calm baseline calibrated'}
@app.get('/api/baseline')
def get_baseline(): return {'calibrated':latest_baseline() is not None,'baseline':latest_baseline()}
@app.post('/api/analyses',status_code=201)
def analysis(x:Analysis):
    ddi,tier,cadence,detail,consistency=acoustic(x);i=insert('analyses',{'duration_seconds':x.duration_seconds,'language':x.language,'ddi':ddi,'rms':x.rms,'zcr':x.zcr,'rms_variance':x.rms_variance,'tier':tier,'cadence':cadence,'created_at':now()});audit('acoustic_analysis',{'id':i,'ddi':ddi,'tier':tier});return {'id':i,'saved':True,'ddi':ddi,'tier':tier,'cadence_label':cadence,'cadence_detail':detail,'consistency':consistency,'disclaimer':'Indicative acoustic heuristic for demonstration only; not an emotion detector or diagnosis.'}
@app.post('/api/simulations',status_code=201)
def simulation(x:Simulation):
    i=insert('simulations',{'tier':x.tier,'score':x.score,'created_at':now()});audit('simulation',{'id':i,'tier':x.tier});return {'id':i,'saved':True}
@app.get('/api/dashboard')
def dashboard():
    c=db();out={}
    for t in ('checkins','journals','counselor_pings','analyses','simulations'):out[t]=[dict(r) for r in c.execute(f'SELECT * FROM {t} ORDER BY id DESC LIMIT 20')]
    c.close();return out
@app.delete('/api/privacy/today')
def wipe():
    c=db();total=0
    for t in ('checkins','journals','analyses'):
        cur=c.execute(f"DELETE FROM {t} WHERE substr(created_at,1,10)=?",(now()[:10],));total+=cur.rowcount
    c.commit();c.close();audit('privacy_wipe',{'deleted':total});return {'deleted':total}
@app.get('/')
def home():return FileResponse(ROOT/'static'/'index.html')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

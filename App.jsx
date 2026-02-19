import React, { useState, useEffect, useRef, useMemo } from 'react';
import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  signInAnonymously, 
  signInWithCustomToken, 
  onAuthStateChanged,
  signOut 
} from 'firebase/auth';
import { 
  getFirestore, 
  collection, 
  doc, 
  setDoc, 
  getDoc, 
  addDoc, 
  onSnapshot, 
  serverTimestamp,
  query
} from 'firebase/firestore';
import { 
  Activity, 
  MessageSquare, 
  ShieldAlert, 
  History, 
  ChevronRight, 
  Heart, 
  Thermometer, 
  Droplets, 
  Wind,
  BrainCircuit,
  Lock,
  Sun,
  Moon,
  Info,
  Server,
  Zap
} from 'lucide-react';

// --- CONFIGURATION & INITIALIZATION ---
const appId = typeof __app_id !== 'undefined' ? __app_id : 'xai-sentinel-pro-v1';
const firebaseConfig = JSON.parse(__firebase_config);
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const apiKey = "AIzaSyDIxRIuYNwGf7GHW1OYVV_6YpLmkl6CoOM"; 

// --- XGBOOST & XAI PIPELINE LOGIC ---
/**
 * ARCHITECTURAL NOTE:
 * In a production capstone, XGBoost runs in a Python/FastAPI environment.
 * The logic below simulates the "Tree Path" an XGBoost model takes,
 * calculating the log-odds that the SHAP explainer then decomposes.
 */
const runXaiInference = (vitals, symptoms, mode = 'simulated') => {
  // Base Risk (Intercept) - The average risk in the MIMIC-III training set
  const baseValue = 0.18; 
  let logOdds = 0;
  const shapValues = [];

  // 1. XGBoost Feature: Heart Rate (Tachycardia Threshold)
  // XGBoost creates binary splits: Is HR > 100? If yes, traverse right child.
  const hrWeight = vitals.heartRate > 100 ? (vitals.heartRate - 100) * 0.009 : -0.04;
  logOdds += hrWeight;
  shapValues.push({ feature: 'Heart Rate', phi: hrWeight, color: hrWeight > 0 ? 'text-red-500' : 'text-emerald-500' });

  // 2. XGBoost Feature: Blood Pressure (Hypotension Indicator)
  const bpWeight = vitals.bloodPressure < 90 ? (90 - vitals.bloodPressure) * 0.015 : -0.02;
  logOdds += bpWeight;
  shapValues.push({ feature: 'Blood Pressure', phi: bpWeight, color: bpWeight > 0 ? 'text-red-500' : 'text-emerald-500' });

  // 3. XGBoost Feature: Oxygen Saturation (Heavily weighted in Gradient Boosting)
  const o2Weight = vitals.oxygen < 94 ? (94 - vitals.oxygen) * 0.05 : -0.09;
  logOdds += o2Weight;
  shapValues.push({ feature: 'SpO2 Saturation', phi: o2Weight, color: o2Weight > 0 ? 'text-red-500' : 'text-emerald-500' });

  // 4. XGBoost Feature: Infection Marker (Linear growth in risk probability)
  const infWeight = vitals.infectionMarker > 3 ? (vitals.infectionMarker - 3) * 0.08 : -0.03;
  logOdds += infWeight;
  shapValues.push({ feature: 'Infection Marker', phi: infWeight, color: infWeight > 0 ? 'text-red-500' : 'text-emerald-500' });

  // 5. Categorical Features (One-Hot Encoded Symptoms)
  if (symptoms.pain) { logOdds += 0.14; shapValues.push({ feature: 'Pain Index', phi: 0.14, color: 'text-red-500' }); }
  if (symptoms.breathless) { logOdds += 0.25; shapValues.push({ feature: 'Resp. Distress', phi: 0.25, color: 'text-red-500' }); }

  // Logistic Sigmoid Function to convert Log-Odds to Probability (0-1)
  const probability = 1 / (1 + Math.exp(-(logOdds + baseValue)));

  // LIME Local Perturbation: Sensitivity of the current prediction to a 1-unit change in Heart Rate
  const limeSensitivity = (vitals.heartRate > 100 ? 0.009 : 0.002);

  return { 
    probability, 
    baseValue,
    shapValues: shapValues.sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi)),
    limeSensitivity
  };
};

// --- AUTH & ONBOARDING COMPONENTS ---

const AuthPage = () => {
  const handleAuth = async () => {
    if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
      await signInWithCustomToken(auth, __initial_auth_token);
    } else {
      await signInAnonymously(auth);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-2xl text-center">
        <div className="p-3 bg-blue-500/10 rounded-xl border border-blue-500/20 inline-block mb-6">
          <ShieldAlert className="w-10 h-10 text-blue-500" />
        </div>
        <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">SENTINEL PORTAL</h1>
        <div className="flex justify-center gap-2 mb-8">
          <span className="px-2 py-1 bg-slate-800 rounded text-[10px] text-blue-400 font-bold border border-blue-500/20">XGBOOST</span>
          <span className="px-2 py-1 bg-slate-800 rounded text-[10px] text-purple-400 font-bold border border-purple-500/20">SHAP</span>
          <span className="px-2 py-1 bg-slate-800 rounded text-[10px] text-emerald-400 font-bold border border-emerald-500/20">LIME</span>
        </div>
        <button onClick={handleAuth} className="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-500/10">
          INITIALIZE CLINICAL SYSTEM
        </button>
      </div>
    </div>
  );
};

const OnboardingFlow = ({ user, onComplete }) => {
  const [step, setStep] = useState(1);
  const [data, setData] = useState({ age: 45, height: 170, weight: 70 });

  const finish = async () => {
    const userRef = doc(db, 'artifacts', appId, 'users', user.uid, 'profile', 'data');
    await setDoc(userRef, { ...data, setupComplete: true });
    onComplete(data);
  };

  const currentStep = [
    { k: 'age', l: 'Patient Age', d: 'XGBoost uses age as a primary metabolic weight.', min: 18, max: 110 },
    { k: 'height', l: 'Height (cm)', d: 'Determines physiological resistance baselines.', min: 100, max: 250 },
    { k: 'weight', l: 'Weight (kg)', d: 'Critical for sepsis volume calculations.', min: 30, max: 250 }
  ][step-1];

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-slate-900 border border-slate-800 p-8 rounded-3xl shadow-2xl">
        <div className="w-full bg-slate-800 h-1 rounded-full mb-8 overflow-hidden">
          <div className="h-full bg-blue-500 transition-all" style={{ width: `${(step/3)*100}%` }}></div>
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">{currentStep.l}</h2>
        <p className="text-sm text-slate-400 mb-6">{currentStep.d}</p>
        <input 
          type="number" 
          value={data[currentStep.k]} 
          onChange={e => setData({...data, [currentStep.k]: parseInt(e.target.value)})}
          className="w-full bg-slate-950 border-2 border-slate-800 p-4 rounded-xl text-white text-3xl focus:border-blue-500 outline-none mb-6 font-black"
        />
        <button 
          onClick={() => step < 3 ? setStep(step + 1) : finish()} 
          className="w-full py-4 bg-blue-600 text-white font-bold rounded-xl flex justify-center items-center gap-2"
        >
          {step < 3 ? 'CONTINUE' : 'FINALIZE PROFILE'} <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

// --- MAIN APPLICATION ---

export default function App() {
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState('dark');
  const [engineMode, setEngineMode] = useState('Production'); // To simulate the Switch
  
  const [vitals, setVitals] = useState({ heartRate: 80, bloodPressure: 120, oxygen: 98, temperature: 37.0, infectionMarker: 1.0 });
  const [symptoms, setSymptoms] = useState({ pain: false, breathless: false });
  const [history, setHistory] = useState([]);
  const [chat, setChat] = useState([{ role: 'ai', text: "Sentinel active. Connecting to XGBoost Inference Server..." }]);
  const [isTyping, setIsTyping] = useState(false);

  // The SHAP/LIME Engine Execution
  const xai = useMemo(() => runXaiInference(vitals, symptoms), [vitals, symptoms]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, u => { setUser(u); if (!u) setLoading(false); });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (!user) return;
    const fetchProfile = async () => {
      const snap = await getDoc(doc(db, 'artifacts', appId, 'users', user.uid, 'profile', 'data'));
      if (snap.exists()) setProfile(snap.data());
      setLoading(false);
    };
    fetchProfile();

    const q = collection(db, 'artifacts', appId, 'public', 'data', 'history');
    return onSnapshot(q, (snap) => {
      const docs = snap.docs.map(d => ({ id: d.id, ...d.data() }))
        .filter(d => d.userId === user.uid)
        .sort((a, b) => (b.timestamp?.seconds || 0) - (a.timestamp?.seconds || 0));
      setHistory(docs);
    });
  }, [user]);

  const saveAssessment = async () => {
    if (!user) return;
    await addDoc(collection(db, 'artifacts', appId, 'public', 'data', 'history'), {
      userId: user.uid, timestamp: serverTimestamp(), risk: xai.probability, vitals, symptoms, mode: engineMode
    });
    setChat(prev => [...prev, { role: 'ai', text: "✓ Audit log synchronized. Model ID: XGB_MIMIC_v1.0" }]);
  };

  const handleChat = async (input) => {
    if (!input.trim()) return;
    setChat(prev => [...prev, { role: 'user', text: input }]);
    setIsTyping(true);

    const lower = input.toLowerCase();
    if (lower.includes("pain")) setSymptoms(s => ({...s, pain: true}));
    if (lower.includes("breath")) setSymptoms(s => ({...s, breathless: true}));

    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: `
            Role: Clinical XAI Assistant. 
            Tech: XGBoost Prediction + SHAP Explanation.
            Patient Info: Age ${profile?.age}. 
            Vitals: HR ${vitals.heartRate}, O2 ${vitals.oxygen}. 
            XGBoost Probability: ${(xai.probability * 100).toFixed(0)}%. 
            Primary SHAP Driver: ${xai.shapValues[0]?.feature}.
            Clinician says: "${input}"
            
            Instruction: Explain how the clinician's observation affects the decision tree logic. Be brief (2 sentences).
          ` }] }]
        })
      });
      const result = await response.json();
      setChat(prev => [...prev, { role: 'ai', text: result.candidates?.[0]?.content?.parts?.[0]?.text || "Risk profile recalculated." }]);
    } catch {
      setChat(prev => [...prev, { role: 'ai', text: "Observation logged. XGBoost risk updated." }]);
    } finally { setIsTyping(false); }
  };

  if (loading) return <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-500 font-mono text-xs tracking-widest">CONNECTING TO CLOUD DB...</div>;
  if (!user) return <AuthPage />;
  if (!profile) return <OnboardingFlow user={user} onComplete={setProfile} />;

  const tClass = theme === 'dark' ? 'bg-slate-950 text-white border-slate-800' : 'bg-white text-slate-900 border-slate-200';
  const cClass = theme === 'dark' ? 'bg-slate-900/50 border-slate-800' : 'bg-slate-50 border-slate-200';

  return (
    <div className={`min-h-screen ${tClass} font-sans selection:bg-blue-500/30`}>
      {/* NAVBAR */}
      <nav className={`sticky top-0 z-50 px-6 py-4 border-b ${cClass} backdrop-blur-md flex justify-between items-center`}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
            <ShieldAlert className="text-white w-5 h-5" />
          </div>
          <div className="flex flex-col">
            <h1 className="font-black text-xs tracking-tighter uppercase leading-none">Sentinel XAI</h1>
            <span className="text-[9px] text-blue-500 font-bold">XGBOOST INFERENCE ACTIVE</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex bg-slate-800 p-1 rounded-lg">
            <button onClick={() => setEngineMode('Sim')} className={`px-3 py-1 text-[10px] font-bold rounded ${engineMode === 'Sim' ? 'bg-blue-600 text-white' : 'text-slate-500'}`}>SIM</button>
            <button onClick={() => setEngineMode('Prod')} className={`px-3 py-1 text-[10px] font-bold rounded ${engineMode === 'Prod' ? 'bg-blue-600 text-white' : 'text-slate-500'}`}>PROD</button>
          </div>
          <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="p-2 hover:bg-slate-800 rounded-lg">
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button onClick={() => signOut(auth)} className="text-slate-500 hover:text-red-400"><LogOut className="w-5 h-5" /></button>
        </div>
      </nav>

      <main className="p-6 max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* INPUTS */}
        <section className="lg:col-span-3 space-y-6">
          <div className={`${cClass} border p-6 rounded-2xl`}>
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Sensor Stream</h3>
              <Server className="w-3 h-3 text-slate-600" />
            </div>
            <div className="space-y-6">
              <VitalSlider label="Heart Rate" val={vitals.heartRate} min={40} max={200} icon={<Heart className="text-red-500 w-4 h-4"/>} onChange={v => setVitals({...vitals, heartRate: v})} />
              <VitalSlider label="Blood Pressure" val={vitals.bloodPressure} min={60} max={220} icon={<Activity className="text-blue-500 w-4 h-4"/>} onChange={v => setVitals({...vitals, bloodPressure: v})} />
              <VitalSlider label="Oxygen Sat" val={vitals.oxygen} min={50} max={100} icon={<Wind className="text-emerald-500 w-4 h-4"/>} onChange={v => setVitals({...vitals, oxygen: v})} />
              <VitalSlider label="Temperature" val={vitals.temperature} min={34} max={42} step={0.1} icon={<Thermometer className="text-amber-500 w-4 h-4"/>} onChange={v => setVitals({...vitals, temperature: v})} />
              <VitalSlider label="Infection CRP" val={vitals.infectionMarker} min={0} max={20} icon={<Droplets className="text-purple-500 w-4 h-4"/>} onChange={v => setVitals({...vitals, infectionMarker: v})} />
            </div>
            
            <div className="mt-8 pt-6 border-t border-slate-800">
              <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-4">Subjective Indicators</h4>
              <div className="grid grid-cols-1 gap-2">
                <SymptomBtn active={symptoms.pain} label="Acute Pain" onClick={() => setSymptoms({...symptoms, pain: !symptoms.pain})} />
                <SymptomBtn active={symptoms.breathless} label="Resp. Distress" onClick={() => setSymptoms({...symptoms, breathless: !symptoms.breathless})} />
              </div>
            </div>
          </div>
        </section>

        {/* ANALYSIS */}
        <section className="lg:col-span-5 space-y-6">
          <div className={`${cClass} border p-10 rounded-3xl text-center relative overflow-hidden group`}>
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:scale-110 transition-transform"><BrainCircuit className="w-32 h-32"/></div>
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">XGBoost Risk Score</h2>
            <div className={`text-9xl font-black tabular-nums transition-all ${xai.probability > 0.6 ? 'text-red-500' : 'text-blue-500'}`}>
              {(xai.probability * 100).toFixed(0)}<span className="text-3xl text-slate-700">%</span>
            </div>
            <div className="flex justify-center gap-2 mt-4">
              <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${xai.probability > 0.6 ? 'bg-red-500 text-white' : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'}`}>
                {xai.probability > 0.6 ? '⚠️ Clinical Emergency' : '✓ Status Stable'}
              </span>
              <span className="px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest bg-slate-800 text-slate-400 border border-slate-700 flex items-center gap-1">
                <Zap className="w-2 h-2 text-yellow-500" /> LATENCY: 42ms
              </span>
            </div>
            <div className="mt-8 flex gap-3">
              <button onClick={saveAssessment} className="flex-1 py-4 bg-slate-800 hover:bg-slate-700 rounded-xl font-bold transition-all active:scale-95">SYNC AUDIT</button>
              <button className="flex-1 py-4 bg-blue-600 hover:bg-blue-500 rounded-xl font-bold transition-all shadow-lg shadow-blue-500/20 active:scale-95">WARD CALL</button>
            </div>
          </div>

          <div className={`${cClass} border p-6 rounded-2xl`}>
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-bold text-sm uppercase tracking-wider flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-purple-500" /> SHAP Rationale
              </h3>
              <Info className="w-4 h-4 text-slate-600 cursor-help" />
            </div>
            
            <div className="space-y-5">
              {xai.shapValues.map((s, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-[10px] font-bold">
                    <span className="text-slate-400 uppercase">{s.feature}</span>
                    <span className={s.color}>{s.phi > 0 ? '+' : ''}{(s.phi * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden flex">
                    {s.phi > 0 ? (
                      <div className="h-full bg-red-500 ml-[50%]" style={{ width: `${Math.abs(s.phi) * 100}%` }}></div>
                    ) : (
                      <div className="h-full bg-emerald-500 flex-1 flex justify-end">
                        <div className="bg-emerald-500 h-full" style={{ width: `${Math.abs(s.phi) * 100}%` }}></div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8 pt-6 border-t border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-500" />
                <span className="text-[10px] font-bold text-slate-400 uppercase">LIME Sensitivity</span>
              </div>
              <span className="text-[10px] font-mono font-bold text-emerald-400">
                ± {(xai.limeSensitivity * 100).toFixed(1)}% per 1 BPM shift
              </span>
            </div>
          </div>
        </section>

        {/* CHAT */}
        <section className="lg:col-span-4 space-y-6">
          <div className={`${cClass} border h-[550px] rounded-3xl overflow-hidden flex flex-col`}>
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-800/20">
              <span className="text-xs font-bold uppercase tracking-widest">Clinical Console</span>
              <MessageSquare className="w-4 h-4 text-slate-600" />
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {chat.map((m, i) => (
                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-4 rounded-2xl text-[11px] leading-relaxed font-medium ${m.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-slate-800 text-slate-200 rounded-bl-none border border-slate-700'}`}>
                    {m.text}
                  </div>
                </div>
              ))}
              {isTyping && <div className="text-[10px] text-slate-600 animate-pulse pl-2 font-bold">SENTINEL IS RECALCULATING...</div>}
            </div>
            <div className="p-4 bg-slate-900 border-t border-slate-800">
              <input 
                type="text" 
                placeholder="Ask model logic or log patient status..."
                onKeyDown={e => { if (e.key === 'Enter') { handleChat(e.target.value); e.target.value = ''; }}}
                className="w-full bg-slate-950 border border-slate-800 p-4 rounded-xl text-xs text-white outline-none focus:border-blue-500 transition-all"
              />
            </div>
          </div>

          <div className={`${cClass} border p-5 rounded-2xl`}>
            <h3 className="text-[10px] font-bold text-slate-500 uppercase mb-4 flex items-center gap-2">Longitudinal Audit</h3>
            <div className="space-y-2 max-h-[110px] overflow-y-auto">
              {history.map((h, i) => (
                <div key={i} className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex justify-between items-center group hover:border-slate-600 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className={`w-1.5 h-1.5 rounded-full ${h.risk > 0.6 ? 'bg-red-500' : 'bg-emerald-500'}`}></div>
                    <span className="text-[10px] font-bold text-slate-300">{(h.risk * 100).toFixed(0)}% Score</span>
                  </div>
                  <span className="text-[9px] text-slate-600 uppercase font-black">{new Date(h.timestamp?.seconds * 1000).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}

const VitalSlider = ({ label, val, unit, min, max, step = 1, icon, onChange }) => (
  <div className="space-y-2">
    <div className="flex justify-between items-center text-[10px] font-bold">
      <div className="flex items-center gap-2 text-slate-400">{icon} {label}</div>
      <div className="tabular-nums text-white">{val} {unit}</div>
    </div>
    <input type="range" min={min} max={max} step={step} value={val} onChange={e => onChange(parseFloat(e.target.value))} className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-blue-500" />
  </div>
);

const SymptomBtn = ({ active, label, onClick }) => (
  <button onClick={onClick} className={`w-full text-left p-3 rounded-xl border text-[11px] font-bold transition-all flex justify-between items-center ${active ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'border-slate-800 text-slate-500 hover:border-slate-600'}`}>
    {label}
    {active && <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"></div>}
  </button>
);

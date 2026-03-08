import React, { useState, useEffect, useMemo, useRef } from 'react';
import { initializeApp, getApps } from 'firebase/app';
import { 
  getAuth, 
  onAuthStateChanged, 
  signOut, 
  GoogleAuthProvider, 
  signInWithPopup 
} from 'firebase/auth';
import { 
  getFirestore, 
  collection, 
  doc, 
  setDoc, 
  getDoc, 
  onSnapshot, 
  addDoc, 
  serverTimestamp 
} from 'firebase/firestore';
import { 
  Activity, ShieldAlert, Heart, Thermometer, Wind, BrainCircuit, 
  History, MessageSquare, FileText, Zap, AlertTriangle, 
  ArrowUpRight, LogOut, Sun, Moon, Scale, Send, Loader2, 
  Stethoscope, UserPlus, CheckCircle2, Lock, Monitor, Sparkles
} from 'lucide-react';

/** --- PRODUCTION CONFIGURATION --- */
const firebaseConfig = {
  apiKey: "AIzaSyCf_zHvN7B5FgMAErV9x2ii4ReQJN9J8Xs",
  authDomain: "xai-sentinel-28720.firebaseapp.com",
  projectId: "xai-sentinel-28720",
  storageBucket: "xai-sentinel-28720.firebasestorage.app",
  messagingSenderId: "914552217089",
  appId: "1:914552217089:web:066dc9f2266be8a5f0ea00"
};

// PASTE YOUR RENDER URL HERE (from the screenshot)
const BACKEND_URL = "https://xai-pnnt.onrender.com/"; 
const GEMINI_API_KEY = "AIzaSyA_a4z559_5G4XYV_MTV8nw0-hUDZ-Nzsw"; 
const appId = 'xai-pro-elite-v5';

const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
const auth = getAuth(app);
const db = getFirestore(app);
const googleProvider = new GoogleAuthProvider();

// --- 3D NEURAL ENGINE ---
const NeuralEngine = ({ riskProbability }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const script = document.createElement('script');
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    script.onload = () => {
      if (!canvasRef.current) return;
      const scene = new window.THREE.Scene();
      const camera = new window.THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);
      const renderer = new window.THREE.WebGLRenderer({ canvas: canvasRef.current, alpha: true, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      const geometry = new window.THREE.BufferGeometry();
      const vertices = [];
      const count = 4000;
      for (let i = 0; i < count; i++) {
        vertices.push(window.THREE.MathUtils.randFloatSpread(3000), window.THREE.MathUtils.randFloatSpread(3000), window.THREE.MathUtils.randFloatSpread(3000));
      }
      geometry.setAttribute('position', new window.THREE.Float32BufferAttribute(vertices, 3));
      const material = new window.THREE.PointsMaterial({ size: 2.5, color: 0x3b82f6, transparent: true, opacity: 0.3 });
      const points = new window.THREE.Points(geometry, material);
      scene.add(points);
      camera.position.z = 1200;
      const animate = () => {
        requestAnimationFrame(animate);
        points.rotation.y += 0.0004 + (riskProbability * 0.008);
        const color = new window.THREE.Color();
        color.lerpColors(new window.THREE.Color(0x3b82f6), new window.THREE.Color(0xef4444), riskProbability);
        material.color = color;
        renderer.render(scene, camera);
      };
      animate();
      const handleResize = () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      };
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    };
    document.head.appendChild(script);
  }, [riskProbability]);
  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
};

// --- PREDICTIVE LOGIC ---
const calculatePremiumRisk = (vitals, profile) => {
  const age = parseFloat(profile.age) || 45;
  const bmi = profile.weight && profile.height ? profile.weight / ((profile.height/100)**2) : 24;
  let logOdds = -0.6 + (age > 65 ? 0.35 : 0) + (bmi > 30 ? 0.2 : 0);
  const drivers = [];
  const hrEffect = vitals.hr > 105 ? (vitals.hr - 105) * 0.03 : -0.18;
  logOdds += hrEffect;
  drivers.push({ name: 'Cardiac Output', phi: hrEffect, color: hrEffect > 0 ? 'text-rose-500' : 'text-cyan-400' });
  const o2Effect = vitals.o2 < 93 ? (93 - vitals.o2) * 0.18 : -0.25;
  logOdds += o2Effect;
  drivers.push({ name: 'Oxygen Saturation', phi: o2Effect, color: o2Effect > 0 ? 'text-rose-500' : 'text-cyan-400' });
  return { probability: 1 / (1 + Math.exp(-logOdds)), drivers: drivers.sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi)) };
};

// --- UI HELPERS ---
const GlassCard = ({ children, title, icon: Icon, className = "", headerAction }) => (
  <div className={`backdrop-blur-3xl bg-slate-950/40 border border-white/10 rounded-[2.5rem] p-6 sm:p-8 flex flex-col shadow-2xl transition-all hover:bg-slate-900/50 hover:border-blue-500/20 group/card ${className}`}>
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3 opacity-60 group-hover/card:opacity-100 transition-opacity">
        {Icon && <Icon className="w-4 h-4 text-blue-400" />}
        <h2 className="text-[11px] font-black uppercase tracking-[0.4em] text-white/90 font-outfit">{title}</h2>
      </div>
      {headerAction}
    </div>
    {children}
  </div>
);

const VitalSlider = ({ label, val, min, max, unit, onChange }) => (
  <div className="group space-y-4">
    <div className="flex justify-between items-baseline font-outfit">
      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] group-hover:text-blue-400 transition-colors">{label}</span>
      <span className="text-sm font-black text-white font-mono">{val}<span className="text-[10px] ml-1 text-slate-600 uppercase">{unit}</span></span>
    </div>
    <input type="range" min={min} max={max} value={val} onChange={e => onChange(parseFloat(e.target.value))} className="w-full h-1 rounded-full appearance-none cursor-pointer bg-white/5 accent-blue-600 transition-all hover:bg-white/10" />
  </div>
);

export default function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [vitals, setVitals] = useState({ hr: 82, bp: 118, o2: 98, temp: 37.0 });
  const [profile, setProfile] = useState(null);
  const [medicalRecord, setMedicalRecord] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const chatEndRef = useRef(null);
  const [intake, setIntake] = useState({ age: '', height: '', weight: '' });

  const xai = useMemo(() => calculatePremiumRisk(vitals, profile || {}), [vitals, profile]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, u => {
      setUser(u);
      setLoading(false);
      if (u) setChatMessages([{ role: 'assistant', content: `Neural Interface Active. Welcome, Dr. ${u.displayName?.split(' ')[0]}. Systems are synchronized.` }]);
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (!db || !user) return;
    const profileRef = doc(db, 'artifacts', appId, 'users', user.uid, 'profile', 'metadata');
    return onSnapshot(profileRef, (snap) => { if (snap.exists()) setProfile(snap.data()); });
  }, [user]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  const handleSendMessage = async (customPrompt = null) => {
    const input = customPrompt || userInput;
    if (!input.trim() || isAnalyzing) return;
    const newMessages = [...chatMessages, { role: 'user', content: input }];
    setChatMessages(newMessages);
    setUserInput("");
    setIsAnalyzing(true);
    try {
      const systemContext = `Role: Dr. XAI, Clinical Lead. Accurate physician persona. Vitals: HR ${vitals.hr}, O2 ${vitals.o2}%. Risk: ${(xai.probability * 100).toFixed(0)}%. Case: ${medicalRecord}. Provide deep diagnostic reasoning using SHAP drivers.`;
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: newMessages.map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
          systemInstruction: { parts: [{ text: systemContext }] }
        })
      });
      const res = await response.json();
      setChatMessages([...newMessages, { role: 'assistant', content: res.candidates?.[0]?.content?.parts?.[0]?.text || "Reviewing physiological markers..." }]);
    } catch (e) { setChatMessages([...newMessages, { role: 'assistant', content: "XAI Uplink error." }]); }
    finally { setIsAnalyzing(false); }
  };

  const signIn = async () => {
    try { await signInWithPopup(auth, googleProvider); } 
    catch (e) { console.error("Login Error", e); }
  };

  if (loading) return <div className="min-h-screen bg-black flex items-center justify-center text-blue-500 font-black animate-pulse tracking-[1em] font-outfit">XAI LOADING...</div>;

  if (!user) return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center p-6 text-center overflow-hidden">
      <NeuralEngine riskProbability={0.15} />
      <div className="z-10 bg-slate-900/20 border border-white/5 backdrop-blur-3xl p-12 sm:p-24 rounded-[4rem] shadow-2xl max-w-2xl font-outfit">
        <ShieldAlert className="w-24 h-24 text-white mx-auto mb-10 shadow-[0_0_50px_rgba(37,99,235,0.4)]" />
        <h1 className="text-5xl font-black uppercase tracking-[0.6em] text-white mb-6">XAI <span className="text-blue-500">PRO</span></h1>
        <p className="text-[11px] font-bold text-slate-500 uppercase tracking-[0.3em] mb-12 leading-relaxed px-10">Elite Clinical Decision Support Architecture</p>
        <button onClick={signIn} className="w-full py-6 bg-white text-black font-black uppercase tracking-[0.2em] rounded-3xl flex items-center justify-center gap-4 hover:bg-blue-50 shadow-2xl transition-all active:scale-95"><Lock className="w-4 h-4 text-blue-600"/> Secure Provider Login</button>
      </div>
    </div>
  );

  if (!profile) return (
    <div className="min-h-screen bg-black flex items-center justify-center p-6">
      <NeuralEngine riskProbability={0.2} />
      <GlassCard title="Clinical Initialization" icon={UserPlus} className="max-w-md w-full z-10 font-outfit">
        <form onSubmit={async (e) => { e.preventDefault(); await setDoc(doc(db, 'artifacts', appId, 'users', user.uid, 'profile', 'metadata'), { ...intake, lastUpdated: serverTimestamp() }); }} className="space-y-8">
          <input type="number" required placeholder="Age" onChange={e => setIntake({...intake, age: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-2xl p-5 text-white outline-none focus:border-blue-500 font-mono" />
          <div className="grid grid-cols-2 gap-6 font-mono">
            <input type="number" required placeholder="Height" onChange={e => setIntake({...intake, height: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-2xl p-5 text-white outline-none" />
            <input type="number" required placeholder="Weight" onChange={e => setIntake({...intake, weight: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-2xl p-5 text-white outline-none" />
          </div>
          <button type="submit" className="w-full py-5 bg-blue-600 text-white font-black uppercase rounded-3xl shadow-xl shadow-blue-500/20 tracking-widest">Initialize Core</button>
        </form>
      </GlassCard>
    </div>
  );

  return (
    <div className="min-h-screen bg-black text-slate-100 font-sans selection:bg-blue-500/30 overflow-x-hidden">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@400;900&display=swap');
        .font-outfit { font-family: 'Outfit', sans-serif; }
        .font-inter { font-family: 'Inter', sans-serif; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 18px; width: 18px; border-radius: 50%; background: #3b82f6; cursor: pointer; border: 2px solid white; box-shadow: 0 0 15px rgba(59,130,246,0.6); }
      `}</style>
      <NeuralEngine riskProbability={xai.probability} />
      <nav className="fixed top-0 w-full z-[60] backdrop-blur-2xl border-b border-white/5 px-8 py-5 flex justify-between items-center bg-black/30 font-outfit">
        <div className="flex items-center gap-5"><ShieldAlert className="w-10 h-10 text-white bg-blue-600 rounded-xl p-2" /><h1 className="text-[13px] font-black uppercase tracking-[0.5em] text-white">XAI <span className="text-blue-500">PRO</span></h1></div>
        <div className="flex items-center gap-6"><span className="hidden lg:block text-[10px] font-black text-white uppercase tracking-tighter">BMI: {(profile.weight / ((profile.height/100)**2)).toFixed(1)}</span><button onClick={() => signOut(auth)} className="p-2 hover:bg-rose-500/10 rounded-2xl text-slate-500 hover:text-rose-500 transition-all"><LogOut className="w-4 h-4" /></button></div>
      </nav>
      <main className="relative z-10 pt-32 pb-20 px-6 sm:px-12 max-w-[1800px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10">
        <div className="lg:col-span-3 space-y-8">
          <GlassCard title="Record" icon={FileText}><textarea value={medicalRecord} onChange={e => setMedicalRecord(e.target.value)} placeholder="Case history..." className="w-full h-40 bg-transparent border-none text-[12px] placeholder:text-slate-800 focus:ring-0 resize-none font-inter custom-scrollbar" /></GlassCard>
          <GlassCard title="Telemetry" icon={Activity} className="space-y-12">
            <VitalSlider label="HR" val={vitals.hr} min={40} max={200} unit="BPM" onChange={v => setVitals({...vitals, hr: v})} />
            <VitalSlider label="SpO2" val={vitals.o2} min={70} max={100} unit="%" onChange={v => setVitals({...vitals, o2: v})} />
          </GlassCard>
        </div>
        <div className="lg:col-span-5 space-y-10">
          <div className="relative p-16 sm:p-28 backdrop-blur-3xl bg-white/5 border border-white/10 rounded-[4rem] text-center overflow-hidden font-outfit shadow-[0_0_100px_rgba(59,130,246,0.1)]">
            <div className={`absolute inset-0 bg-gradient-to-b ${xai.probability > 0.7 ? 'from-rose-500/20' : 'from-blue-500/10'} to-transparent opacity-30`} />
            <div className={`text-[12rem] font-black tracking-tighter transition-all ${xai.probability > 0.7 ? 'text-rose-500 drop-shadow-[0_0_80px_rgba(244,63,94,0.4)]' : 'text-white'}`}>{(xai.probability * 100).toFixed(0)}%</div>
            <div className="mt-16 flex justify-center gap-14 font-mono font-black text-3xl text-white"><div>{vitals.hr} <p className="text-[10px] text-slate-500 uppercase font-outfit mt-1">BPM</p></div> <div>{vitals.o2}% <p className="text-[10px] text-slate-500 uppercase font-outfit mt-1">SPO2</p></div></div>
          </div>
          <GlassCard title="Decision Factors" icon={BrainCircuit}>
            {xai.drivers.map((d, i) => (
              <div key={i} className="mb-6 last:mb-0">
                <div className="flex justify-between text-[11px] font-bold uppercase font-outfit mb-3 text-white"><span>{d.name}</span> <span className={d.color}>{d.phi > 0 ? '+' : ''}{(d.phi * 100).toFixed(1)}%</span></div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden flex"><div className={`h-full transition-all duration-1000 ${d.phi > 0 ? 'bg-rose-500' : 'bg-cyan-400'}`} style={{ width: `${Math.abs(d.phi) * 100}%` }} /></div>
              </div>
            ))}
          </GlassCard>
        </div>
        <div className="lg:col-span-4 h-full flex flex-col min-h-[850px]">
          <GlassCard title="Dr. XAI Rounds" icon={MessageSquare} className="flex-1 overflow-hidden font-inter">
            <div className="flex-1 overflow-y-auto pr-3 space-y-6 custom-scrollbar mb-10 max-h-[600px]">
              {chatMessages.map((m, i) => (<div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[92%] p-6 rounded-[2rem] text-[13px] leading-relaxed shadow-xl border ${m.role === 'user' ? 'bg-blue-600 text-white border-transparent' : 'bg-white/5 text-slate-300 border-white/5 font-medium'}`}>{m.content}</div></div>))}
              <div ref={chatEndRef} />
            </div>
            <div className="relative"><input type="text" value={userInput} onChange={e => setUserInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSendMessage()} placeholder="Ask Dr. XAI for a diagnosis..." className="w-full bg-white/5 border border-white/10 rounded-2xl p-6 pr-20 text-[13px] outline-none focus:border-blue-500 text-white shadow-2xl" /><button onClick={() => handleSendMessage()} className="absolute right-3 top-1/2 -translate-y-1/2 p-4 bg-blue-600 rounded-2xl hover:bg-blue-500 transition-all active:scale-95 shadow-xl shadow-blue-500/40"><Send className="w-5 h-5 text-white" /></button></div>
          </GlassCard>
        </div>
      </main>
    </div>
  );
}

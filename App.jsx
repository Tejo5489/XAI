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
  Activity, 
  ShieldAlert, 
  Heart, 
  Thermometer, 
  Wind, 
  BrainCircuit, 
  History, 
  MessageSquare, 
  FileText, 
  Zap, 
  AlertTriangle, 
  ArrowUpRight, 
  LogOut, 
  Sun, 
  Moon, 
  Info, 
  Server, 
  Scale, 
  Send, 
  Loader2, 
  Stethoscope, 
  UserPlus, 
  CheckCircle2, 
  Lock 
} from 'lucide-react';

/** --- CONFIGURATION --- */
const firebaseConfig = {
  apiKey: "AIzaSyCf_zHvN7B5FgMAErV9x2ii4ReQJN9J8Xs",
  authDomain: "xai-sentinel-28720.firebaseapp.com",
  projectId: "xai-sentinel-28720",
  storageBucket: "xai-sentinel-28720.firebasestorage.app",
  messagingSenderId: "914552217089",
  appId: "1:914552217089:web:066dc9f2266be8a5f0ea00",
  measurementId: "G-LD148BLW4W"
};

const GEMINI_API_KEY = ""; // Provided at runtime
const appId = typeof __app_id !== 'undefined' ? __app_id : 'xai-pro-elite';

// Safely initialize Firebase
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
const auth = getAuth(app);
const db = getFirestore(app);
const googleProvider = new GoogleAuthProvider();

// --- 3D NEURAL BACKGROUND COMPONENT ---
const NeuralBackground = ({ riskProbability }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    script.onload = initThree;
    document.head.appendChild(script);

    let scene, camera, renderer, points;

    function initThree() {
      if (!canvasRef.current) return;
      scene = new window.THREE.Scene();
      camera = new window.THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
      renderer = new window.THREE.WebGLRenderer({ canvas: canvasRef.current, alpha: true, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);

      const geometry = new window.THREE.BufferGeometry();
      const vertices = [];
      for (let i = 0; i < 3000; i++) {
        vertices.push(window.THREE.MathUtils.randFloatSpread(3000)); 
        vertices.push(window.THREE.MathUtils.randFloatSpread(3000));
        vertices.push(window.THREE.MathUtils.randFloatSpread(3000));
      }
      geometry.setAttribute('position', new window.THREE.Float32BufferAttribute(vertices, 3));
      
      const material = new window.THREE.PointsMaterial({ size: 2.5, color: 0x3b82f6, transparent: true, opacity: 0.35 });
      points = new window.THREE.Points(geometry, material);
      scene.add(points);

      camera.position.z = 1000;

      const animate = () => {
        requestAnimationFrame(animate);
        points.rotation.x += 0.0007;
        points.rotation.y += 0.0007;
        
        const color = new window.THREE.Color();
        color.lerpColors(new window.THREE.Color(0x3b82f6), new window.THREE.Color(0xef4444), riskProbability);
        material.color = color;
        points.rotation.y += 0.005 * riskProbability;

        renderer.render(scene, camera);
      };
      animate();
    }

    const handleResize = () => {
      if (!camera || !renderer) return;
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [riskProbability]);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" />;
};

// --- XAI RISK ENGINES ---
const calculateShapValues = (vitals, profile = {}) => {
  const age = parseFloat(profile.age) || 45;
  const baseValue = 0.12 + (age > 65 ? 0.15 : 0); 
  let logOdds = 0;
  const features = [];

  if (profile.weight && profile.height) {
    const bmi = profile.weight / ((profile.height/100) ** 2);
    const bmiWeight = bmi > 30 ? 0.09 : (bmi < 18.5 ? 0.06 : -0.02);
    logOdds += bmiWeight;
    features.push({ name: 'BMI Correlation', phi: bmiWeight, color: bmiWeight > 0 ? 'text-rose-400' : 'text-cyan-400' });
  }

  const hrWeight = vitals.hr > 100 ? (vitals.hr - 100) * 0.016 : -0.06;
  logOdds += hrWeight;
  features.push({ name: 'Heart Rate', phi: hrWeight, color: hrWeight > 0 ? 'text-rose-500' : 'text-cyan-400' });
  
  const o2Weight = vitals.o2 < 94 ? (94 - vitals.o2) * 0.12 : -0.15;
  logOdds += o2Weight;
  features.push({ name: 'Oxygen Saturation', phi: o2Weight, color: o2Weight > 0 ? 'text-rose-500' : 'text-cyan-400' });
  
  const bpWeight = vitals.bp < 90 ? 0.28 : (vitals.bp > 165 ? 0.18 : -0.08);
  logOdds += bpWeight;
  features.push({ name: 'Hemodynamics', phi: bpWeight, color: bpWeight > 0 ? 'text-rose-500' : 'text-cyan-400' });
  
  const probability = 1 / (1 + Math.exp(-(logOdds + baseValue)));
  return { probability, features: features.sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi)) };
};

// --- REUSABLE UI COMPONENTS ---
const GlassCard = ({ children, title, icon: Icon, className = "", headerAction }) => (
  <div className={`backdrop-blur-2xl bg-slate-900/50 border border-white/5 shadow-2xl rounded-[2.5rem] p-6 sm:p-8 flex flex-col transition-all hover:bg-slate-900/60 hover:border-white/10 ${className}`}>
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3 opacity-80">
        {Icon && <Icon className="w-4 h-4 text-blue-400" />}
        <h2 className="text-[10px] font-black uppercase tracking-[0.4em] text-white/90">{title}</h2>
      </div>
      {headerAction}
    </div>
    {children}
  </div>
);

const Slider = ({ label, val, min, max, unit, step = 1, onChange }) => (
  <div className="group space-y-4">
    <div className="flex justify-between items-baseline">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest group-hover:text-blue-400 transition-colors">{label}</span>
      <span className="text-sm font-black text-white">{val}<span className="text-[10px] ml-1 text-slate-500 font-medium">{unit}</span></span>
    </div>
    <div className="relative flex items-center h-4">
      <input 
        type="range" min={min} max={max} step={step} value={val} 
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-white/10 accent-blue-500 outline-none hover:bg-white/20 transition-all"
      />
    </div>
  </div>
);

// --- MAIN APPLICATION ---
export default function App() {
  const [user, setUser] = useState(null);
  const [vitals, setVitals] = useState({ hr: 82, bp: 118, o2: 98, temp: 37.0 });
  const [profile, setProfile] = useState(null);
  const [medicalRecord, setMedicalRecord] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const chatEndRef = useRef(null);
  const [intakeData, setIntakeData] = useState({ age: '', height: '', weight: '' });
  const [isAuthProcessing, setIsAuthProcessing] = useState(false);

  const xai = useMemo(() => calculateShapValues(vitals, profile || {}), [vitals, profile]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, u => {
      setUser(u);
      if (u) {
        setChatMessages([
          { role: 'assistant', content: `Welcome back, ${u.displayName || 'Doctor'}. XAI neural links are stable. How shall we begin this clinical review?` }
        ]);
      }
    });
    return () => unsubscribe();
  }, []);

  // Sync Profile from Firestore
  useEffect(() => {
    if (!db || !user) return;
    const profileRef = doc(db, 'artifacts', appId, 'users', user.uid, 'profile', 'metadata');
    const unsubscribe = onSnapshot(profileRef, (snap) => {
      if (snap.exists()) setProfile(snap.data());
    });
    return () => unsubscribe();
  }, [user]);

  useEffect(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), [chatMessages]);

  const signInWithGoogle = async () => {
    setIsAuthProcessing(true);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err) {
      console.error("Auth Error:", err);
    } finally {
      setIsAuthProcessing(false);
    }
  };

  const handleProfileSubmit = async (e) => {
    e.preventDefault();
    if (!intakeData.age || !intakeData.height || !intakeData.weight || !user) return;
    try {
      const docRef = doc(db, 'artifacts', appId, 'users', user.uid, 'profile', 'metadata');
      await setDoc(docRef, { ...intakeData, lastUpdated: serverTimestamp() });
    } catch (err) { console.error(err); }
  };

  const handleSendMessage = async (customPrompt = null) => {
    const input = customPrompt || userInput;
    if (!input.trim() || isAnalyzing) return;

    const newMessages = [...chatMessages, { role: 'user', content: input }];
    setChatMessages(newMessages);
    setUserInput("");
    setIsAnalyzing(true);

    try {
      const systemInstruction = `
        You are Dr. XAI, a Chief Medical Officer and Specialist in Explainable AI.
        You provide high-fidelity medical consultation with the accuracy and depth of a world-class physician.
        
        PATIENT PARAMETERS:
        - Age: ${profile?.age || 'Unset'}, Weight: ${profile?.weight || 'Unset'}kg, Height: ${profile?.height || 'Unset'}cm.
        - Vitals: Heart Rate ${vitals.hr} bpm, Blood Pressure ${vitals.bp} mmHg, SpO2 ${vitals.o2}%.
        - Current Risk Probability: ${(xai.probability * 100).toFixed(0)}%.
        - Primary Driver (SHAP): ${xai.features[0].name} contributing ${(xai.features[0].phi * 100).toFixed(1)}% to the variance.
        - Medical History Context: ${medicalRecord || 'None provided.'}
        
        MANDATE:
        1. Speak professionally using clinical terminology (hemodynamics, hypoxemia, tachycardia).
        2. Correlate current vitals with history and physical profile.
        3. Explain the AI's "Black Box" reasoning using the SHAP data.
        4. Suggest specific stabilization steps or further tests (ABG, ECG, Lab work).
      `;

      // Formatting for Gemini API
      const contents = newMessages.map(m => ({ 
        role: m.role === 'assistant' ? 'model' : 'user', 
        parts: [{ text: m.content }] 
      }));

      const callWithBackoff = async (attempt = 0) => {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${GEMINI_API_KEY}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            contents,
            systemInstruction: { parts: [{ text: systemInstruction }] }
          })
        });
        
        if (!response.ok && attempt < 5) {
          await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 1000));
          return callWithBackoff(attempt + 1);
        }
        return response.json();
      };

      const res = await callWithBackoff();
      const aiText = res.candidates?.[0]?.content?.parts?.[0]?.text || "Clinical analysis in progress...";
      setChatMessages([...newMessages, { role: 'assistant', content: aiText }]);
    } catch (e) {
      setChatMessages([...newMessages, { role: 'assistant', content: "XAI uplink disrupted. Check clinical credentials." }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // LOGIN PAGE
  if (!user) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-center">
        <NeuralBackground riskProbability={0.15} />
        <div className="relative z-10 space-y-8 max-w-lg">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-[2.5rem] flex items-center justify-center mx-auto shadow-2xl shadow-blue-500/40">
            <ShieldAlert className="w-10 h-10 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-black uppercase tracking-[0.5em] text-white">XAI <span className="text-blue-500">Elite</span></h1>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-4 leading-relaxed px-10">
              Advanced Clinical Decision Support & Explainable AI for Critical Care Environments.
            </p>
          </div>
          <button 
            onClick={signInWithGoogle}
            disabled={isAuthProcessing}
            className="w-full py-5 bg-white text-slate-950 text-[10px] font-black uppercase tracking-[0.2em] rounded-3xl transition-all hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-4 shadow-2xl"
          >
            {isAuthProcessing ? <Loader2 className="w-4 h-4 animate-spin"/> : <Lock className="w-4 h-4 text-blue-600"/>}
            Secure Google Login
          </button>
          <p className="text-[9px] font-bold text-slate-600 uppercase tracking-widest">Authorized Clinical Personnel Only</p>
        </div>
      </div>
    );
  }

  // INTAKE OVERLAY
  if (user && !profile) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
        <NeuralBackground riskProbability={0.2} />
        <GlassCard title="Clinical Intake: Patient Baseline" icon={UserPlus} className="max-w-md w-full relative z-10">
          <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-10 leading-relaxed">
            Specify patient metrics to calibrate the XAI risk engine for this session.
          </p>
          <form onSubmit={handleProfileSubmit} className="space-y-8">
            <div className="space-y-3">
              <label className="text-[9px] font-black uppercase tracking-tighter text-blue-500">Patient Age</label>
              <input 
                type="number" required value={intakeData.age}
                onChange={e => setIntakeData({...intakeData, age: e.target.value})}
                placeholder="Years"
                className="w-full bg-white/5 border border-white/10 rounded-2xl p-5 text-sm font-bold outline-none focus:border-blue-500 transition-all"
              />
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-3">
                <label className="text-[9px] font-black uppercase tracking-tighter text-blue-500">Height (cm)</label>
                <input 
                  type="number" required value={intakeData.height}
                  onChange={e => setIntakeData({...intakeData, height: e.target.value})}
                  className="w-full bg-white/5 border border-white/10 rounded-2xl p-5 text-sm font-bold outline-none focus:border-blue-500 transition-all"
                />
              </div>
              <div className="space-y-3">
                <label className="text-[9px] font-black uppercase tracking-tighter text-blue-500">Weight (kg)</label>
                <input 
                  type="number" required value={intakeData.weight}
                  onChange={e => setIntakeData({...intakeData, weight: e.target.value})}
                  className="w-full bg-white/5 border border-white/10 rounded-2xl p-5 text-sm font-bold outline-none focus:border-blue-500 transition-all"
                />
              </div>
            </div>
            <button type="submit" className="w-full py-5 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black uppercase tracking-widest rounded-3xl transition-all shadow-xl shadow-blue-500/30">
              Initialize XAI Engine
            </button>
          </form>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans">
      <NeuralBackground riskProbability={xai.probability} />

      {/* NAVBAR */}
      <nav className="fixed top-0 w-full z-[60] backdrop-blur-2xl border-b border-white/5 px-8 py-4 flex justify-between items-center bg-slate-950/40">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
            <ShieldAlert className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xs font-black uppercase tracking-[0.4em] leading-none">XAI <span className="text-blue-500">Elite</span></h1>
            <p className="text-[8px] font-bold text-slate-500 uppercase tracking-widest mt-1">Provider: {user.displayName}</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden lg:flex items-center gap-4 px-5 py-2.5 bg-white/5 border border-white/10 rounded-full">
             <div className="flex items-center gap-2 pr-4 border-r border-white/10">
                <Stethoscope className="w-3 h-3 text-blue-500" />
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-tighter">Clinical Mode</span>
             </div>
             <span className="text-[9px] font-black text-white uppercase tracking-tighter">Age {profile.age} / {profile.weight}kg</span>
          </div>
          <button onClick={() => signOut(auth)} className="flex items-center gap-2 p-2 px-4 hover:bg-red-500/10 rounded-2xl transition-all text-slate-500 hover:text-red-500 border border-transparent hover:border-red-500/20">
            <LogOut className="w-4 h-4" />
            <span className="text-[9px] font-black uppercase tracking-widest">Logoff</span>
          </button>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <main className="relative z-10 pt-32 pb-16 px-6 sm:px-12 max-w-[1700px] mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          
          <div className="lg:col-span-3 space-y-8">
            <GlassCard title="Clinical Record" icon={FileText}>
              <textarea 
                value={medicalRecord}
                onChange={e => setMedicalRecord(e.target.value)}
                placeholder="Paste longitudinal notes..."
                className="w-full h-40 bg-transparent border-none text-[11px] placeholder:text-slate-700 focus:ring-0 resize-none font-medium leading-relaxed custom-scrollbar"
              />
              <button 
                onClick={() => handleSendMessage("Dr. XAI, I need a diagnostic impression of this clinical history.")}
                className="mt-6 w-full py-4 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 text-[10px] font-black uppercase tracking-widest rounded-2xl transition-all border border-blue-500/20"
              >
                Ingest Record
              </button>
            </GlassCard>

            <GlassCard title="Live Telemetry" icon={Activity}>
              <div className="space-y-12 py-4">
                <Slider label="Heart Rate" val={vitals.hr} min={40} max={180} unit="BPM" onChange={v => setVitals({...vitals, hr: v})} />
                <Slider label="Blood Pressure" val={vitals.bp} min={60} max={220} unit="mmHg" onChange={v => setVitals({...vitals, bp: v})} />
                <Slider label="Oxygen Sat" val={vitals.o2} min={70} max={100} unit="%" onChange={v => setVitals({...vitals, o2: v})} />
                <Slider label="Body Temp" val={vitals.temp} min={34} max={42} step={0.1} unit="°C" onChange={v => setVitals({...vitals, temp: v})} />
              </div>
            </GlassCard>
          </div>

          <div className="lg:col-span-5 space-y-10">
            <div className="relative p-12 sm:p-20 backdrop-blur-3xl bg-white/5 border border-white/10 rounded-[4rem] flex flex-col items-center justify-center text-center group transition-all overflow-hidden shadow-2xl">
              <div className={`absolute inset-0 bg-gradient-to-b ${xai.probability > 0.7 ? 'from-rose-500/20' : 'from-blue-500/10'} to-transparent pointer-events-none opacity-40`} />
              <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.6em] mb-8">Clinical Deterioration Risk</h3>
              <div className={`text-[10rem] sm:text-[13rem] leading-none font-black tracking-tighter transition-all duration-700 ${xai.probability > 0.7 ? 'text-rose-500 drop-shadow-[0_0_80px_rgba(244,63,94,0.4)]' : 'text-blue-500 drop-shadow-[0_0_80px_rgba(59,130,246,0.3)]'}`}>
                {(xai.probability * 100).toFixed(0)}%
              </div>
              <div className="mt-14 flex gap-12 items-center">
                <div className="text-center">
                  <div className="text-3xl font-black">{vitals.hr}</div>
                  <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mt-1">BPM</div>
                </div>
                <div className="h-16 w-px bg-white/10" />
                <div className="text-center">
                  <div className="text-3xl font-black">{vitals.o2}%</div>
                  <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mt-1">SPO2</div>
                </div>
              </div>
            </div>

            <GlassCard title="XAI Decision Drivers (SHAP)" icon={BrainCircuit}>
              <div className="space-y-8">
                {xai.features.map((f, i) => (
                  <div key={i} className="space-y-3">
                    <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest">
                      <span className="text-slate-400">{f.name}</span>
                      <span className={f.color}>{f.phi > 0 ? '+' : ''}{(f.phi * 100).toFixed(1)}% Contribution</span>
                    </div>
                    <div className="h-2 bg-white/5 rounded-full overflow-hidden flex shadow-inner">
                      <div 
                        className={`h-full transition-all duration-1000 ${f.phi > 0 ? 'bg-rose-500 ml-[50%] shadow-[0_0_20px_rgba(244,63,94,0.5)]' : 'bg-cyan-400 ml-auto mr-[50%] shadow-[0_0_20px_rgba(34,211,238,0.5)]'}`}
                        style={{ width: `${Math.abs(f.phi) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>

          <div className="lg:col-span-4 h-full flex flex-col">
            <GlassCard 
              title="Dr. XAI: Elite Rounds" 
              icon={Bot} 
              className="flex-1 min-h-[600px] sm:min-h-[800px]"
              headerAction={
                <button onClick={() => setChatMessages([])} className="text-[8px] font-black text-slate-600 hover:text-white transition-colors uppercase tracking-widest">Clear Log</button>
              }
            >
              <div className="flex-1 overflow-y-auto pr-2 space-y-6 custom-scrollbar mb-8 max-h-[500px] sm:max-h-[700px]">
                {chatMessages.map((m, i) => (
                  <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[90%] p-6 rounded-[2rem] text-[11px] leading-relaxed ${m.role === 'user' ? 'bg-blue-600 text-white shadow-xl shadow-blue-500/20 font-medium' : 'bg-white/5 text-slate-300 border border-white/5 font-medium shadow-inner'}`}>
                      {m.content.split('\n').map((line, j) => <p key={j} className="mb-2 last:mb-0">{line}</p>)}
                    </div>
                  </div>
                ))}
                {isAnalyzing && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-3 px-6 py-4 bg-white/5 rounded-3xl border border-white/5 shadow-lg">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                      <span className="text-[10px] font-black uppercase text-slate-500 tracking-[0.2em]">Consulting XAI Brain...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <div className="mt-auto relative">
                <input 
                  type="text"
                  value={userInput}
                  onChange={e => setUserInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask Dr. XAI for a diagnosis..."
                  className="w-full bg-white/5 border border-white/10 rounded-2xl p-6 pr-16 text-xs outline-none focus:border-blue-500 transition-all font-medium placeholder:text-slate-700"
                />
                <button 
                  onClick={() => handleSendMessage()} 
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-blue-600 rounded-2xl hover:bg-blue-500 transition-all active:scale-95 shadow-xl shadow-blue-500/40"
                >
                  <Send className="w-5 h-5 text-white" />
                </button>
              </div>
            </GlassCard>
          </div>
        </div>
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
        
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          box-shadow: 0 0 20px rgba(59,130,246,0.7);
          transition: all 0.2s;
          border: 2px solid white;
        }
        input[type=range]::-webkit-slider-thumb:hover { 
          transform: scale(1.2); 
          box-shadow: 0 0 30px rgba(59,130,246,0.9);
        }
      `}</style>
    </div>
  );
}

const Bot = ({ className }) => <Stethoscope className={className} />;

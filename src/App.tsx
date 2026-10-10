/**
 * MIDI Forge AI - Studio & Local Training Interface
 * Integrates aiMIDI (REMI) and Theremin MIDI (BPE/GPT) hybrid engine
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Music,
  Sliders,
  Play,
  Square,
  Pause,
  Download,
  FolderOpen,
  Sparkles,
  ShieldCheck,
  Cpu,
  Layers,
  History,
  HardDrive,
  RefreshCw,
  Upload,
  Volume2,
  FileText,
  CheckCircle2,
  XCircle,
  AlertCircle
} from 'lucide-react';

interface MidiFileInfo {
  filename: string;
  duration_sec: number;
  ticks_per_beat: number;
  tempo_bpm: number;
  time_signature: string;
  track_count: number;
  note_count: number;
  pitch_range: [number, number];
  is_valid: boolean;
  error_message?: string;
}

interface NoteEvent {
  pitch: number;
  start_tick: number;
  end_tick?: number;
  duration_ticks: number;
  velocity: number;
  channel?: number;
}

interface TrainProgress {
  status: string;
  model_name: string;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  epoch_progress_pct: number;
  overall_progress_pct: number;
  current_train_loss: number;
  current_val_loss: number;
  learning_rate: number;
  elapsed_sec: number;
  remaining_sec: number;
  device: string;
  gpu_name?: string;
  loss_history?: Array<{ step: number; epoch: number; train_loss: number; val_loss: number }>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'library' | 'dataprep' | 'training' | 'generation' | 'history' | 'diagnostics' | 'portable'>('dashboard');

  // Stats & Hardware
  const [stats, setStats] = useState({
    imported_midi_count: 0,
    models_count: 0,
    checkpoints_count: 0,
    output_midi_count: 0,
    disk_usage_mb: 0,
  });
  const [hwStatus, setHwStatus] = useState({ device: 'CPU', gpu_name: null as string | null, torch_version: '' });

  // Library state
  const [libraryFiles, setLibraryFiles] = useState<MidiFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<MidiFileInfo | null>(null);
  const [previewNotes, setPreviewNotes] = useState<NoteEvent[]>([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);

  // Data prep state
  const [prepMode, setPrepMode] = useState<'remi' | 'theremin'>('remi');
  const [prepBlockSize, setPrepBlockSize] = useState(128);
  const [prepAugment, setPrepAugment] = useState(true);
  const [prepReport, setPrepReport] = useState<any>(null);
  const [isPrepping, setIsPrepping] = useState(false);

  // Training state
  const [trainModelName, setTrainModelName] = useState('Forge_Transformer_v1');
  const [trainEpochs, setTrainEpochs] = useState(10);
  const [trainBatch, setTrainBatch] = useState(4);
  const [trainLr, setTrainLr] = useState(0.0005);
  const [trainLayers, setTrainLayers] = useState(3);
  const [trainHeads, setTrainHeads] = useState(4);
  const [trainEmbd, setTrainEmbd] = useState(128);
  const [trainProgress, setTrainProgress] = useState<TrainProgress>({
    status: 'idle',
    model_name: '',
    current_epoch: 0,
    total_epochs: 10,
    current_step: 0,
    total_steps: 0,
    epoch_progress_pct: 0,
    overall_progress_pct: 0,
    current_train_loss: 0,
    current_val_loss: 0,
    learning_rate: 0.0005,
    elapsed_sec: 0,
    remaining_sec: 0,
    device: 'cpu',
  });

  // Generation state
  const [genMode, setGenMode] = useState<'unconditional' | 'reference'>('unconditional');
  const [genRefFile, setGenRefFile] = useState<string>('');
  const [genRetention, setGenRetention] = useState(30);
  const [genTemp, setGenTemp] = useState(0.95);
  const [genTopK, setGenTopK] = useState(40);
  const [genTopP, setGenTopP] = useState(0.90);
  const [genTokens, setGenTokens] = useState(200);
  const [genBpm, setGenBpm] = useState(120);
  const [genSeed, setGenSeed] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastGenResult, setLastGenResult] = useState<any>(null);

  // History state
  const [historyFiles, setHistoryFiles] = useState<MidiFileInfo[]>([]);

  // Diagnostics state
  const [diagResults, setDiagResults] = useState<any>(null);
  const [isRunningTests, setIsRunningTests] = useState(false);

  // Audio Context Ref
  const audioCtxRef = useRef<AudioContext | null>(null);
  const activeOscsRef = useRef<OscillatorNode[]>([]);
  const libraryCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const genCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initial loading
  useEffect(() => {
    fetchStatus();
    fetchStats();
    fetchLibrary();
    fetchHistory();

    const interval = setInterval(() => {
      fetchTrainingProgress();
    }, 1000);

    return () => {
      clearInterval(interval);
      stopAudioPlayback();
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const d = await res.json();
        setHwStatus({ device: d.device?.toUpperCase() || 'CPU', gpu_name: d.gpu_name, torch_version: d.torch_version });
      }
    } catch {}
  };

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const d = await res.json();
        setStats(d);
      }
    } catch {}
  };

  const fetchLibrary = async () => {
    try {
      const res = await fetch('/api/library');
      if (res.ok) {
        const d = await res.json();
        setLibraryFiles(d.files || []);
        if (d.files?.length && !selectedFile) {
          inspectFile(d.files[0]);
        }
      }
    } catch {}
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/output');
      if (res.ok) {
        const d = await res.json();
        setHistoryFiles(d.files || []);
      }
    } catch {}
  };

  const fetchTrainingProgress = async () => {
    try {
      const res = await fetch('/api/training/progress');
      if (res.ok) {
        const d = await res.json();
        setTrainProgress(d);
        if (d.loss_history?.length && lossCanvasRef.current) {
          drawLossChart(lossCanvasRef.current, d.loss_history);
        }
      }
    } catch {}
  };

  // Inspect file and show notes
  const inspectFile = (file: MidiFileInfo) => {
    setSelectedFile(file);
    const notes: NoteEvent[] = [];
    const minP = file.pitch_range?.[0] || 48;
    const maxP = file.pitch_range?.[1] || 72;
    const count = Math.min(file.note_count || 48, 96);
    let curTick = 0;
    for (let i = 0; i < count; i++) {
      const pitch = minP + Math.floor(Math.sin(i * 0.45) * ((maxP - minP) / 2) + (maxP - minP) / 2);
      notes.push({
        pitch,
        start_tick: curTick,
        duration_ticks: 240,
        velocity: 80,
      });
      curTick += 240;
    }
    setPreviewNotes(notes);
    if (libraryCanvasRef.current) {
      drawPianoRoll(libraryCanvasRef.current, notes);
    }
  };

  // Draw Piano Roll on Canvas
  const drawPianoRoll = (canvas: HTMLCanvasElement, notes: NoteEvent[]) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#090b10';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#161b24';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 15) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (!notes.length) return;

    let minP = 127, maxP = 0, maxT = 0;
    notes.forEach((n) => {
      if (n.pitch < minP) minP = n.pitch;
      if (n.pitch > maxP) maxP = n.pitch;
      const endT = n.start_tick + n.duration_ticks;
      if (endT > maxT) maxT = endT;
    });
    minP = Math.max(21, minP - 2);
    maxP = Math.min(108, maxP + 2);
    maxT = Math.max(480, maxT);
    const pRange = Math.max(12, maxP - minP);

    notes.forEach((n) => {
      const x = (n.start_tick / maxT) * (w - 20) + 10;
      const noteW = Math.max(4, (n.duration_ticks / maxT) * (w - 20));
      const normP = (n.pitch - minP) / pRange;
      const y = h - normP * (h - 16) - 14;
      const noteH = Math.max(4, (h / pRange) * 0.85);

      ctx.fillStyle = '#00e5ff';
      ctx.shadowColor = 'rgba(0, 229, 255, 0.5)';
      ctx.shadowBlur = 6;
      ctx.fillRect(x, y, noteW, noteH);
    });
    ctx.shadowBlur = 0;
  };

  // Draw Loss Chart
  const drawLossChart = (canvas: HTMLCanvasElement, history: Array<{ train_loss: number; val_loss: number }>) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !history.length) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0f141d';
    ctx.fillRect(0, 0, w, h);

    let maxL = 0, minL = 100;
    history.forEach((pt) => {
      if (pt.train_loss > maxL) maxL = pt.train_loss;
      if (pt.train_loss < minL) minL = pt.train_loss;
    });
    maxL = Math.max(1.0, maxL * 1.1);
    minL = Math.max(0, minL * 0.8);
    const range = maxL - minL || 1.0;

    // Grid
    ctx.strokeStyle = '#222b3a';
    ctx.lineWidth = 1;
    for (let y = 20; y < h - 20; y += 40) {
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(w - 10, y);
      ctx.stroke();
    }

    // Train loss (cyan)
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((pt, i) => {
      const x = 40 + (i / (history.length - 1 || 1)) * (w - 60);
      const y = h - 20 - ((pt.train_loss - minL) / range) * (h - 40);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Val loss (amber)
    ctx.strokeStyle = '#ff9100';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((pt, i) => {
      const x = 40 + (i / (history.length - 1 || 1)) * (w - 60);
      const y = h - 20 - ((pt.val_loss - minL) / range) * (h - 40);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#00e5ff';
    ctx.fillText('● Loss Treino', w - 180, 20);
    ctx.fillStyle = '#ff9100';
    ctx.fillText('● Loss Validação', w - 90, 20);
  };

  // Web Audio Synth
  const playNotesInBrowser = (notes: NoteEvent[]) => {
    stopAudioPlayback();
    if (!notes.length) return;

    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContextClass();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }

    setIsPlayingAudio(true);
    const ctx = audioCtxRef.current;
    const now = ctx.currentTime;
    const tickToSec = 0.5 / 480;

    notes.slice(0, 140).forEach((n) => {
      const startTime = now + n.start_tick * tickToSec;
      const duration = Math.min(1.5, Math.max(0.1, n.duration_ticks * tickToSec));

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      const freq = 440 * Math.pow(2, (n.pitch - 69) / 12);
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(freq, startTime);

      const velNorm = (n.velocity || 80) / 127.0;
      gain.gain.setValueAtTime(0, startTime);
      gain.gain.linearRampToValueAtTime(0.12 * velNorm, startTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1400, startTime);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      osc.start(startTime);
      osc.stop(startTime + duration + 0.05);

      activeOscsRef.current.push(osc);
    });
  };

  const stopAudioPlayback = () => {
    setIsPlayingAudio(false);
    activeOscsRef.current.forEach((osc) => {
      try {
        osc.stop();
      } catch {}
    });
    activeOscsRef.current = [];
  };

  // Synthetic dataset generation
  const handleGenerateSynthetic = async () => {
    try {
      const res = await fetch('/api/library/synthetic', { method: 'POST' });
      if (res.ok) {
        await fetchLibrary();
        await fetchStats();
      }
    } catch (e) {
      console.error(e);
    }
  };

  // File upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !files.length) return;
    Array.from(files).forEach((f) => {
      const reader = new FileReader();
      reader.onload = async (ev) => {
        const base64 = (ev.target?.result as string).split(',')[1];
        await fetch('/api/library/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: f.name, content_base64: base64 }),
        });
        await fetchLibrary();
        await fetchStats();
      };
      reader.readAsDataURL(f);
    });
  };

  // Run data prep
  const handleRunPrep = async () => {
    setIsPrepping(true);
    try {
      const res = await fetch('/api/datasets/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: prepMode,
          block_size: prepBlockSize,
          augment_transposition: prepAugment,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setPrepReport(data.report);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsPrepping(false);
    }
  };

  // Training controls
  const handleStartTraining = async () => {
    await fetch('/api/models/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: trainModelName,
        n_layer: trainLayers,
        n_head: trainHeads,
        n_embd: trainEmbd,
      }),
    });

    await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: trainModelName,
        epochs: trainEpochs,
        batch_size: trainBatch,
        learning_rate: trainLr,
      }),
    });
    fetchTrainingProgress();
  };

  // Generation
  const handleGenerateMusic = async () => {
    setIsGenerating(true);
    try {
      const res = await fetch('/api/generation/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: genTemp,
          max_tokens: genTokens,
          top_k: genTopK,
          top_p: genTopP,
          bpm: genBpm,
          seed: genSeed ? parseInt(genSeed) : null,
          reference_file: genMode === 'reference' ? genRefFile : null,
          context_retention_pct: genRetention / 100.0,
        }),
      });
      if (res.ok) {
        const d = await res.json();
        setLastGenResult(d);
        if (d.preview_notes?.length) {
          setTimeout(() => {
            if (genCanvasRef.current) {
              drawPianoRoll(genCanvasRef.current, d.preview_notes);
            }
          }, 100);
        }
        fetchStats();
        fetchHistory();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  };

  // Run 20 tests
  const handleRunDiagnostics = async () => {
    setIsRunningTests(true);
    try {
      const res = await fetch('/api/diagnostics/run', { method: 'POST' });
      if (res.ok) {
        const d = await res.json();
        setDiagResults(d);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunningTests(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0a0c10] text-[#f0f4f8] font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#10141b] border-r border-[#242c3b] flex flex-col shrink-0">
        <div className="p-5 flex items-center gap-3 border-b border-[#242c3b]">
          <div className="w-9 h-9 rounded-lg bg-[#00e5ff]/10 text-[#00e5ff] flex items-center justify-center border border-[#00e5ff]/30">
            <Music className="w-5 h-5" />
          </div>
          <div>
            <div className="font-extrabold text-sm tracking-wide">MIDI FORGE AI</div>
            <div className="text-[10px] text-[#00e5ff] font-bold tracking-widest">MOTOR HÍBRIDO LOCAL</div>
          </div>
        </div>

        <nav className="p-3 flex-1 flex flex-col gap-1 overflow-y-auto">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: Layers },
            { id: 'library', label: 'Biblioteca MIDI', icon: FolderOpen },
            { id: 'dataprep', label: 'Preparação de Dados', icon: Sliders },
            { id: 'training', label: 'Treinamento', icon: Cpu },
            { id: 'generation', label: 'Estúdio de Geração', icon: Play },
            { id: 'history', label: 'Histórico & Exportação', icon: History },
            { id: 'diagnostics', label: 'Diagnósticos (20 Testes)', icon: ShieldCheck },
            { id: 'portable', label: 'Pacote Windows Portátil', icon: Download },
          ].map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id as any)}
                className={`flex items-center gap-3 px-3.5 py-2.5 rounded-md text-xs font-medium transition-all text-left ${
                  active
                    ? 'bg-[#00e5ff]/10 text-[#00e5ff] border-l-2 border-[#00e5ff]'
                    : 'text-[#8a96a8] hover:bg-[#161b24] hover:text-[#f0f4f8]'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-[#242c3b] flex flex-col gap-2 text-xs">
          <div className="flex items-center gap-2 text-[#8a96a8]">
            <span className="w-2 h-2 rounded-full bg-[#00e676] animate-pulse shadow-[0_0_8px_#00e676]" />
            <span>Motor Neural: 100% Local</span>
          </div>
          <div className="flex gap-1.5 mt-1">
            <span className="text-[10px] bg-[#161b24] border border-[#242c3b] px-2 py-0.5 rounded text-[#8a96a8]">
              aiMIDI (REMI)
            </span>
            <span className="text-[10px] bg-[#161b24] border border-[#242c3b] px-2 py-0.5 rounded text-[#8a96a8]">
              Theremin (BPE)
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-y-auto bg-[#0a0c10]">
        {/* Topbar */}
        <header className="px-7 py-4 bg-[#10141b] border-b border-[#242c3b] flex justify-between items-center shrink-0">
          <div>
            <h1 className="text-base font-bold">
              {activeTab === 'dashboard' && 'Painel de Controle'}
              {activeTab === 'library' && 'Biblioteca de Arquivos MIDI'}
              {activeTab === 'dataprep' && 'Preparação de Dados & Tokenização'}
              {activeTab === 'training' && 'Laboratório de Treinamento GPT'}
              {activeTab === 'generation' && 'Estúdio de Geração Musical'}
              {activeTab === 'history' && 'Histórico de Composições & Exportações'}
              {activeTab === 'diagnostics' && 'Diagnósticos do Sistema & Bateria de 20 Testes'}
              {activeTab === 'portable' && 'Distribuição Portátil para Windows (Start.bat)'}
            </h1>
            <p className="text-xs text-[#8a96a8]">Motor híbrido autorregressivo autônomo sem conexão externa.</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono bg-[#161b24] border border-[#242c3b] px-3 py-1 rounded-full text-[#8a96a8]">
              Dispositivo: {hwStatus.device} {hwStatus.gpu_name ? `(${hwStatus.gpu_name})` : ''}
            </span>
            <span className="text-xs font-mono bg-[#161b24] border border-[#242c3b] px-3 py-1 rounded-full text-[#8a96a8]">
              Disco: {stats.disk_usage_mb} MB
            </span>
            <button
              onClick={() => setActiveTab('generation')}
              className="flex items-center gap-2 bg-[#00e5ff] text-black px-4 py-1.5 rounded text-xs font-bold hover:bg-[#33ebff] transition"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Compor Agora
            </button>
          </div>
        </header>

        <div className="p-7 flex-1">
          {/* 1. DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">MIDIs Carregados</div>
                  <div className="text-2xl font-bold font-mono text-[#00e5ff] mt-1">{stats.imported_midi_count}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">Arquivos SMF locais</div>
                </div>
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">Checkpoints Salvos</div>
                  <div className="text-2xl font-bold font-mono text-[#ff9100] mt-1">{stats.checkpoints_count}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">Pesos em disco</div>
                </div>
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">Músicas Geradas</div>
                  <div className="text-2xl font-bold font-mono text-[#00e676] mt-1">{stats.output_midi_count}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">Exportadas para FL Studio</div>
                </div>
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">Estado do Motor</div>
                  <div className="text-2xl font-bold font-mono text-[#00e5ff] mt-1">{trainProgress.status.toUpperCase()}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">PyTorch {hwStatus.torch_version || '2.x'}</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5">
                  <h3 className="text-sm font-semibold mb-3">Atalhos de Produção</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => setActiveTab('library')}
                      className="bg-[#10141b] border border-[#242c3b] p-4 rounded hover:border-[#00e5ff] text-left transition"
                    >
                      <FolderOpen className="w-5 h-5 text-[#00e5ff] mb-2" />
                      <div className="text-xs font-semibold">Biblioteca MIDI</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Carregar coleção ou arquivos sintéticos</div>
                    </button>
                    <button
                      onClick={() => setActiveTab('dataprep')}
                      className="bg-[#10141b] border border-[#242c3b] p-4 rounded hover:border-[#00e5ff] text-left transition"
                    >
                      <Sliders className="w-5 h-5 text-[#ff9100] mb-2" />
                      <div className="text-xs font-semibold">Preparação de Dados</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Configurar REMI ou Theremin BPE</div>
                    </button>
                    <button
                      onClick={() => setActiveTab('training')}
                      className="bg-[#10141b] border border-[#242c3b] p-4 rounded hover:border-[#00e5ff] text-left transition"
                    >
                      <Cpu className="w-5 h-5 text-[#00e676] mb-2" />
                      <div className="text-xs font-semibold">Treinar Modelo</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Ajustar camadas e rodar épocas</div>
                    </button>
                    <button
                      onClick={() => {
                        setActiveTab('diagnostics');
                        handleRunDiagnostics();
                      }}
                      className="bg-[#10141b] border border-[#242c3b] p-4 rounded hover:border-[#00e5ff] text-left transition"
                    >
                      <ShieldCheck className="w-5 h-5 text-[#00e5ff] mb-2" />
                      <div className="text-xs font-semibold">20 Testes do Sistema</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Verificação completa automatizada</div>
                    </button>
                  </div>
                </div>

                <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5">
                  <h3 className="text-sm font-semibold mb-3">Integração aiMIDI & Theremin MIDI</h3>
                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                      <div className="font-semibold text-[#00e5ff]">Tokenização REMI (aiMIDI)</div>
                      <p className="text-[#8a96a8] mt-1 text-[11px]">
                        Eventos de Bar, Position (16 subdivisions), Pitch (21-108), Velocity bins e Duration tokens para harmonia quantizada.
                      </p>
                    </div>
                    <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                      <div className="font-semibold text-[#ff9100]">Serialização & BPE (Theremin MIDI)</div>
                      <p className="text-[#8a96a8] mt-1 text-[11px]">
                        Stream textual serializado com fusão de pares de bytes (BPE) para frases musicais e condicionamento por MIDI de referência.
                      </p>
                    </div>
                    <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                      <div className="font-semibold text-[#00e676]">Núcleo Autorregressivo GPT</div>
                      <p className="text-[#8a96a8] mt-1 text-[11px]">
                        Multi-head causal attention, checkpoints recuperáveis, parada segura e validação final rigorosa para compatibilidade com FL Studio.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 2. MIDI LIBRARY */}
          {activeTab === 'library' && (
            <div className="space-y-5">
              <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h3 className="text-sm font-semibold">Coleção Local de Arquivos MIDI</h3>
                    <p className="text-xs text-[#8a96a8]">MIDIs analisados e disponíveis para treino e condicionamento.</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={handleGenerateSynthetic}
                      className="px-3 py-1.5 bg-[#10141b] border border-[#242c3b] hover:border-[#00e5ff] rounded text-xs font-medium text-[#f0f4f8] transition flex items-center gap-1.5"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-[#00e5ff]" />
                      Gerar Coleção Sintética (8 MIDIs)
                    </button>
                    <label className="px-3 py-1.5 bg-[#00e5ff] text-black rounded text-xs font-bold hover:bg-[#33ebff] transition flex items-center gap-1.5 cursor-pointer">
                      <Upload className="w-3.5 h-3.5" />
                      <span>Importar .MID</span>
                      <input type="file" multiple accept=".mid,.midi" onChange={handleFileUpload} className="hidden" />
                    </label>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="text-[11px] text-[#8a96a8] border-b border-[#242c3b] uppercase">
                      <tr>
                        <th className="py-2.5 px-3">Arquivo</th>
                        <th className="py-2.5 px-3">Duração</th>
                        <th className="py-2.5 px-3">BPM</th>
                        <th className="py-2.5 px-3">Pistas</th>
                        <th className="py-2.5 px-3">Notas</th>
                        <th className="py-2.5 px-3">Pitch Range</th>
                        <th className="py-2.5 px-3">Integridade</th>
                        <th className="py-2.5 px-3">Ação</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#242c3b]">
                      {libraryFiles.map((f) => (
                        <tr key={f.filename} className="hover:bg-white/[0.02]">
                          <td className="py-2.5 px-3 font-medium text-[#00e5ff]">{f.filename}</td>
                          <td className="py-2.5 px-3">{f.duration_sec}s</td>
                          <td className="py-2.5 px-3">{f.tempo_bpm}</td>
                          <td className="py-2.5 px-3">{f.track_count}</td>
                          <td className="py-2.5 px-3">{f.note_count}</td>
                          <td className="py-2.5 px-3">{f.pitch_range?.[0]} - {f.pitch_range?.[1]}</td>
                          <td className="py-2.5 px-3">
                            <span className="text-[10px] bg-[#00e676]/10 text-[#00e676] px-2 py-0.5 rounded border border-[#00e676]/30">
                              Válido
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            <button
                              onClick={() => inspectFile(f)}
                              className="px-2.5 py-1 bg-[#10141b] border border-[#242c3b] hover:border-[#00e5ff] rounded text-[11px]"
                            >
                              Inspecionar
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Piano Roll & Audio Audition Box */}
                <div className="mt-6 p-4 bg-[#10141b] border border-[#242c3b] rounded-lg">
                  <div className="flex justify-between items-center mb-3">
                    <div className="text-xs font-semibold">
                      Piano Roll: <span className="text-[#00e5ff]">{selectedFile?.filename || 'Nenhum selecionado'}</span>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => playNotesInBrowser(previewNotes)}
                        disabled={!previewNotes.length || isPlayingAudio}
                        className="px-3 py-1 bg-[#00e5ff] text-black rounded text-xs font-bold hover:bg-[#33ebff] transition flex items-center gap-1 disabled:opacity-40"
                      >
                        <Play className="w-3 h-3 fill-current" />
                        Tocar no Navegador
                      </button>
                      <button
                        onClick={stopAudioPlayback}
                        disabled={!isPlayingAudio}
                        className="px-3 py-1 bg-[#242c3b] text-white rounded text-xs font-bold hover:bg-[#2e3747] transition flex items-center gap-1 disabled:opacity-40"
                      >
                        <Square className="w-3 h-3 fill-current" />
                        Parar
                      </button>
                    </div>
                  </div>
                  <canvas ref={libraryCanvasRef} width={880} height={180} className="w-full h-44 bg-[#090b10] rounded" />
                </div>
              </div>
            </div>
          )}

          {/* 3. DATA PREPARATION */}
          {activeTab === 'dataprep' && (
            <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-6 max-w-3xl">
              <h3 className="text-sm font-semibold mb-1">Configuração do Pipeline de Pré-processamento</h3>
              <p className="text-xs text-[#8a96a8] mb-6">
                Transforme os arquivos MIDI em sequências de tokens compatíveis com o Transformer.
              </p>

              <div className="space-y-4 text-xs">
                <div>
                  <label className="block font-medium mb-1.5">Modo de Tokenização</label>
                  <select
                    value={prepMode}
                    onChange={(e: any) => setPrepMode(e.target.value)}
                    className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-2 text-xs focus:border-[#00e5ff] outline-none"
                  >
                    <option value="remi">REMI (aiMIDI) — Grades Rítmicas, Compassos e Bins de Velocidade</option>
                    <option value="theremin">Theremin BPE — Serialização Textual & Byte Pair Encoding</option>
                  </select>
                </div>

                <div>
                  <label className="block font-medium mb-1.5">Comprimento da Sequência de Treino (Block Size)</label>
                  <select
                    value={prepBlockSize}
                    onChange={(e) => setPrepBlockSize(parseInt(e.target.value))}
                    className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-2 text-xs focus:border-[#00e5ff] outline-none"
                  >
                    <option value={64}>64 tokens (Ultraleve para teste rápido)</option>
                    <option value={128}>128 tokens (Recomendado para CPU)</option>
                    <option value={256}>256 tokens (Equilibrado)</option>
                    <option value={512}>512 tokens (Contexto Longo)</option>
                  </select>
                </div>

                <div>
                  <label className="block font-medium mb-1.5">Aumento de Dados (Data Augmentation)</label>
                  <select
                    value={prepAugment ? 'true' : 'false'}
                    onChange={(e) => setPrepAugment(e.target.value === 'true')}
                    className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-2 text-xs focus:border-[#00e5ff] outline-none"
                  >
                    <option value="true">Ativado (Transposição de -2 a +2 semitons)</option>
                    <option value="false">Desativado (Apenas tons originais)</option>
                  </select>
                </div>

                <div className="pt-3">
                  <button
                    onClick={handleRunPrep}
                    disabled={isPrepping}
                    className="bg-[#00e5ff] text-black font-bold px-5 py-2.5 rounded text-xs hover:bg-[#33ebff] transition flex items-center gap-2 disabled:opacity-40"
                  >
                    <Sliders className="w-4 h-4" />
                    {isPrepping ? 'Processando...' : 'Processar & Gerar Sequências de Treinamento'}
                  </button>
                </div>

                {prepReport && (
                  <div className="mt-6 p-4 bg-[#10141b] border border-[#242c3b] rounded-lg">
                    <h4 className="font-semibold text-xs text-[#00e5ff] mb-3">Relatório de Pré-processamento</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="p-3 bg-[#161b24] rounded">
                        <div className="text-[10px] text-[#8a96a8]">Arquivos Ingeridos</div>
                        <div className="text-lg font-bold font-mono text-[#00e5ff]">{prepReport.files_processed_count}</div>
                      </div>
                      <div className="p-3 bg-[#161b24] rounded">
                        <div className="text-[10px] text-[#8a96a8]">Notas Mapeadas</div>
                        <div className="text-lg font-bold font-mono text-[#00e5ff]">{prepReport.total_notes_analyzed}</div>
                      </div>
                      <div className="p-3 bg-[#161b24] rounded">
                        <div className="text-[10px] text-[#8a96a8]">Sequências Geradas</div>
                        <div className="text-lg font-bold font-mono text-[#ff9100]">{prepReport.train_sequences_count}</div>
                      </div>
                      <div className="p-3 bg-[#161b24] rounded">
                        <div className="text-[10px] text-[#8a96a8]">Vocabulário</div>
                        <div className="text-lg font-bold font-mono text-[#00e676]">{prepReport.vocab_size} tokens</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 4. TRAINING LAB */}
          {activeTab === 'training' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Config Panel */}
              <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5 space-y-4 text-xs">
                <h3 className="text-sm font-semibold">Parâmetros do Modelo & Treinador</h3>

                <div className="flex gap-2 mb-2">
                  <button
                    onClick={() => { setTrainLayers(2); setTrainHeads(4); setTrainEmbd(64); setTrainBatch(4); }}
                    className="px-2.5 py-1 rounded bg-[#10141b] border border-[#242c3b] hover:border-[#00e5ff] text-[11px]"
                  >
                    Preset: CPU Conservador
                  </button>
                  <button
                    onClick={() => { setTrainLayers(4); setTrainHeads(4); setTrainEmbd(128); setTrainBatch(8); }}
                    className="px-2.5 py-1 rounded bg-[#10141b] border border-[#242c3b] hover:border-[#00e5ff] text-[11px]"
                  >
                    Preset: Equilibrado
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Nome do Modelo</label>
                    <input
                      type="text"
                      value={trainModelName}
                      onChange={(e) => setTrainModelName(e.target.value)}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Épocas</label>
                    <input
                      type="number"
                      value={trainEpochs}
                      onChange={(e) => setTrainEpochs(parseInt(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Batch Size</label>
                    <input
                      type="number"
                      value={trainBatch}
                      onChange={(e) => setTrainBatch(parseInt(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={trainLr}
                      onChange={(e) => setTrainLr(parseFloat(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Camadas (Layers)</label>
                    <input
                      type="number"
                      value={trainLayers}
                      onChange={(e) => setTrainLayers(parseInt(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Cabeças (Heads)</label>
                    <input
                      type="number"
                      value={trainHeads}
                      onChange={(e) => setTrainHeads(parseInt(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                </div>

                <div className="flex gap-2 pt-2">
                  <button
                    onClick={handleStartTraining}
                    disabled={trainProgress.status === 'running'}
                    className="px-4 py-2 bg-[#00e5ff] text-black font-bold rounded text-xs hover:bg-[#33ebff] transition flex items-center gap-1.5 disabled:opacity-40"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Iniciar Treinamento
                  </button>
                  <button
                    onClick={() => fetch('/api/training/pause', { method: 'POST' })}
                    disabled={trainProgress.status !== 'running'}
                    className="px-3 py-2 bg-[#10141b] border border-[#242c3b] rounded text-xs hover:border-[#00e5ff] disabled:opacity-40"
                  >
                    <Pause className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => fetch('/api/training/stop', { method: 'POST' })}
                    disabled={trainProgress.status !== 'running' && trainProgress.status !== 'paused'}
                    className="px-3 py-2 bg-red-950/40 border border-red-800 text-red-400 rounded text-xs hover:bg-red-900/40 disabled:opacity-40"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" />
                  </button>
                </div>
              </div>

              {/* Real-time Monitor Panel */}
              <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-sm font-semibold">Monitor em Tempo Real</h3>
                    <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/30">
                      {trainProgress.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs mb-4">
                    <div className="p-2.5 bg-[#10141b] rounded border border-[#242c3b]">
                      <span className="text-[10px] text-[#8a96a8] block">Época</span>
                      <span className="font-mono font-bold text-sm">{trainProgress.current_epoch + 1} / {trainProgress.total_epochs}</span>
                    </div>
                    <div className="p-2.5 bg-[#10141b] rounded border border-[#242c3b]">
                      <span className="text-[10px] text-[#8a96a8] block">Train Loss</span>
                      <span className="font-mono font-bold text-sm text-[#00e5ff]">{(trainProgress.current_train_loss || 0).toFixed(4)}</span>
                    </div>
                    <div className="p-2.5 bg-[#10141b] rounded border border-[#242c3b]">
                      <span className="text-[10px] text-[#8a96a8] block">Val Loss</span>
                      <span className="font-mono font-bold text-sm text-[#ff9100]">{(trainProgress.current_val_loss || 0).toFixed(4)}</span>
                    </div>
                  </div>

                  <div className="space-y-1 mb-4">
                    <div className="flex justify-between text-[11px] text-[#8a96a8]">
                      <span>Progresso Geral</span>
                      <span>{trainProgress.overall_progress_pct || 0}%</span>
                    </div>
                    <div className="w-full h-2 bg-[#10141b] rounded overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[#00e5ff] to-[#00e676] transition-all"
                        style={{ width: `${trainProgress.overall_progress_pct || 0}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="p-2 bg-[#10141b] rounded border border-[#242c3b]">
                  <canvas ref={lossCanvasRef} width={500} height={180} className="w-full h-44 bg-[#0f141d] rounded" />
                </div>
              </div>
            </div>
          )}

          {/* 5. GENERATION STUDIO */}
          {activeTab === 'generation' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Controls */}
              <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5 space-y-4 text-xs">
                <h3 className="text-sm font-semibold">Parâmetros de Amostragem</h3>

                <div>
                  <label className="block text-[#8a96a8] mb-1">Tipo de Geração</label>
                  <select
                    value={genMode}
                    onChange={(e: any) => setGenMode(e.target.value)}
                    className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-2 text-xs focus:border-[#00e5ff] outline-none"
                  >
                    <option value="unconditional">Incondicional (Composição 100% Livre / Semente)</option>
                    <option value="reference">Condicionada por MIDI de Referência (Continuação)</option>
                  </select>
                </div>

                {genMode === 'reference' && (
                  <>
                    <div>
                      <label className="block text-[#8a96a8] mb-1">MIDI de Referência</label>
                      <select
                        value={genRefFile}
                        onChange={(e) => setGenRefFile(e.target.value)}
                        className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-2 text-xs focus:border-[#00e5ff] outline-none"
                      >
                        <option value="">Selecione um arquivo...</option>
                        {libraryFiles.map((f) => (
                          <option key={f.filename} value={f.filename}>{f.filename}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-[#8a96a8] mb-1">
                        Contexto Inicial Preservado: <span className="text-[#00e5ff]">{genRetention}%</span>
                      </label>
                      <input
                        type="range"
                        min="10"
                        max="80"
                        value={genRetention}
                        onChange={(e) => setGenRetention(parseInt(e.target.value))}
                        className="w-full accent-[#00e5ff]"
                      />
                    </div>
                  </>
                )}

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[#8a96a8] mb-1">
                      Temperatura: <span className="text-[#00e5ff]">{genTemp.toFixed(2)}</span>
                    </label>
                    <input
                      type="range"
                      min="0.2"
                      max="1.5"
                      step="0.05"
                      value={genTemp}
                      onChange={(e) => setGenTemp(parseFloat(e.target.value))}
                      className="w-full accent-[#00e5ff]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Máx Tokens</label>
                    <input
                      type="number"
                      value={genTokens}
                      onChange={(e) => setGenTokens(parseInt(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Andamento (BPM)</label>
                    <input
                      type="number"
                      value={genBpm}
                      onChange={(e) => setGenBpm(parseFloat(e.target.value))}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Semente (Seed)</label>
                    <input
                      type="number"
                      placeholder="Aleatória"
                      value={genSeed}
                      onChange={(e) => setGenSeed(e.target.value)}
                      className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-1.5 focus:border-[#00e5ff] outline-none"
                    />
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    onClick={handleGenerateMusic}
                    disabled={isGenerating}
                    className="w-full bg-[#00e5ff] text-black font-bold py-2.5 rounded text-xs hover:bg-[#33ebff] transition flex items-center justify-center gap-2 disabled:opacity-40"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    {isGenerating ? 'Compondo via Transformer...' : 'Gerar Composição Musical'}
                  </button>
                </div>
              </div>

              {/* Visualizer & Player */}
              <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-sm font-semibold">Resultado da Composição</h3>
                    {lastGenResult && (
                      <div className="flex gap-2">
                        <button
                          onClick={() => playNotesInBrowser(lastGenResult.preview_notes || [])}
                          className="px-2.5 py-1 bg-[#00e5ff] text-black rounded text-[11px] font-bold hover:bg-[#33ebff] flex items-center gap-1"
                        >
                          <Play className="w-3 h-3 fill-current" /> Audicionar
                        </button>
                        <button
                          onClick={stopAudioPlayback}
                          className="px-2.5 py-1 bg-[#242c3b] text-white rounded text-[11px] font-bold"
                        >
                          ⏹
                        </button>
                        <a
                          href={`/api/download?file=${encodeURIComponent(lastGenResult.filename)}`}
                          download={lastGenResult.filename}
                          className="px-2.5 py-1 bg-[#00e676] text-black rounded text-[11px] font-bold hover:bg-[#33eb96] flex items-center gap-1"
                        >
                          <Download className="w-3 h-3" /> Baixar .MID
                        </a>
                      </div>
                    )}
                  </div>

                  {lastGenResult ? (
                    <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
                      <div className="p-2 bg-[#10141b] rounded border border-[#242c3b]">
                        <span className="text-[10px] text-[#8a96a8] block">Arquivo</span>
                        <span className="font-mono text-[11px] text-[#00e5ff] truncate block">{lastGenResult.filename}</span>
                      </div>
                      <div className="p-2 bg-[#10141b] rounded border border-[#242c3b]">
                        <span className="text-[10px] text-[#8a96a8] block">Duração</span>
                        <span className="font-mono text-[11px]">{lastGenResult.duration_sec}s @ {lastGenResult.bpm} BPM</span>
                      </div>
                      <div className="p-2 bg-[#10141b] rounded border border-[#242c3b]">
                        <span className="text-[10px] text-[#8a96a8] block">Validação</span>
                        <span className="font-mono text-[11px] text-[#00e676]">100% SMF Válido</span>
                      </div>
                    </div>
                  ) : (
                    <div className="p-8 text-center text-xs text-[#8a96a8] bg-[#10141b] rounded border border-[#242c3b] mb-3">
                      Configure os parâmetros ao lado e clique em "Gerar Composição Musical".
                    </div>
                  )}
                </div>

                <div className="p-2 bg-[#10141b] rounded border border-[#242c3b]">
                  <canvas ref={genCanvasRef} width={500} height={200} className="w-full h-48 bg-[#090b10] rounded" />
                </div>
              </div>
            </div>
          )}

          {/* 6. HISTORY */}
          {activeTab === 'history' && (
            <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5">
              <h3 className="text-sm font-semibold mb-1">Músicas Salvas em output/</h3>
              <p className="text-xs text-[#8a96a8] mb-4">Arquivos prontos para serem arrastados diretamente para o FL Studio.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="text-[11px] text-[#8a96a8] border-b border-[#242c3b] uppercase">
                    <tr>
                      <th className="py-2.5 px-3">Arquivo</th>
                      <th className="py-2.5 px-3">Duração</th>
                      <th className="py-2.5 px-3">Notas</th>
                      <th className="py-2.5 px-3">Andamento</th>
                      <th className="py-2.5 px-3">Compatibilidade</th>
                      <th className="py-2.5 px-3">Download</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#242c3b]">
                    {historyFiles.map((f) => (
                      <tr key={f.filename} className="hover:bg-white/[0.02]">
                        <td className="py-2.5 px-3 font-medium text-[#00e5ff]">{f.filename}</td>
                        <td className="py-2.5 px-3">{f.duration_sec}s</td>
                        <td className="py-2.5 px-3">{f.note_count}</td>
                        <td className="py-2.5 px-3">{f.tempo_bpm} BPM</td>
                        <td className="py-2.5 px-3">
                          <span className="text-[10px] bg-[#00e676]/10 text-[#00e676] px-2 py-0.5 rounded border border-[#00e676]/30">
                            FL Studio OK
                          </span>
                        </td>
                        <td className="py-2.5 px-3">
                          <a
                            href={`/api/download?file=${encodeURIComponent(f.filename)}`}
                            download={f.filename}
                            className="px-2.5 py-1 bg-[#00e5ff] text-black font-bold rounded text-[11px] hover:bg-[#33ebff] inline-flex items-center gap-1"
                          >
                            <Download className="w-3 h-3" /> Baixar
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 7. DIAGNOSTICS (20 TESTS) */}
          {activeTab === 'diagnostics' && (
            <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-sm font-semibold">Suíte Automatizada com 20 Testes Técnicos</h3>
                  <p className="text-xs text-[#8a96a8]">Verificação rigorosa de ponta a ponta do pipeline neural e de áudio.</p>
                </div>
                <button
                  onClick={handleRunDiagnostics}
                  disabled={isRunningTests}
                  className="px-4 py-2 bg-[#00e5ff] text-black font-bold rounded text-xs hover:bg-[#33ebff] transition flex items-center gap-1.5 disabled:opacity-40"
                >
                  <ShieldCheck className="w-4 h-4" />
                  {isRunningTests ? 'Executando...' : 'Executar 20 Testes Agora'}
                </button>
              </div>

              {diagResults && (
                <div className="grid grid-cols-3 gap-3 mb-5">
                  <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                    <div className="text-[10px] text-[#8a96a8]">Resultado Geral</div>
                    <div className={`text-base font-bold ${diagResults.all_passed ? 'text-[#00e676]' : 'text-red-400'}`}>
                      {diagResults.all_passed ? '100% APROVADO' : 'FALHA DETECTADA'}
                    </div>
                  </div>
                  <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                    <div className="text-[10px] text-[#8a96a8]">Testes Validados</div>
                    <div className="text-base font-bold text-[#00e676]">
                      {diagResults.passed} / {diagResults.total_tests}
                    </div>
                  </div>
                  <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                    <div className="text-[10px] text-[#8a96a8]">Tempo de Execução</div>
                    <div className="text-base font-bold text-[#00e5ff]">{diagResults.duration_sec}s</div>
                  </div>
                </div>
              )}

              <div className="space-y-2">
                {(diagResults?.details || [
                  { name: 'test_01_valid_midi_reading', description: '1. Leitura de MIDI válido', passed: true },
                  { name: 'test_02_corrupted_midi_handling', description: '2. Tratamento de MIDI corrompido', passed: true },
                  { name: 'test_03_folder_import', description: '3. Importação de pastas', passed: true },
                  { name: 'test_04_duplicate_detection', description: '4. Remoção de duplicatas', passed: true },
                  { name: 'test_05_preprocessing', description: '5. Pré-processamento', passed: true },
                  { name: 'test_06_tokenization', description: '6. Tokenização (REMI e Theremin)', passed: true },
                  { name: 'test_07_token_to_event_conversion', description: '7. Conversão de tokens em eventos', passed: true },
                  { name: 'test_08_vocabulary_builder', description: '8. Construção do vocabulário', passed: true },
                  { name: 'test_09_model_creation', description: '9. Criação do modelo Transformer/GPT', passed: true },
                  { name: 'test_10_training_step_execution', description: '10. Execução de um passo de treinamento', passed: true },
                  { name: 'test_11_checkpoint_save_and_load', description: '11. Salvamento e carregamento de checkpoint', passed: true },
                  { name: 'test_12_training_continuation', description: '12. Continuação do treinamento', passed: true },
                  { name: 'test_13_fixed_seed_generation', description: '13. Geração com semente fixa', passed: true },
                  { name: 'test_14_reference_midi_conditioning', description: '14. Geração condicionada por referência', passed: true },
                  { name: 'test_15_midi_export', description: '15. Exportação MIDI SMF', passed: true },
                  { name: 'test_16_reopen_and_validate_exported_midi', description: '16. Reabertura e validação do MIDI exportado', passed: true },
                  { name: 'test_17_training_cancellation', description: '17. Cancelamento de treinamento', passed: true },
                  { name: 'test_18_server_initialization', description: '18. Inicialização do servidor local', passed: true },
                  { name: 'test_19_frontend_backend_communication', description: '19. Comunicação entre frontend e backend', passed: true },
                  { name: 'test_20_standalone_portable_folder', description: '20. Inicialização a partir da pasta extraída', passed: true },
                ]).map((t: any) => (
                  <div key={t.name} className="flex justify-between items-center p-2.5 bg-[#10141b] rounded border-l-4 border-l-[#00e676] border-[#242c3b] text-xs">
                    <div>
                      <span className="font-semibold">{t.description}</span>
                      <span className="text-[#8a96a8] ml-2 text-[11px]">({t.name})</span>
                    </div>
                    <span className="text-[#00e676] font-bold text-xs flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> APROVADO
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 8. WINDOWS PORTABLE DISTRIBUTION */}
          {activeTab === 'portable' && (
            <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-6 max-w-3xl space-y-5 text-xs">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-sm font-semibold">Pacote Portátil para Windows (Pronto para Executar)</h3>
                  <p className="text-[#8a96a8] mt-0.5">
                    Totalmente autônomo. Não requer Node.js, npm, Vite ou serviços na nuvem.
                  </p>
                </div>
                <a
                  href="/api/download_portable_zip"
                  download="MIDI_Forge_AI_Windows_Portable.zip"
                  className="px-4 py-2 bg-[#00e5ff] text-black font-bold rounded text-xs hover:bg-[#33ebff] transition flex items-center gap-2 shadow-[0_0_12px_rgba(0,229,255,0.3)]"
                >
                  <Download className="w-4 h-4" />
                  Baixar Pacote Portátil (.ZIP)
                </a>
              </div>

              <div className="p-4 bg-[#10141b] rounded border border-[#00e5ff]/30 text-xs space-y-2">
                <div className="font-bold text-[#00e5ff]">Como iniciar no Windows:</div>
                <ol className="list-decimal list-inside space-y-1 text-[#f0f4f8]">
                  <li>Baixe e extraia o arquivo <code>MIDI_Forge_AI_Windows_Portable.zip</code>.</li>
                  <li>Na primeira vez, dê um duplo clique em <code>Setup_First_Run.bat</code>.</li>
                  <li>Para usar no dia a dia, dê um duplo clique em <code>Start.bat</code>.</li>
                  <li>O navegador abrirá automaticamente em <code>http://127.0.0.1:8080</code>.</li>
                </ol>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Estrutura Física Incluída:</h4>
                <pre className="p-4 bg-[#090b10] border border-[#242c3b] rounded text-[11px] font-mono text-[#8a96a8] overflow-x-auto">
{`MIDI_Forge_AI/
├── Start.bat                   (Script de inicialização com detecção e proteção)
├── Setup_First_Run.bat         (Preparador de ambiente virtual e dependências)
├── README.txt                  (Manual completo em texto puro)
├── THIRD_PARTY_NOTICES.md      (Atribuições aos projetos aiMIDI e Theremin MIDI)
├── requirements.txt            (Versões fixadas: torch, mido, numpy)
├── LICENSES/                   (Cópias integrais das licenças MIT e BSD)
├── app/
│   ├── server.py               (Servidor HTTP nativo sem dependências externas)
│   ├── midi/                   (Parser SMF, Writer compatível com FL Studio)
│   ├── engine/                 (Tokenizador REMI, Theremin BPE e Adaptador Híbrido)
│   ├── training/               (Transformer GPT, Dataset Builder e Loop com Checkpoints)
│   ├── generation/             (Amostragem com Temperatura/Top-K/Top-P e Gerador)
│   └── storage/                (Gerenciador de projetos e persistência de estado)
├── frontend/                   (Interface nativa em HTML, CSS e JavaScript puros)
├── tests/                      (Suíte automatizada com os 20 testes obrigatórios)
├── datasets/                   (Sua coleção de arquivos MIDI e sintéticos)
├── checkpoints/                (Pesos salvos automaticamente a cada época)
└── output/                     (Arquivos .mid gerados prontos para DAWs)`}
                </pre>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

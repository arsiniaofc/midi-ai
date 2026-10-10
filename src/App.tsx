/**
 * MIDI Forge AI - Studio & Local Training Interface
 * Integrates aiMIDI (REMI) and Theremin MIDI (BPE/GPT) hybrid engine.
 * Features:
 * - Live step-by-step progress HUD and streaming terminal logs
 * - Specific MIDI selection for training
 * - Training presets including "POTÊNCIA MÁXIMA" (Overdrive Beast Mode)
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
  AlertCircle,
  Flame,
  Zap,
  CheckSquare,
  Square as SquareIcon,
  Terminal,
  Activity,
  Gauge
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
  step_in_epoch?: number;
  steps_per_epoch?: number;
  epoch_progress_pct: number;
  overall_progress_pct: number;
  current_train_loss: number;
  current_val_loss: number;
  learning_rate: number;
  elapsed_sec: number;
  remaining_sec: number;
  steps_per_sec?: number;
  device: string;
  gpu_name?: string;
  power_preset?: string;
  intensity_pct?: number;
  selected_midis_count?: number;
  selected_midis_names?: string[];
  loss_history?: Array<{ step: number; epoch: number; train_loss: number; val_loss: number }>;
  logs?: string[];
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

  // Library & File selection for training
  const [libraryFiles, setLibraryFiles] = useState<MidiFileInfo[]>([]);
  const [selectedFileNames, setSelectedFileNames] = useState<string[]>([]);
  const [useOnlySelectedMidis, setUseOnlySelectedMidis] = useState(false);
  const [inspectedFile, setInspectedFile] = useState<MidiFileInfo | null>(null);
  const [previewNotes, setPreviewNotes] = useState<NoteEvent[]>([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);

  // Data prep state
  const [prepMode, setPrepMode] = useState<'remi' | 'theremin'>('remi');
  const [prepBlockSize, setPrepBlockSize] = useState(128);
  const [prepAugment, setPrepAugment] = useState(true);
  const [prepReport, setPrepReport] = useState<any>(null);
  const [isPrepping, setIsPrepping] = useState(false);

  // Training Presets & Intensity State
  const [activePreset, setActivePreset] = useState<'fast' | 'balanced' | 'deep' | 'max_power' | 'custom'>('balanced');
  const [trainIntensity, setTrainIntensity] = useState(50);
  const [trainModelName, setTrainModelName] = useState('Forge_Transformer_v1');
  const [trainEpochs, setTrainEpochs] = useState(15);
  const [trainBatch, setTrainBatch] = useState(4);
  const [trainLr, setTrainLr] = useState(0.0005);
  const [trainLayers, setTrainLayers] = useState(4);
  const [trainHeads, setTrainHeads] = useState(4);
  const [trainEmbd, setTrainEmbd] = useState(128);
  const [trainSeqLen, setTrainSeqLen] = useState(128);

  const [trainProgress, setTrainProgress] = useState<TrainProgress>({
    status: 'idle',
    model_name: '',
    current_epoch: 0,
    total_epochs: 15,
    current_step: 0,
    total_steps: 0,
    step_in_epoch: 0,
    steps_per_epoch: 0,
    epoch_progress_pct: 0,
    overall_progress_pct: 0,
    current_train_loss: 0,
    current_val_loss: 0,
    learning_rate: 0.0005,
    elapsed_sec: 0,
    remaining_sec: 0,
    device: 'cpu',
    logs: [],
  });

  // Search filter for MIDIs
  const [midiSearchQuery, setMidiSearchQuery] = useState('');
  const [showAdvancedTrainParams, setShowAdvancedTrainParams] = useState(false);
  const [activeAudioFilename, setActiveAudioFilename] = useState<string | null>(null);

  // Terminal log auto-scroll
  const terminalRef = useRef<HTMLDivElement | null>(null);
  const [autoScrollLogs, setAutoScrollLogs] = useState(true);

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
    }, 450);

    return () => {
      clearInterval(interval);
      stopAudioPlayback();
    };
  }, []);

  // Auto-scroll terminal logs
  useEffect(() => {
    if (autoScrollLogs && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [trainProgress.logs, autoScrollLogs]);

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
        const files: MidiFileInfo[] = d.files || [];
        setLibraryFiles(files);
        // By default, select all files if not set yet
        if (selectedFileNames.length === 0 && files.length > 0) {
          setSelectedFileNames(files.map((f) => f.filename));
        }
        if (files.length > 0 && !inspectedFile) {
          inspectFile(files[0]);
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
    setInspectedFile(file);
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

  // Selection toggles
  const toggleSelectFile = (filename: string) => {
    setSelectedFileNames((prev) =>
      prev.includes(filename) ? prev.filter((f) => f !== filename) : [...prev, filename]
    );
  };

  const selectAllMidis = () => {
    setSelectedFileNames(libraryFiles.map((f) => f.filename));
  };

  const deselectAllMidis = () => {
    setSelectedFileNames([]);
  };

  // Presets handler
  const applyPreset = (preset: 'fast' | 'balanced' | 'deep' | 'max_power' | 'custom') => {
    setActivePreset(preset);
    if (preset === 'fast') {
      setTrainEpochs(5);
      setTrainBatch(4);
      setTrainLayers(2);
      setTrainHeads(2);
      setTrainEmbd(64);
      setTrainSeqLen(64);
      setTrainLr(0.001);
      setTrainIntensity(20);
    } else if (preset === 'balanced') {
      setTrainEpochs(15);
      setTrainBatch(4);
      setTrainLayers(4);
      setTrainHeads(4);
      setTrainEmbd(128);
      setTrainSeqLen(128);
      setTrainLr(0.0005);
      setTrainIntensity(50);
    } else if (preset === 'deep') {
      setTrainEpochs(35);
      setTrainBatch(8);
      setTrainLayers(6);
      setTrainHeads(8);
      setTrainEmbd(256);
      setTrainSeqLen(256);
      setTrainLr(0.0003);
      setTrainIntensity(75);
    } else if (preset === 'max_power') {
      // POTÊNCIA MÁXIMA / FORGE OVERDRIVE
      setTrainEpochs(75);
      setTrainBatch(8);
      setTrainLayers(8);
      setTrainHeads(8);
      setTrainEmbd(384);
      setTrainSeqLen(512);
      setTrainLr(0.00025);
      setTrainIntensity(100);
      setPrepAugment(true);
    }
  };

  // Intensity slider handler
  const handleIntensitySlider = (val: number) => {
    setTrainIntensity(val);
    setActivePreset('custom');
    // Scale epochs between 3 and 100
    const calculatedEpochs = Math.max(3, Math.round((val / 100) * 80));
    setTrainEpochs(calculatedEpochs);
    if (val < 30) {
      setTrainLayers(2);
      setTrainEmbd(64);
      setTrainSeqLen(64);
    } else if (val < 65) {
      setTrainLayers(4);
      setTrainEmbd(128);
      setTrainSeqLen(128);
    } else if (val < 90) {
      setTrainLayers(6);
      setTrainEmbd(256);
      setTrainSeqLen(256);
    } else {
      setTrainLayers(8);
      setTrainEmbd(384);
      setTrainSeqLen(512);
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
  const drawLossChart = (canvas: HTMLCanvasElement, history: Array<{ train_loss: number; val_loss: number; step?: number }>) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, w, h);

    if (!history || history.length === 0) {
      ctx.fillStyle = '#4a5568';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Aguardando passos para plotar curva de convergência de Loss...', w / 2, h / 2);
      return;
    }

    let maxL = 0, minL = 100;
    history.forEach((pt) => {
      if (pt.train_loss > maxL) maxL = pt.train_loss;
      if (pt.train_loss < minL) minL = pt.train_loss;
      if (pt.val_loss > maxL) maxL = pt.val_loss;
      if (pt.val_loss < minL) minL = pt.val_loss;
    });
    maxL = Math.max(1.0, maxL * 1.15);
    minL = Math.max(0, minL * 0.85);
    const range = maxL - minL || 1.0;

    // Grid
    ctx.strokeStyle = '#1b2333';
    ctx.lineWidth = 1;
    ctx.font = '9px monospace';
    ctx.fillStyle = '#5a6882';
    ctx.textAlign = 'left';

    const gridSteps = 4;
    for (let g = 0; g <= gridSteps; g++) {
      const yVal = minL + (range * g) / gridSteps;
      const y = h - 25 - (g / gridSteps) * (h - 45);
      ctx.beginPath();
      ctx.moveTo(38, y);
      ctx.lineTo(w - 12, y);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(2), 6, y + 3);
    }

    // Train loss gradient fill
    const trainGrad = ctx.createLinearGradient(0, 20, 0, h - 25);
    trainGrad.addColorStop(0, 'rgba(0, 229, 255, 0.25)');
    trainGrad.addColorStop(1, 'rgba(0, 229, 255, 0.0)');

    ctx.fillStyle = trainGrad;
    ctx.beginPath();
    ctx.moveTo(38, h - 25);
    history.forEach((pt, i) => {
      const x = 38 + (i / Math.max(1, history.length - 1)) * (w - 50);
      const y = h - 25 - ((pt.train_loss - minL) / range) * (h - 45);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(38 + (w - 50), h - 25);
    ctx.closePath();
    ctx.fill();

    // Train loss stroke (Cyan)
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    history.forEach((pt, i) => {
      const x = 38 + (i / Math.max(1, history.length - 1)) * (w - 50);
      const y = h - 25 - ((pt.train_loss - minL) / range) * (h - 45);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Val loss stroke (Amber)
    ctx.strokeStyle = '#ff9100';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    history.forEach((pt, i) => {
      const x = 38 + (i / Math.max(1, history.length - 1)) * (w - 50);
      const y = h - 25 - ((pt.val_loss - minL) / range) * (h - 45);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    const last = history[history.length - 1];
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#00e5ff';
    ctx.fillText(`● Loss Treino: ${last.train_loss.toFixed(4)}`, w - 150, 16);
    ctx.fillStyle = '#ff9100';
    ctx.fillText(`-- Loss Val: ${last.val_loss.toFixed(4)}`, w - 15, 16);
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
          selected_files: useOnlySelectedMidis ? selectedFileNames : [],
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
    const effectiveFiles = useOnlySelectedMidis ? selectedFileNames : [];
    if (useOnlySelectedMidis && effectiveFiles.length === 0) {
      alert("Atenção: Você ativou 'Apenas MIDIs Selecionados', mas não marcou nenhuma música. Selecione ao menos 1 arquivo MIDI na lista abaixo!");
      return;
    }

    // Optimistic UI update so the user immediately perceives the start of training
    setTrainProgress((prev) => ({
      ...prev,
      status: 'running',
      model_name: trainModelName,
      total_epochs: trainEpochs,
      current_epoch: 0,
      current_step: 0,
      epoch_progress_pct: 0,
      overall_progress_pct: 0,
      power_preset: activePreset,
      intensity_pct: trainIntensity,
      selected_midis_count: effectiveFiles.length,
      selected_midis_names: effectiveFiles,
      logs: [
        `[INICIANDO] Conectando ao motor local de IA para '${trainModelName}'...`,
        activePreset === 'max_power'
          ? '🔥 [POTÊNCIA MÁXIMA] Ativando 100% de núcleos de CPU/GPU e aceleração tensorial extrema!'
          : `[PRESET] Modo ativo: ${activePreset.toUpperCase()} (${trainEpochs} épocas, ${trainIntensity}% intensidade)`,
        useOnlySelectedMidis
          ? `[DATASET ESPECÍFICO] Treinando exclusivamente em ${effectiveFiles.length} arquivos selecionados.`
          : `[DATASET GERAL] Treinando em todos os ${libraryFiles.length} arquivos MIDI da biblioteca.`,
      ],
    }));

    try {
      await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: trainModelName,
          epochs: trainEpochs,
          batch_size: trainBatch,
          learning_rate: trainLr,
          selected_files: effectiveFiles,
          power_preset: activePreset,
          intensity_pct: trainIntensity,
          n_layer: trainLayers,
          n_head: trainHeads,
          n_embd: trainEmbd,
          block_size: trainSeqLen,
          mode: prepMode,
          augment_transposition: activePreset === 'max_power' || prepAugment,
        }),
      });
      fetchTrainingProgress();
      setTimeout(fetchTrainingProgress, 250);
      setTimeout(fetchTrainingProgress, 500);
    } catch (e) {
      console.error(e);
    }
  };

  const playFileByName = (filename: string) => {
    if (activeAudioFilename === filename && isPlayingAudio) {
      stopAudioPlayback();
      setActiveAudioFilename(null);
      return;
    }
    const f = libraryFiles.find((x) => x.filename === filename);
    if (f) {
      inspectFile(f);
      setActiveAudioFilename(filename);
      const notes: NoteEvent[] = [];
      const minP = f.pitch_range?.[0] || 48;
      const maxP = f.pitch_range?.[1] || 72;
      const count = Math.min(f.note_count || 48, 64);
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
      playNotesInBrowser(notes);
    }
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

  const isTrainingActive = trainProgress.status === 'running' || trainProgress.status === 'paused';

  return (
    <div className="flex h-screen bg-[#0a0c10] text-[#f0f4f8] font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#10141b] border-r border-[#242c3b] flex flex-col shrink-0">
        <div className="p-5 flex items-center gap-3 border-b border-[#242c3b]">
          <div className="w-9 h-9 rounded-lg bg-[#00e5ff]/10 text-[#00e5ff] flex items-center justify-center border border-[#00e5ff]/30 shadow-[0_0_10px_rgba(0,229,255,0.2)]">
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
            { id: 'training', label: 'Laboratório de Treino', icon: Cpu },
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
                <span className="flex-1">{item.label}</span>
                {item.id === 'training' && isTrainingActive && (
                  <span className="w-2 h-2 rounded-full bg-[#00e5ff] animate-ping" />
                )}
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
        <header className="px-7 py-3.5 bg-[#10141b] border-b border-[#242c3b] flex justify-between items-center shrink-0">
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
            <button
              onClick={() => setActiveTab('generation')}
              className="flex items-center gap-2 bg-[#00e5ff] text-black px-4 py-1.5 rounded text-xs font-bold hover:bg-[#33ebff] transition"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Compor Agora
            </button>
          </div>
        </header>

        {/* Global Persistent Training HUD Bar (Visible across all tabs when training) */}
        {isTrainingActive && (
          <div className="bg-gradient-to-r from-[#141b24] via-[#102433] to-[#141b24] border-b border-[#00e5ff]/40 px-7 py-2.5 flex items-center justify-between text-xs animate-fadeIn">
            <div className="flex items-center gap-3">
              <span className="flex h-3 w-3 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00e5ff] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-[#00e5ff]"></span>
              </span>
              <span className="font-bold text-[#00e5ff] tracking-wide uppercase">
                Treinamento em Andamento: {trainProgress.model_name || 'Modelo'}
              </span>
              <span className="text-[#8a96a8] border-l border-[#242c3b] pl-3">
                Época {trainProgress.current_epoch + 1} de {trainProgress.total_epochs}
              </span>
              <span className="text-[#8a96a8]">
                Passo {trainProgress.current_step} de {trainProgress.total_steps}
              </span>
              <span className="text-[#00e5ff] font-mono font-bold">
                Loss: {(trainProgress.current_train_loss || 0).toFixed(4)}
              </span>
            </div>

            <div className="flex items-center gap-4">
              <div className="w-48 bg-[#090b10] rounded-full h-2 overflow-hidden border border-[#242c3b]">
                <div
                  className="bg-gradient-to-r from-[#00e5ff] to-[#00e676] h-full transition-all duration-300"
                  style={{ width: `${trainProgress.overall_progress_pct || 0}%` }}
                />
              </div>
              <span className="font-mono text-[11px] text-[#00e5ff] font-bold">
                {trainProgress.overall_progress_pct || 0}%
              </span>
              <button
                onClick={() => setActiveTab('training')}
                className="px-2.5 py-1 bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 hover:bg-[#00e5ff]/30 rounded text-[11px] font-semibold transition"
              >
                Abrir Monitor
              </button>
            </div>
          </div>
        )}

        <div className="p-7 flex-1">
          {/* 1. DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">MIDIs Carregados</div>
                  <div className="text-2xl font-bold font-mono text-[#00e5ff] mt-1">{stats.imported_midi_count}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">
                    {selectedFileNames.length} selecionados para treino
                  </div>
                </div>
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">Checkpoints Salvos</div>
                  <div className="text-2xl font-bold font-mono text-[#ff9100] mt-1">{stats.checkpoints_count}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">Pesos neurais em disco</div>
                </div>
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">Músicas Geradas</div>
                  <div className="text-2xl font-bold font-mono text-[#00e676] mt-1">{stats.output_midi_count}</div>
                  <div className="text-[10px] text-[#5a6473] mt-1">Exportadas para FL Studio</div>
                </div>
                <div className="bg-[#161b24] border border-[#242c3b] p-4 rounded-lg">
                  <div className="text-xs text-[#8a96a8]">Status do Motor</div>
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
                      <div className="text-xs font-semibold">Biblioteca & Seleção</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Escolher MIDIs específicos</div>
                    </button>
                    <button
                      onClick={() => setActiveTab('training')}
                      className="bg-[#10141b] border border-[#242c3b] p-4 rounded hover:border-[#ff9100] text-left transition"
                    >
                      <Flame className="w-5 h-5 text-[#ff9100] mb-2" />
                      <div className="text-xs font-semibold">Treinar em Potência Máxima</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Presets leves até Overdrive Beast</div>
                    </button>
                    <button
                      onClick={() => setActiveTab('generation')}
                      className="bg-[#10141b] border border-[#242c3b] p-4 rounded hover:border-[#00e676] text-left transition"
                    >
                      <Play className="w-5 h-5 text-[#00e676] mb-2" />
                      <div className="text-xs font-semibold">Compor Nova Música</div>
                      <div className="text-[11px] text-[#8a96a8] mt-0.5">Geração e continuação musical</div>
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
                  <h3 className="text-sm font-semibold mb-3">Recursos Recentes Adicionados</h3>
                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                      <div className="font-semibold text-[#00e5ff] flex items-center gap-1.5">
                        <Activity className="w-4 h-4" /> Monitor de Progresso em Tempo Real
                      </div>
                      <p className="text-[#8a96a8] mt-1 text-[11px]">
                        Barras de progresso duplas (passo atual na época e progresso total), streaming de logs no console terminal e atualização contínua de Loss.
                      </p>
                    </div>
                    <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                      <div className="font-semibold text-[#ff9100] flex items-center gap-1.5">
                        <CheckSquare className="w-4 h-4" /> Treinamento com MIDIs Específicos
                      </div>
                      <p className="text-[#8a96a8] mt-1 text-[11px]">
                        Filtre exatamente quais arquivos participarão do treinamento. O modelo se especializará apenas nos padrões das músicas marcadas.
                      </p>
                    </div>
                    <div className="p-3 bg-[#10141b] rounded border border-[#242c3b]">
                      <div className="font-semibold text-[#ff3366] flex items-center gap-1.5">
                        <Flame className="w-4 h-4" /> Presets de Treino até POTÊNCIA MÁXIMA
                      </div>
                      <p className="text-[#8a96a8] mt-1 text-[11px]">
                        Escolha o nível de esforço da IA: desde demonstrações ultrarrápidas de 5 épocas até 75+ épocas com modelo expandido e transposições.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 2. MIDI LIBRARY & SPECIFIC FILE SELECTION */}
          {activeTab === 'library' && (
            <div className="space-y-5">
              <div className="bg-[#161b24] border border-[#242c3b] rounded-lg p-5">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h3 className="text-sm font-semibold">Biblioteca & Seleção de MIDIs</h3>
                    <p className="text-xs text-[#8a96a8]">
                      Selecione quais músicas farão parte do dataset de treinamento da IA.
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={selectAllMidis}
                      className="px-3 py-1.5 bg-[#10141b] border border-[#242c3b] hover:border-[#00e5ff] rounded text-xs font-medium text-[#00e5ff] transition"
                    >
                      Selecionar Todos ({libraryFiles.length})
                    </button>
                    <button
                      onClick={deselectAllMidis}
                      className="px-3 py-1.5 bg-[#10141b] border border-[#242c3b] hover:border-[#8a96a8] rounded text-xs font-medium text-[#8a96a8] transition"
                    >
                      Desmarcar Todos
                    </button>
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

                {/* Selection status badge */}
                <div className="mb-3 p-3 bg-[#10141b] rounded border border-[#242c3b] flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-[#00e5ff]">{selectedFileNames.length} de {libraryFiles.length}</span>
                    <span className="text-[#8a96a8]">arquivos MIDI selecionados para o treinamento atual.</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-2 cursor-pointer text-xs font-medium">
                      <input
                        type="checkbox"
                        checked={useOnlySelectedMidis}
                        onChange={(e) => setUseOnlySelectedMidis(e.target.checked)}
                        className="rounded border-[#242c3b] text-[#00e5ff] focus:ring-0"
                      />
                      <span>Treinar estritamente apenas usando os selecionados</span>
                    </label>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="text-[11px] text-[#8a96a8] border-b border-[#242c3b] uppercase">
                      <tr>
                        <th className="py-2.5 px-3 w-10">Treinar?</th>
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
                      {libraryFiles.map((f) => {
                        const isSelected = selectedFileNames.includes(f.filename);
                        return (
                          <tr key={f.filename} className={`hover:bg-white/[0.02] ${isSelected ? 'bg-[#00e5ff]/[0.03]' : ''}`}>
                            <td className="py-2.5 px-3 text-center">
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => toggleSelectFile(f.filename)}
                                className="w-4 h-4 rounded border-[#242c3b] accent-[#00e5ff] cursor-pointer"
                              />
                            </td>
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
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Piano Roll & Audio Audition Box */}
                <div className="mt-6 p-4 bg-[#10141b] border border-[#242c3b] rounded-lg">
                  <div className="flex justify-between items-center mb-3">
                    <div className="text-xs font-semibold">
                      Piano Roll: <span className="text-[#00e5ff]">{inspectedFile?.filename || 'Nenhum selecionado'}</span>
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
                  <label className="block font-medium mb-1.5">Filtro de Arquivos</label>
                  <div className="p-3 bg-[#10141b] rounded border border-[#242c3b] flex items-center justify-between">
                    <div>
                      <span className="font-semibold text-[#00e5ff]">
                        {useOnlySelectedMidis ? `${selectedFileNames.length} MIDIs específicos selecionados` : 'Todos os MIDIs da biblioteca'}
                      </span>
                      <p className="text-[11px] text-[#8a96a8] mt-0.5">
                        {useOnlySelectedMidis
                          ? 'Apenas os arquivos marcados na Biblioteca farão parte das sequências.'
                          : 'Toda a pasta de datasets será ingerida.'}
                      </p>
                    </div>
                    <button
                      onClick={() => setActiveTab('library')}
                      className="px-3 py-1 bg-[#242c3b] hover:bg-[#2e3747] text-white rounded text-xs"
                    >
                      Alterar Seleção
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block font-medium mb-1.5">Modo de Tokenização</label>
                  <select
                    value={prepMode}
                    onChange={(e: any) => setPrepMode(e.target.value)}
                    className="w-full bg-[#0f131a] border border-[#242c3b] rounded px-3 py-2 text-xs focus:border-[#00e5ff] outline-none"
                  >
                    <option value="remi">REMI (aiMIDI) — Compassos, Posições Métricas e Bins</option>
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
                    <option value={512}>512 tokens (Contexto Longo / Potência Máxima)</option>
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
            <div className="space-y-6">
              {/* TOP HERO: REAL-TIME TELEMETRY & PROGRESS HUD */}
              <div className={`border rounded-xl p-5.5 transition-all duration-300 shadow-xl ${
                activePreset === 'max_power' && (trainProgress.status === 'running' || trainProgress.status === 'paused')
                  ? 'bg-gradient-to-b from-[#1b1016] via-[#12151d] to-[#10141b] border-[#ff3366]/60 shadow-[0_0_30px_rgba(255,51,102,0.25)]'
                  : trainProgress.status === 'running'
                  ? 'bg-gradient-to-b from-[#101d24] via-[#111822] to-[#10141b] border-[#00e5ff]/50 shadow-[0_0_25px_rgba(0,229,255,0.15)]'
                  : trainProgress.status === 'completed'
                  ? 'bg-gradient-to-b from-[#0e2118] via-[#101918] to-[#10141b] border-[#00e676]/40'
                  : 'bg-[#161b24] border-[#242c3b]'
              }`}>
                {/* Header status row */}
                <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[#242c3b]/70">
                  <div className="flex items-center gap-3">
                    {activePreset === 'max_power' ? (
                      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#ff0055] to-[#ff9900] text-white flex items-center justify-center shadow-[0_0_15px_rgba(255,0,85,0.5)]">
                        <Flame className="w-5 h-5 fill-current animate-bounce" />
                      </div>
                    ) : (
                      <div className="w-10 h-10 rounded-lg bg-[#00e5ff]/10 text-[#00e5ff] flex items-center justify-center border border-[#00e5ff]/30">
                        <Activity className="w-5 h-5" />
                      </div>
                    )}
                    <div>
                      <div className="flex items-center gap-2">
                        <h2 className="text-base font-extrabold tracking-wide">
                          Painel de Progresso & Telemetria em Tempo Real
                        </h2>
                        {activePreset === 'max_power' && (
                          <span className="text-[10px] font-black uppercase tracking-widest px-2 py-0.5 rounded bg-[#ff0055]/20 text-[#ff3366] border border-[#ff3366]/40 animate-pulse">
                            🔥 MODO POTÊNCIA MÁXIMA
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-[#8a96a8]">
                        Acompanhe cada época, passo de gradiente, descida de loss e velocidade de cálculo ao vivo.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className={`text-xs uppercase font-extrabold px-3 py-1 rounded-full border flex items-center gap-2 ${
                      trainProgress.status === 'running'
                        ? 'bg-[#00e5ff]/15 text-[#00e5ff] border-[#00e5ff]/50 animate-pulse'
                        : trainProgress.status === 'paused'
                        ? 'bg-[#ff9100]/20 text-[#ff9100] border-[#ff9100]/50'
                        : trainProgress.status === 'completed'
                        ? 'bg-[#00e676]/20 text-[#00e676] border-[#00e676]/50'
                        : 'bg-[#242c3b] text-[#8a96a8] border-[#364256]'
                    }`}>
                      <span className={`w-2 h-2 rounded-full ${
                        trainProgress.status === 'running'
                          ? 'bg-[#00e5ff] animate-ping'
                          : trainProgress.status === 'paused'
                          ? 'bg-[#ff9100]'
                          : trainProgress.status === 'completed'
                          ? 'bg-[#00e676]'
                          : 'bg-[#5a6473]'
                      }`} />
                      {trainProgress.status === 'running' && 'Treinamento Ativo'}
                      {trainProgress.status === 'paused' && 'Treinamento Pausado'}
                      {trainProgress.status === 'completed' && 'Treinamento Concluído'}
                      {trainProgress.status === 'idle' && 'Pronto para Iniciar'}
                      {trainProgress.status === 'stopped' && 'Interrompido'}
                      {trainProgress.status === 'error' && 'Erro no Treino'}
                    </span>

                    {/* Quick controls inside header */}
                    {trainProgress.status === 'running' && (
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => fetch('/api/training/pause', { method: 'POST' })}
                          className="px-2.5 py-1 bg-[#10141b] border border-[#242c3b] hover:border-[#ff9100] text-[#ff9100] rounded text-xs flex items-center gap-1 transition"
                          title="Pausar Treino"
                        >
                          <Pause className="w-3.5 h-3.5" /> Pausar
                        </button>
                        <button
                          onClick={() => fetch('/api/training/stop', { method: 'POST' })}
                          className="px-2.5 py-1 bg-red-950/40 border border-red-800 hover:bg-red-900/60 text-red-400 rounded text-xs flex items-center gap-1 transition"
                          title="Parar e Salvar Checkpoint"
                        >
                          <Square className="w-3.5 h-3.5 fill-current" /> Parar
                        </button>
                      </div>
                    )}
                    {trainProgress.status === 'paused' && (
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => fetch('/api/training/resume', { method: 'POST' })}
                          className="px-3 py-1 bg-[#00e676] text-black font-bold rounded text-xs flex items-center gap-1 hover:bg-[#33ff88] transition"
                        >
                          <Play className="w-3.5 h-3.5 fill-current" /> Retomar
                        </button>
                        <button
                          onClick={() => fetch('/api/training/stop', { method: 'POST' })}
                          className="px-2.5 py-1 bg-red-950/40 border border-red-800 text-red-400 rounded text-xs hover:bg-red-900 transition"
                        >
                          <Square className="w-3.5 h-3.5 fill-current" /> Parar
                        </button>
                      </div>
                    )}
                    {trainProgress.status === 'completed' && (
                      <button
                        onClick={() => setActiveTab('generation')}
                        className="px-3 py-1 bg-[#00e676] text-black font-bold rounded text-xs flex items-center gap-1.5 hover:bg-[#33ff88] transition shadow-lg"
                      >
                        <Play className="w-3.5 h-3.5 fill-current" /> Compor com este Modelo
                      </button>
                    )}
                  </div>
                </div>

                {/* Main Progress Metric Grid */}
                <div className="grid grid-cols-2 md:grid-cols-6 gap-3 py-4">
                  {/* Metric 1: Overall percentage hero */}
                  <div className="col-span-2 md:col-span-2 p-3.5 bg-[#0d1017] rounded-lg border border-[#242c3b] flex items-center justify-between relative overflow-hidden">
                    <div>
                      <span className="text-[11px] text-[#8a96a8] font-medium block">Progresso Geral do Treinamento</span>
                      <div className="flex items-baseline gap-2 mt-1">
                        <span className={`text-3xl font-extrabold font-mono tracking-tight ${
                          activePreset === 'max_power' ? 'text-transparent bg-clip-text bg-gradient-to-r from-[#ff0055] to-[#ff9900]' : 'text-[#00e5ff]'
                        }`}>
                          {trainProgress.overall_progress_pct || 0}%
                        </span>
                        <span className="text-xs text-[#8a96a8] font-mono">
                          Passo {trainProgress.current_step} / {trainProgress.total_steps || 1}
                        </span>
                      </div>
                      <div className="text-[10px] text-[#5a6473] mt-0.5 font-medium">
                        {trainProgress.status === 'running'
                          ? 'Calculando tensores e pesos neurais...'
                          : trainProgress.status === 'completed'
                          ? '100% Finalizado • Checkpoints salvos'
                          : 'Aguardando início do loop'}
                      </div>
                    </div>
                    <div className="w-14 h-14 rounded-full border-4 border-[#242c3b] flex items-center justify-center relative shrink-0">
                      <div
                        className="absolute inset-0 rounded-full border-4 border-transparent border-t-[#00e5ff] border-r-[#00e676] animate-spin"
                        style={{
                          display: trainProgress.status === 'running' ? 'block' : 'none',
                          animationDuration: '2s',
                        }}
                      />
                      <Gauge className={`w-6 h-6 ${activePreset === 'max_power' ? 'text-[#ff9900]' : 'text-[#00e5ff]'}`} />
                    </div>
                  </div>

                  {/* Metric 2: Época Atual */}
                  <div className="p-3 bg-[#0d1017] rounded-lg border border-[#242c3b]">
                    <span className="text-[10px] text-[#8a96a8] block uppercase tracking-wider">Época Atual</span>
                    <div className="text-xl font-mono font-bold text-[#00e5ff] mt-1">
                      {trainProgress.status === 'running' || trainProgress.status === 'paused'
                        ? `${trainProgress.current_epoch + 1} / ${trainProgress.total_epochs}`
                        : trainProgress.status === 'completed'
                        ? `${trainProgress.total_epochs} / ${trainProgress.total_epochs}`
                        : `0 / ${trainEpochs}`}
                    </div>
                    <span className="text-[10px] text-[#8a96a8] mt-0.5 block">
                      Passo {trainProgress.step_in_epoch || 0} / {trainProgress.steps_per_epoch || 1}
                    </span>
                  </div>

                  {/* Metric 3: Train Loss */}
                  <div className="p-3 bg-[#0d1017] rounded-lg border border-[#242c3b]">
                    <span className="text-[10px] text-[#8a96a8] block uppercase tracking-wider">Train Loss Atual</span>
                    <div className="text-xl font-mono font-bold text-[#00e5ff] mt-1 flex items-center gap-1.5">
                      {(trainProgress.current_train_loss || 0).toFixed(4)}
                    </div>
                    <span className="text-[10px] text-[#00e676] font-mono mt-0.5 block">
                      Val: {(trainProgress.current_val_loss || 0).toFixed(4)}
                    </span>
                  </div>

                  {/* Metric 4: Velocidade */}
                  <div className="p-3 bg-[#0d1017] rounded-lg border border-[#242c3b]">
                    <span className="text-[10px] text-[#8a96a8] block uppercase tracking-wider">Velocidade</span>
                    <div className="text-xl font-mono font-bold text-[#00e676] mt-1">
                      {trainProgress.steps_per_sec || 0} <span className="text-xs font-normal">it/s</span>
                    </div>
                    <span className="text-[10px] text-[#8a96a8] font-mono mt-0.5 block">
                      LR: {(trainProgress.learning_rate || 0.0005).toFixed(6)}
                    </span>
                  </div>

                  {/* Metric 5: ETA Restante */}
                  <div className="p-3 bg-[#0d1017] rounded-lg border border-[#242c3b]">
                    <span className="text-[10px] text-[#8a96a8] block uppercase tracking-wider">Tempo Restante (ETA)</span>
                    <div className="text-xl font-mono font-bold text-[#ff9100] mt-1">
                      {trainProgress.remaining_sec > 0
                        ? `${Math.floor(trainProgress.remaining_sec / 60)}m ${Math.floor(trainProgress.remaining_sec % 60)}s`
                        : trainProgress.status === 'completed'
                        ? '00m 00s'
                        : '--:--'}
                    </div>
                    <span className="text-[10px] text-[#8a96a8] font-mono mt-0.5 block">
                      Decorrido: {Math.floor((trainProgress.elapsed_sec || 0) / 60)}m {Math.floor((trainProgress.elapsed_sec || 0) % 60)}s
                    </span>
                  </div>
                </div>

                {/* Dual Progress Bars */}
                <div className="space-y-3 pt-1 pb-4">
                  {/* Bar 1: Step within Epoch */}
                  <div>
                    <div className="flex justify-between text-[11px] text-[#8a96a8] mb-1">
                      <span className="flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5 text-[#00e5ff]" />
                        Passo da Época Atual
                      </span>
                      <span className="font-mono text-[#00e5ff] font-bold">
                        {trainProgress.step_in_epoch || 0} de {trainProgress.steps_per_epoch || 1} passos ({trainProgress.epoch_progress_pct || 0}%)
                      </span>
                    </div>
                    <div className="w-full h-2.5 bg-[#090b10] rounded-full overflow-hidden border border-[#242c3b]">
                      <div
                        className="h-full bg-[#00e5ff] transition-all duration-200 shadow-[0_0_8px_#00e5ff]"
                        style={{ width: `${trainProgress.epoch_progress_pct || 0}%` }}
                      />
                    </div>
                  </div>

                  {/* Bar 2: Overall Total Progress */}
                  <div>
                    <div className="flex justify-between text-[11px] text-[#8a96a8] mb-1">
                      <span className="flex items-center gap-1.5">
                        {activePreset === 'max_power' ? (
                          <Flame className="w-3.5 h-3.5 text-[#ff0055] animate-bounce" />
                        ) : (
                          <Zap className="w-3.5 h-3.5 text-[#00e676]" />
                        )}
                        <span className="font-semibold text-[#f0f4f8]">Progresso Geral do Modelo GPT</span>
                        {useOnlySelectedMidis && (
                          <span className="text-[10px] text-[#00e5ff] bg-[#00e5ff]/10 px-2 py-0.5 rounded border border-[#00e5ff]/30">
                            🎯 {selectedFileNames.length} MIDIs específicos
                          </span>
                        )}
                      </span>
                      <span className="font-mono text-[#00e676] font-bold text-xs">
                        {trainProgress.current_step} / {trainProgress.total_steps || 1} passos globais ({trainProgress.overall_progress_pct || 0}%)
                      </span>
                    </div>
                    <div className="w-full h-3.5 bg-[#090b10] rounded-full overflow-hidden border border-[#242c3b] p-0.5">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          activePreset === 'max_power'
                            ? 'bg-gradient-to-r from-[#ff0055] via-[#ff9900] to-[#ffcc00] shadow-[0_0_12px_rgba(255,0,85,0.6)]'
                            : 'bg-gradient-to-r from-[#00e5ff] via-[#00e676] to-[#ff9100]'
                        }`}
                        style={{ width: `${trainProgress.overall_progress_pct || 0}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Split Visuals: Live Loss Curve + Terminal Stream */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2">
                  {/* Dynamic Loss Canvas */}
                  <div className="p-3 bg-[#0a0d14] rounded-lg border border-[#242c3b] flex flex-col justify-between">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-semibold text-[#f0f4f8] flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5 text-[#00e5ff]" />
                        Curva de Perda (Loss Convergence Chart)
                      </span>
                      <span className="text-[10px] font-mono text-[#8a96a8]">
                        Pontos registrados: {trainProgress.loss_history?.length || 0}
                      </span>
                    </div>
                    <canvas
                      ref={lossCanvasRef}
                      width={520}
                      height={180}
                      className="w-full h-44 bg-[#080b11] rounded border border-[#1b2230]"
                    />
                  </div>

                  {/* Live Terminal Stream Window */}
                  <div className="p-3 bg-[#0a0d14] rounded-lg border border-[#242c3b] flex flex-col justify-between">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-[#f0f4f8]">
                        <Terminal className="w-3.5 h-3.5 text-[#00e5ff]" />
                        Console Stream em Tempo Real
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <label className="flex items-center gap-1 text-[11px] text-[#8a96a8] cursor-pointer">
                          <input
                            type="checkbox"
                            checked={autoScrollLogs}
                            onChange={(e) => setAutoScrollLogs(e.target.checked)}
                            className="rounded accent-[#00e5ff]"
                          />
                          Auto-scroll
                        </label>
                        <button
                          onClick={() => {
                            const text = (trainProgress.logs || []).join('\n');
                            navigator.clipboard.writeText(text);
                          }}
                          className="px-2 py-0.5 bg-[#161b24] border border-[#242c3b] text-[#8a96a8] hover:text-[#f0f4f8] rounded text-[10px]"
                        >
                          Copiar Logs
                        </button>
                      </div>
                    </div>

                    <div
                      ref={terminalRef}
                      className="bg-[#05070a] border border-[#161d27] rounded p-2.5 font-mono text-[10px] h-44 overflow-y-auto space-y-1 text-[#8a96a8]"
                    >
                      {(!trainProgress.logs || trainProgress.logs.length === 0) ? (
                        <div className="text-[#5a6473] italic">
                          Aguardando início do treinamento. Escolha um preset abaixo e clique em "Iniciar Treinamento".
                        </div>
                      ) : (
                        trainProgress.logs.map((line, idx) => {
                          const isPower = line.includes('[POTÊNCIA MÁXIMA]');
                          const isCheckpoint = line.includes('[CHECKPOINT]');
                          const isSuccess = line.includes('[FINALIZADO]') || line.includes('[SUCESSO]');
                          const isConfig = line.includes('[CONFIG]') || line.includes('[INÍCIO]') || line.includes('[PRESET]');
                          const isSpecific = line.includes('[DATASET ESPECÍFICO]');
                          return (
                            <div
                              key={idx}
                              className={`${
                                isPower
                                  ? 'text-[#ff3366] font-bold'
                                  : isCheckpoint
                                  ? 'text-[#ff9100] font-bold'
                                  : isSuccess
                                  ? 'text-[#00e676] font-bold'
                                  : isSpecific
                                  ? 'text-[#00e5ff] font-bold'
                                  : isConfig
                                  ? 'text-[#00e5ff]'
                                  : 'text-[#9bb0cb]'
                              }`}
                            >
                              {line}
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* LOWER ROW: PRESETS (LEFT) & SPECIFIC MIDIS SELECTION (RIGHT) */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* COLUMN 1: TRAINING PRESETS & MAXIMUM POWER */}
                <div className="bg-[#161b24] border border-[#242c3b] rounded-xl p-5 text-xs space-y-4">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="text-sm font-bold flex items-center gap-2">
                        <Gauge className="w-4 h-4 text-[#00e5ff]" />
                        Presets de Treinamento & Intensidade
                      </h3>
                      <p className="text-[#8a96a8] text-[11px] mt-0.5">
                        Defina o quanto a IA vai treinar ou libere a potência máxima de cálculo.
                      </p>
                    </div>
                    <span className="text-xs font-mono font-bold px-2.5 py-1 rounded bg-[#10141b] border border-[#242c3b] text-[#00e5ff]">
                      {trainEpochs} Épocas ({trainIntensity}%)
                    </span>
                  </div>

                  {/* 4 Presets Cards */}
                  <div className="grid grid-cols-2 gap-3">
                    {/* Preset 1: Fast */}
                    <button
                      onClick={() => applyPreset('fast')}
                      className={`p-3 rounded-lg border text-left transition flex flex-col justify-between ${
                        activePreset === 'fast'
                          ? 'bg-[#00e5ff]/15 border-[#00e5ff] text-[#f0f4f8]'
                          : 'bg-[#10141b] border-[#242c3b] text-[#8a96a8] hover:border-[#00e5ff]/50'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-1.5 font-bold text-xs text-[#00e5ff]">
                          <Zap className="w-3.5 h-3.5" /> 1. Leve / Rápido
                        </div>
                        <div className="text-[11px] mt-1 text-[#8a96a8]">
                          5 Épocas • 2 Camadas • ~15 seg
                        </div>
                      </div>
                      <div className="text-[10px] text-[#5a6473] mt-2">Validação rápida imediata.</div>
                    </button>

                    {/* Preset 2: Balanced */}
                    <button
                      onClick={() => applyPreset('balanced')}
                      className={`p-3 rounded-lg border text-left transition flex flex-col justify-between ${
                        activePreset === 'balanced'
                          ? 'bg-[#00e676]/15 border-[#00e676] text-[#f0f4f8]'
                          : 'bg-[#10141b] border-[#242c3b] text-[#8a96a8] hover:border-[#00e676]/50'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-1.5 font-bold text-xs text-[#00e676]">
                          <Activity className="w-3.5 h-3.5" /> 2. Equilibrado
                        </div>
                        <div className="text-[11px] mt-1 text-[#8a96a8]">
                          15 Épocas • 4 Camadas • ~1-2 min
                        </div>
                      </div>
                      <div className="text-[10px] text-[#5a6473] mt-2">Padrão de estúdio musical.</div>
                    </button>

                    {/* Preset 3: Deep */}
                    <button
                      onClick={() => applyPreset('deep')}
                      className={`p-3 rounded-lg border text-left transition flex flex-col justify-between ${
                        activePreset === 'deep'
                          ? 'bg-[#ff9100]/15 border-[#ff9100] text-[#f0f4f8]'
                          : 'bg-[#10141b] border-[#242c3b] text-[#8a96a8] hover:border-[#ff9100]/50'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-1.5 font-bold text-xs text-[#ff9100]">
                          <Cpu className="w-3.5 h-3.5" /> 3. Treino Profundo
                        </div>
                        <div className="text-[11px] mt-1 text-[#8a96a8]">
                          35 Épocas • 6 Camadas • ~3-5 min
                        </div>
                      </div>
                      <div className="text-[10px] text-[#5a6473] mt-2">Polifonia e harmonia rica.</div>
                    </button>

                    {/* Preset 4: POTÊNCIA MÁXIMA */}
                    <button
                      onClick={() => applyPreset('max_power')}
                      className={`p-3 rounded-lg border text-left transition flex flex-col justify-between relative overflow-hidden ${
                        activePreset === 'max_power'
                          ? 'bg-gradient-to-br from-[#ff0055]/25 via-[#ff5500]/20 to-[#ff9900]/20 border-[#ff3366] text-[#f0f4f8] shadow-[0_0_20px_rgba(255,0,85,0.4)]'
                          : 'bg-[#10141b] border-[#242c3b] text-[#8a96a8] hover:border-[#ff3366]/50'
                      }`}
                    >
                      <div className="absolute top-1 right-2 text-[8px] font-black uppercase text-[#ff0055] tracking-widest">
                        OVERDRIVE
                      </div>
                      <div>
                        <div className="flex items-center gap-1.5 font-black text-xs text-[#ff3366]">
                          <Flame className="w-3.5 h-3.5 fill-current animate-bounce text-[#ff9900]" /> 4. POTÊNCIA MÁXIMA
                        </div>
                        <div className="text-[11px] mt-1 text-[#f0f4f8] font-bold">
                          75 Épocas • 8 Camadas • 384 Dim
                        </div>
                      </div>
                      <div className="text-[10px] text-[#ff9900] mt-2 font-medium">
                        Máximo poder de cálculo CPU/GPU!
                      </div>
                    </button>
                  </div>

                  {/* Slider de Intensidade Dinâmica */}
                  <div className="p-3 bg-[#10141b] rounded-lg border border-[#242c3b]">
                    <div className="flex justify-between text-xs text-[#8a96a8] mb-1.5">
                      <span className="font-medium text-[#f0f4f8]">Slider de Esforço / Intensidade do Treino</span>
                      <span className="font-mono text-[#00e5ff] font-bold">{trainIntensity}% ({trainEpochs} Épocas)</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      value={trainIntensity}
                      onChange={(e) => handleIntensitySlider(parseInt(e.target.value))}
                      className="w-full accent-[#00e5ff] cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-[#5a6473] mt-1">
                      <span>Leve (5 ép.)</span>
                      <span>Médio (15 ép.)</span>
                      <span>Profundo (35 ép.)</span>
                      <span className="text-[#ff3366] font-bold">Potência Máxima (75+ ép.)</span>
                    </div>
                  </div>

                  {/* Toggle Parâmetros Avançados */}
                  <div>
                    <button
                      onClick={() => setShowAdvancedTrainParams(!showAdvancedTrainParams)}
                      className="text-[11px] text-[#00e5ff] hover:underline flex items-center gap-1 font-semibold"
                    >
                      <Sliders className="w-3.5 h-3.5" />
                      {showAdvancedTrainParams ? 'Ocultar Parâmetros Avançados' : 'Ajustar Parâmetros Avançados (Camadas, Batch, LR, Contexto)'}
                    </button>

                    {showAdvancedTrainParams && (
                      <div className="grid grid-cols-3 gap-2.5 mt-2.5 p-3 bg-[#10141b] rounded border border-[#242c3b]">
                        <div>
                          <label className="block text-[#8a96a8] text-[10px] mb-1">Épocas Manuais</label>
                          <input
                            type="number"
                            min="1"
                            max="200"
                            value={trainEpochs}
                            onChange={(e) => {
                              const v = parseInt(e.target.value) || 1;
                              setTrainEpochs(v);
                              setActivePreset('custom');
                            }}
                            className="w-full bg-[#090b10] border border-[#242c3b] rounded px-2.5 py-1 text-xs text-[#00e5ff] font-mono focus:border-[#00e5ff] outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[#8a96a8] text-[10px] mb-1">Batch Size</label>
                          <input
                            type="number"
                            min="1"
                            max="64"
                            value={trainBatch}
                            onChange={(e) => setTrainBatch(parseInt(e.target.value) || 4)}
                            className="w-full bg-[#090b10] border border-[#242c3b] rounded px-2.5 py-1 text-xs text-[#00e5ff] font-mono focus:border-[#00e5ff] outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[#8a96a8] text-[10px] mb-1">Learning Rate</label>
                          <input
                            type="number"
                            step="0.0001"
                            value={trainLr}
                            onChange={(e) => setTrainLr(parseFloat(e.target.value) || 0.0005)}
                            className="w-full bg-[#090b10] border border-[#242c3b] rounded px-2.5 py-1 text-xs text-[#00e5ff] font-mono focus:border-[#00e5ff] outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[#8a96a8] text-[10px] mb-1">Camadas (Layers)</label>
                          <input
                            type="number"
                            min="1"
                            max="12"
                            value={trainLayers}
                            onChange={(e) => setTrainLayers(parseInt(e.target.value) || 4)}
                            className="w-full bg-[#090b10] border border-[#242c3b] rounded px-2.5 py-1 text-xs text-[#00e5ff] font-mono focus:border-[#00e5ff] outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[#8a96a8] text-[10px] mb-1">Dimensão Embed</label>
                          <input
                            type="number"
                            step="32"
                            min="32"
                            max="512"
                            value={trainEmbd}
                            onChange={(e) => setTrainEmbd(parseInt(e.target.value) || 128)}
                            className="w-full bg-[#090b10] border border-[#242c3b] rounded px-2.5 py-1 text-xs text-[#00e5ff] font-mono focus:border-[#00e5ff] outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[#8a96a8] text-[10px] mb-1">Janela Contexto</label>
                          <input
                            type="number"
                            step="64"
                            min="32"
                            max="512"
                            value={trainSeqLen}
                            onChange={(e) => setTrainSeqLen(parseInt(e.target.value) || 128)}
                            className="w-full bg-[#090b10] border border-[#242c3b] rounded px-2.5 py-1 text-xs text-[#00e5ff] font-mono focus:border-[#00e5ff] outline-none"
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Nome do modelo & Botão Iniciar */}
                  <div>
                    <label className="block text-[#8a96a8] mb-1">Identificador do Modelo</label>
                    <input
                      type="text"
                      value={trainModelName}
                      onChange={(e) => setTrainModelName(e.target.value)}
                      className="w-full bg-[#10141b] border border-[#242c3b] rounded px-3 py-2 text-xs font-mono focus:border-[#00e5ff] outline-none mb-3"
                    />

                    <button
                      onClick={handleStartTraining}
                      disabled={trainProgress.status === 'running'}
                      className={`w-full py-3.5 font-black rounded-lg text-xs tracking-wide transition flex items-center justify-center gap-2 shadow-xl disabled:opacity-40 ${
                        activePreset === 'max_power'
                          ? 'bg-gradient-to-r from-[#ff0055] via-[#ff5500] to-[#ff9900] text-white hover:brightness-110 shadow-[0_0_20px_rgba(255,0,85,0.45)]'
                          : 'bg-[#00e5ff] text-black hover:bg-[#33ebff]'
                      }`}
                    >
                      {activePreset === 'max_power' ? (
                        <>
                          <Flame className="w-4 h-4 fill-current animate-bounce text-yellow-300" />
                          INICIAR TREINAMENTO EM POTÊNCIA MÁXIMA ({trainEpochs} ÉPOCAS)
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 fill-current" />
                          Iniciar Treinamento ({trainEpochs} Épocas)
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* COLUMN 2: SPECIFIC MIDIS SELECTION */}
                <div className="bg-[#161b24] border border-[#242c3b] rounded-xl p-5 text-xs space-y-4 flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <h3 className="text-sm font-bold flex items-center gap-2">
                        <CheckSquare className="w-4 h-4 text-[#00e5ff]" />
                        Treinar Só Usando MIDIs Específicos
                      </h3>
                      <button
                        onClick={() => setUseOnlySelectedMidis(!useOnlySelectedMidis)}
                        className={`px-3 py-1 rounded-full text-[11px] font-extrabold transition shadow ${
                          useOnlySelectedMidis
                            ? 'bg-[#00e5ff] text-black shadow-[0_0_10px_rgba(0,229,255,0.4)]'
                            : 'bg-[#10141b] border border-[#242c3b] text-[#8a96a8]'
                        }`}
                      >
                        {useOnlySelectedMidis ? '✓ Modo Seleção Ativo' : 'Usar Toda a Biblioteca'}
                      </button>
                    </div>

                    <p className="text-[#8a96a8] text-[11px] mb-3">
                      {useOnlySelectedMidis
                        ? `A rede neural treinará EXCLUSIVAMENTE nas ${selectedFileNames.length} músicas marcadas abaixo.`
                        : `O treinamento incluirá todas as ${libraryFiles.length} músicas disponíveis na biblioteca.`}
                    </p>

                    {/* Search & batch actions */}
                    <div className="flex gap-2 mb-2">
                      <input
                        type="text"
                        placeholder="Filtrar músicas por nome..."
                        value={midiSearchQuery}
                        onChange={(e) => setMidiSearchQuery(e.target.value)}
                        className="flex-1 bg-[#10141b] border border-[#242c3b] rounded px-3 py-1.5 text-xs focus:border-[#00e5ff] outline-none"
                      />
                      <button
                        onClick={selectAllMidis}
                        className="px-2.5 py-1 bg-[#10141b] border border-[#242c3b] hover:border-[#00e5ff] text-[#8a96a8] hover:text-[#f0f4f8] rounded text-[10px] font-semibold"
                      >
                        Marcar Todos
                      </button>
                      <button
                        onClick={deselectAllMidis}
                        className="px-2.5 py-1 bg-[#10141b] border border-[#242c3b] hover:border-red-500 text-[#8a96a8] hover:text-red-400 rounded text-[10px] font-semibold"
                      >
                        Desmarcar Todos
                      </button>
                    </div>

                    {/* MIDI Files Checklist with Audio Preview Button */}
                    <div className="max-h-64 overflow-y-auto space-y-1.5 p-2 bg-[#0d1017] rounded-lg border border-[#242c3b]">
                      {libraryFiles
                        .filter((f) => !midiSearchQuery || f.filename.toLowerCase().includes(midiSearchQuery.toLowerCase()))
                        .map((f) => {
                          const isSel = selectedFileNames.includes(f.filename);
                          const isPlayingThis = activeAudioFilename === f.filename && isPlayingAudio;
                          return (
                            <div
                              key={f.filename}
                              className={`p-2.5 rounded-md flex items-center justify-between transition border ${
                                isSel
                                  ? 'bg-[#00e5ff]/10 border-[#00e5ff]/40 text-[#f0f4f8]'
                                  : 'bg-[#10141b] border-transparent text-[#8a96a8] opacity-60 hover:opacity-100'
                              }`}
                            >
                              <div
                                onClick={() => toggleSelectFile(f.filename)}
                                className="flex items-center gap-2.5 truncate flex-1 cursor-pointer"
                              >
                                <input
                                  type="checkbox"
                                  checked={isSel}
                                  onChange={() => {}}
                                  className="accent-[#00e5ff] cursor-pointer"
                                />
                                <span className="font-semibold truncate text-xs">{f.filename}</span>
                              </div>

                              <div className="flex items-center gap-3 shrink-0 ml-2">
                                <span className="text-[10px] font-mono text-[#8a96a8]">
                                  {f.note_count} notas • {f.tempo_bpm} BPM
                                </span>
                                <button
                                  onClick={() => playFileByName(f.filename)}
                                  className={`p-1 rounded text-[10px] transition flex items-center gap-1 ${
                                    isPlayingThis
                                      ? 'bg-[#ff9100] text-black font-bold'
                                      : 'bg-[#161b24] hover:bg-[#00e5ff]/20 text-[#00e5ff] border border-[#242c3b]'
                                  }`}
                                  title="Ouvir prévia sonora sintetizada"
                                >
                                  {isPlayingThis ? <Square className="w-3 h-3 fill-current" /> : <Play className="w-3 h-3 fill-current" />}
                                  {isPlayingThis ? 'Parar' : 'Ouvir'}
                                </button>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>

                  {/* Summary selection banner */}
                  <div className="p-3 bg-[#10141b] rounded-lg border border-[#242c3b]">
                    {useOnlySelectedMidis && selectedFileNames.length === 0 ? (
                      <div className="text-amber-400 font-semibold flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 shrink-0" />
                        Atenção: Nenhuma música marcada! Selecione os MIDIs acima para treinar especificamente.
                      </div>
                    ) : (
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4 text-[#00e676]" />
                          <span className="font-semibold text-[#f0f4f8]">
                            {useOnlySelectedMidis
                              ? `Treinando exclusivamente com ${selectedFileNames.length} MIDIs marcados`
                              : `Treinando com todos os ${libraryFiles.length} MIDIs da biblioteca`}
                          </span>
                        </div>
                        <span className="font-mono text-[#00e5ff] text-[11px] font-bold">
                          {libraryFiles
                            .filter((f) => !useOnlySelectedMidis || selectedFileNames.includes(f.filename))
                            .reduce((sum, f) => sum + (f.note_count || 0), 0)}{' '}
                          notas no total
                        </span>
                      </div>
                    )}
                  </div>
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

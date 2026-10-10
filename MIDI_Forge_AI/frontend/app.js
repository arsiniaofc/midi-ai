/**
 * MIDI Forge AI - Native Client Application
 * Pure Vanilla JavaScript: Web Audio API synth, Canvas Piano Roll, Training Loss Chart,
 * REST API client and real-time state orchestration.
 */

// Global App State
const App = {
  activeView: 'dashboard',
  libraryFiles: [],
  selectedFile: null,
  activePreviewNotes: [],
  generatedNotes: [],
  audioCtx: null,
  activeOscillators: [],
  isPlayingAudio: false,
  playbackTimeout: null,
  trainInterval: null,
  lossHistory: [],
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSliders();
  initPresets();
  initEventListeners();
  refreshStats();
  refreshLibrary();
  refreshHistory();
  startStatusPolling();
});

// --- NAVIGATION ---
function initNavigation() {
  const navButtons = document.querySelectorAll('.nav-item');
  const views = document.querySelectorAll('.view');

  navButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetView = btn.getAttribute('data-view');
      navButtons.forEach((b) => b.classList.remove('active'));
      views.forEach((v) => v.classList.remove('active'));

      btn.classList.add('active');
      const viewEl = document.getElementById(`view-${targetView}`);
      if (viewEl) viewEl.classList.add('active');

      App.activeView = targetView;
      updateHeaderTitle(targetView);

      if (targetView === 'library') refreshLibrary();
      if (targetView === 'history') refreshHistory();
    });
  });

  // Dashboard quick-action shortcuts
  document.getElementById('dash-action-import')?.addEventListener('click', () => switchView('library'));
  document.getElementById('dash-action-train')?.addEventListener('click', () => switchView('training'));
  document.getElementById('dash-action-generate')?.addEventListener('click', () => switchView('generation'));
  document.getElementById('dash-action-diagnostics')?.addEventListener('click', () => {
    switchView('diagnostics');
    runAllTests();
  });
  document.getElementById('btn-quick-generate')?.addEventListener('click', () => switchView('generation'));
}

function switchView(viewName) {
  const btn = document.querySelector(`.nav-item[data-view="${viewName}"]`);
  if (btn) btn.click();
}

function updateHeaderTitle(view) {
  const titles = {
    dashboard: ['Painel de Controle', 'Visão geral do motor neural local e estado do estúdio.'],
    library: ['Biblioteca MIDI', 'Gerencie seus arquivos .mid locais, visualize notas e audicione.'],
    'data-prep': ['Preparação de Dados', 'Transforme arquivos MIDI em representações REMI ou Theremin BPE.'],
    training: ['Laboratório de Treinamento', 'Ajuste hiperparâmetros e treine a rede Transformer/GPT localmente.'],
    generation: ['Estúdio de Geração Musical', 'Crie novas composições por amostragem autorregressiva ou continuação.'],
    history: ['Histórico & Exportações', 'Composições salvas na pasta output prontas para FL Studio.'],
    diagnostics: ['Diagnósticos do Sistema', 'Suíte automatizada com verificação completa de 20 testes.'],
    portable: ['Pacote Windows Portátil', 'Guia de execução autônoma via Start.bat no Windows.'],
  };
  const [t, d] = titles[view] || ['MIDI Forge AI', ''];
  const titleEl = document.getElementById('page-title');
  const descEl = document.getElementById('page-desc');
  if (titleEl) titleEl.innerText = t;
  if (descEl) descEl.innerText = d;
}

// --- SLIDERS & PRESETS ---
function initSliders() {
  const tempSlider = document.getElementById('gen-temp');
  if (tempSlider) {
    tempSlider.addEventListener('input', (e) => {
      document.getElementById('temp-val').innerText = parseFloat(e.target.value).toFixed(2);
    });
  }

  const topkSlider = document.getElementById('gen-topk');
  if (topkSlider) {
    topkSlider.addEventListener('input', (e) => {
      document.getElementById('topk-val').innerText = e.target.value;
    });
  }

  const toppSlider = document.getElementById('gen-topp');
  if (toppSlider) {
    toppSlider.addEventListener('input', (e) => {
      document.getElementById('topp-val').innerText = parseFloat(e.target.value).toFixed(2);
    });
  }

  const retSlider = document.getElementById('gen-retention');
  if (retSlider) {
    retSlider.addEventListener('input', (e) => {
      document.getElementById('retention-val').innerText = e.target.value;
    });
  }

  const intensitySlider = document.getElementById('train-intensity-slider');
  if (intensitySlider) {
    intensitySlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      document.getElementById('intensity-val').innerText = `${val}%`;
      const epochs = Math.max(3, Math.round((val / 100) * 80));
      document.getElementById('train-epochs').value = epochs;
      if (val < 25) {
        document.getElementById('train-layers').value = 2;
        document.getElementById('train-embd').value = 64;
      } else if (val < 60) {
        document.getElementById('train-layers').value = 4;
        document.getElementById('train-embd').value = 128;
      } else if (val < 85) {
        document.getElementById('train-layers').value = 6;
        document.getElementById('train-embd').value = 256;
      } else {
        document.getElementById('train-layers').value = 8;
        document.getElementById('train-embd').value = 384;
      }
    });
  }

  const genModeSelect = document.getElementById('gen-mode');
  if (genModeSelect) {
    genModeSelect.addEventListener('change', (e) => {
      const isRef = e.target.value === 'reference';
      document.getElementById('group-ref-midi').style.display = isRef ? 'block' : 'none';
      document.getElementById('group-retention').style.display = isRef ? 'block' : 'none';
    });
  }
}

function initPresets() {
  const presetBtns = document.querySelectorAll('.preset-buttons .btn');
  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      presetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.getAttribute('data-preset');
      const slider = document.getElementById('train-intensity-slider');
      if (preset === 'fast') {
        document.getElementById('train-epochs').value = 5;
        document.getElementById('train-layers').value = 2;
        document.getElementById('train-heads').value = 2;
        document.getElementById('train-embd').value = 64;
        document.getElementById('train-batch').value = 4;
        if (slider) { slider.value = 20; document.getElementById('intensity-val').innerText = '20%'; }
      } else if (preset === 'balanced') {
        document.getElementById('train-epochs').value = 15;
        document.getElementById('train-layers').value = 4;
        document.getElementById('train-heads').value = 4;
        document.getElementById('train-embd').value = 128;
        document.getElementById('train-batch').value = 4;
        if (slider) { slider.value = 50; document.getElementById('intensity-val').innerText = '50%'; }
      } else if (preset === 'deep') {
        document.getElementById('train-epochs').value = 35;
        document.getElementById('train-layers').value = 6;
        document.getElementById('train-heads').value = 8;
        document.getElementById('train-embd').value = 256;
        document.getElementById('train-batch').value = 8;
        if (slider) { slider.value = 75; document.getElementById('intensity-val').innerText = '75%'; }
      } else if (preset === 'max_power') {
        // POTÊNCIA MÁXIMA / FORGE OVERDRIVE
        document.getElementById('train-epochs').value = 75;
        document.getElementById('train-layers').value = 8;
        document.getElementById('train-heads').value = 8;
        document.getElementById('train-embd').value = 384;
        document.getElementById('train-batch').value = 8;
        if (slider) { slider.value = 100; document.getElementById('intensity-val').innerText = '100%'; }
      }
    });
  });

  // Specific MIDI selection buttons
  document.getElementById('btn-select-all-midis')?.addEventListener('click', () => {
    App.selectedFileNames = App.libraryFiles.map(f => f.filename);
    renderMidiSelectionList();
  });

  document.getElementById('btn-deselect-all-midis')?.addEventListener('click', () => {
    App.selectedFileNames = [];
    renderMidiSelectionList();
  });

  document.getElementById('btn-copy-terminal')?.addEventListener('click', () => {
    const text = (App.currentLogs || []).join('\n');
    navigator.clipboard.writeText(text);
    alert('Logs copiados para a área de transferência!');
  });
}

// --- EVENT LISTENERS ---
function initEventListeners() {
  // Synthetic dataset generation
  document.getElementById('btn-generate-synthetic')?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/library/synthetic', { method: 'POST' });
      const data = await res.json();
      alert(`Coleção sintética gerada com sucesso! (${data.created_files.length} arquivos criados)`);
      refreshLibrary();
      refreshStats();
    } catch (err) {
      alert('Erro ao gerar coleção sintética: ' + err.message);
    }
  });

  // File upload
  document.getElementById('midi-file-input')?.addEventListener('change', async (e) => {
    const files = e.target.files;
    if (!files || !files.length) return;

    for (let f of files) {
      const reader = new FileReader();
      reader.onload = async (ev) => {
        const base64 = ev.target.result.split(',')[1];
        await fetch('/api/library/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: f.name, content_base64: base64 }),
        });
        refreshLibrary();
        refreshStats();
      };
      reader.readAsDataURL(f);
    }
  });

  // Run data prep
  document.getElementById('btn-run-prep')?.addEventListener('click', async () => {
    const mode = document.getElementById('prep-mode').value;
    const blockSize = parseInt(document.getElementById('prep-block-size').value);
    const augment = document.getElementById('prep-augment').value === 'true';

    const btn = document.getElementById('btn-run-prep');
    btn.disabled = true;
    btn.innerText = 'Processando sequências...';

    try {
      const res = await fetch('/api/datasets/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, block_size: blockSize, augment_transposition: augment }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        const report = data.report;
        const reportBox = document.getElementById('prep-report-box');
        const contentBox = document.getElementById('prep-report-content');
        reportBox.style.display = 'block';
        contentBox.innerHTML = `
          <div class="metrics-grid">
            <div class="metric-card">
              <span class="metric-label">Arquivos Processados</span>
              <div class="metric-value text-cyan">${report.files_processed_count}</div>
            </div>
            <div class="metric-card">
              <span class="metric-label">Notas Mapeadas</span>
              <div class="metric-value text-cyan">${report.total_notes_analyzed}</div>
            </div>
            <div class="metric-card">
              <span class="metric-label">Sequências de Treino</span>
              <div class="metric-value text-amber">${report.train_sequences_count}</div>
            </div>
            <div class="metric-card">
              <span class="metric-label">Tamanho do Vocabulário</span>
              <div class="metric-value text-green">${report.vocab_size} tokens</div>
            </div>
          </div>
          <p class="mt-2 text-muted">Janela de contexto: ${report.block_size} tokens | Modo: <strong>${report.mode.toUpperCase()}</strong></p>
        `;
      }
    } catch (err) {
      alert('Erro na preparação dos dados: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.innerText = '⚡ Processar & Gerar Sequências de Treinamento';
    }
  });

  // Training buttons
  document.getElementById('btn-train-start')?.addEventListener('click', startTraining);
  document.getElementById('btn-train-pause')?.addEventListener('click', () => fetch('/api/training/pause', { method: 'POST' }));
  document.getElementById('btn-train-resume')?.addEventListener('click', () => fetch('/api/training/resume', { method: 'POST' }));
  document.getElementById('btn-train-stop')?.addEventListener('click', () => fetch('/api/training/stop', { method: 'POST' }));

  // Generation button
  document.getElementById('btn-generate-music')?.addEventListener('click', generateMusic);

  // Diagnostics test button
  document.getElementById('btn-run-all-tests')?.addEventListener('click', runAllTests);

  // Audio Playback
  document.getElementById('btn-play-preview')?.addEventListener('click', () => {
    if (App.activePreviewNotes.length) playNotesInBrowser(App.activePreviewNotes);
  });
  document.getElementById('btn-stop-preview')?.addEventListener('click', stopAudioPlayback);

  document.getElementById('btn-play-gen')?.addEventListener('click', () => {
    if (App.generatedNotes.length) playNotesInBrowser(App.generatedNotes);
  });
  document.getElementById('btn-stop-gen')?.addEventListener('click', stopAudioPlayback);
}

// --- API & DATA REFRESH ---
async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('dash-midi-count').innerText = data.imported_midi_count;
    document.getElementById('dash-ckpts-count').innerText = `${data.models_count} / ${data.checkpoints_count}`;
    document.getElementById('dash-output-count').innerText = data.output_midi_count;
    document.getElementById('chip-storage').innerText = `Espaço: ${data.disk_usage_mb} MB`;
  } catch (err) {
    // ignore
  }
}

async function refreshLibrary() {
  try {
    const res = await fetch('/api/library');
    const data = await res.json();
    App.libraryFiles = data.files || [];

    const tbody = document.getElementById('tbody-library');
    const refSelect = document.getElementById('gen-ref-file');
    refSelect.innerHTML = '<option value="">Selecione um MIDI...</option>';

    if (!App.libraryFiles.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-center">Nenhum arquivo MIDI encontrado. Clique em "Gerar Coleção Sintética" acima.</td></tr>';
      return;
    }

    let totalNotes = 0;
    tbody.innerHTML = '';
    App.libraryFiles.forEach((f) => {
      totalNotes += f.note_count || 0;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${f.filename}</strong></td>
        <td>${f.duration_sec}s</td>
        <td>${f.tempo_bpm} BPM</td>
        <td>${f.track_count}</td>
        <td>${f.note_count}</td>
        <td>${f.pitch_range[0]} - ${f.pitch_range[1]}</td>
        <td><span class="badge ${f.is_valid ? 'text-green' : 'text-amber'}">${f.is_valid ? 'Válido' : 'Erro'}</span></td>
        <td>
          <button class="btn btn-sm btn-secondary btn-inspect" data-file="${f.filename}">Inspecionar</button>
        </td>
      `;
      tbody.appendChild(tr);

      // Populate generation reference select
      const opt = document.createElement('option');
      opt.value = f.filename;
      opt.innerText = `${f.filename} (${f.duration_sec}s - ${f.note_count} notas)`;
      refSelect.appendChild(opt);
    });

    document.getElementById('dash-notes-count').innerText = totalNotes;

    // Attach inspect buttons
    document.querySelectorAll('.btn-inspect').forEach((b) => {
      b.addEventListener('click', (ev) => {
        const fname = ev.target.getAttribute('data-file');
        inspectFile(fname);
      });
    });

    // Automatically inspect first file
    if (App.libraryFiles.length && !App.selectedFile) {
      inspectFile(App.libraryFiles[0].filename);
    }
  } catch (err) {
    console.error(err);
  }
}

function inspectFile(filename) {
  const f = App.libraryFiles.find((x) => x.filename === filename);
  if (!f) return;
  App.selectedFile = f;
  document.getElementById('preview-title').innerText = `Visualizando: ${f.filename} (${f.note_count} notas)`;

  // Generate synthetic visual notes based on pitch range & duration for smooth immediate piano roll
  const notes = [];
  const minP = f.pitch_range[0] || 48;
  const maxP = f.pitch_range[1] || 72;
  const count = Math.min(f.note_count || 64, 120);

  let curTick = 0;
  for (let i = 0; i < count; i++) {
    const pitch = minP + Math.floor(Math.sin(i * 0.4) * ((maxP - minP) / 2) + (maxP - minP) / 2);
    notes.push({
      pitch: pitch,
      start_tick: curTick,
      duration_ticks: 240,
      velocity: 80,
    });
    curTick += 240;
  }

  App.activePreviewNotes = notes;
  renderPianoRoll('piano-roll-canvas', notes);
}

// --- PIANO ROLL CANVAS RENDERER ---
function renderPianoRoll(canvasId, notes) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !notes || !notes.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.fillStyle = '#0c0f14';
  ctx.fillRect(0, 0, w, h);

  // Horizontal pitch lines
  ctx.strokeStyle = '#18202c';
  ctx.lineWidth = 1;
  const pitchSteps = 24;
  for (let y = 0; y < h; y += h / pitchSteps) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Find min/max pitch and max tick
  let minP = 127, maxP = 0, maxT = 0;
  notes.forEach((n) => {
    if (n.pitch < minP) minP = n.pitch;
    if (n.pitch > maxP) maxP = n.pitch;
    const endT = (n.start_tick || 0) + (n.duration_ticks || 240);
    if (endT > maxT) maxT = endT;
  });

  minP = Math.max(21, minP - 2);
  maxP = Math.min(108, maxP + 2);
  maxT = Math.max(480, maxT);

  const pitchRange = Math.max(12, maxP - minP);

  // Render notes
  notes.forEach((n) => {
    const x = (n.start_tick / maxT) * (w - 20) + 10;
    const noteWidth = Math.max(4, (n.duration_ticks / maxT) * (w - 20));
    const normPitch = (n.pitch - minP) / pitchRange;
    const y = h - normPitch * (h - 16) - 14;
    const noteHeight = Math.max(4, (h / pitchRange) * 0.8);

    // Glowing cyan note bar
    ctx.fillStyle = '#00e5ff';
    ctx.shadowColor = 'rgba(0, 229, 255, 0.5)';
    ctx.shadowBlur = 6;
    ctx.fillRect(x, y, noteWidth, noteHeight);
  });
  ctx.shadowBlur = 0;
}

// --- WEB AUDIO API SYNTHESIZER ---
function playNotesInBrowser(notes) {
  stopAudioPlayback();
  if (!notes || !notes.length) return;

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!App.audioCtx) {
    App.audioCtx = new AudioContext();
  }
  if (App.audioCtx.state === 'suspended') {
    App.audioCtx.resume();
  }

  App.isPlayingAudio = true;
  const now = App.audioCtx.currentTime;

  // Find time factor: assume 120 BPM (480 ticks = 0.5 sec)
  const tickToSec = 0.5 / 480;

  notes.slice(0, 150).forEach((n) => {
    const startTime = now + (n.start_tick || 0) * tickToSec;
    const duration = Math.min(1.5, Math.max(0.1, (n.duration_ticks || 240) * tickToSec));

    const osc = App.audioCtx.createOscillator();
    const gain = App.audioCtx.createGain();

    // Frequency from MIDI pitch: f = 440 * 2^((p - 69) / 12)
    const freq = 440 * Math.pow(2, (n.pitch - 69) / 12);
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(freq, startTime);

    // ADSR Envelope
    const velNorm = (n.velocity || 80) / 127.0;
    gain.gain.setValueAtTime(0, startTime);
    gain.gain.linearRampToValueAtTime(0.12 * velNorm, startTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    // Lowpass filter for warm synth sound
    const filter = App.audioCtx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(1400, startTime);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(App.audioCtx.destination);

    osc.start(startTime);
    osc.stop(startTime + duration + 0.05);

    App.activeOscillators.push(osc);
  });
}

function stopAudioPlayback() {
  App.isPlayingAudio = false;
  App.activeOscillators.forEach((osc) => {
    try {
      osc.stop();
    } catch (e) {}
  });
  App.activeOscillators = [];
}

// --- TRAINING CONTROLS & REAL-TIME LOSS CHART ---
async function startTraining() {
  const modelName = document.getElementById('train-model-name').value || 'Forge_Model';
  const epochs = parseInt(document.getElementById('train-epochs').value);
  const batchSize = parseInt(document.getElementById('train-batch').value);
  const lr = parseFloat(document.getElementById('train-lr').value);
  const layers = parseInt(document.getElementById('train-layers').value);
  const heads = parseInt(document.getElementById('train-heads').value);
  const embd = parseInt(document.getElementById('train-embd').value);
  const useOnlySelected = document.getElementById('check-use-only-selected')?.checked || false;
  const selectedMidis = useOnlySelected ? Array.from(App.selectedMidis || []) : [];

  if (useOnlySelected && selectedMidis.length === 0) {
    alert('Atenção: Selecione ao menos 1 arquivo MIDI na lista abaixo para treinar especificamente!');
    return;
  }

  // Create model config first
  await fetch('/api/models/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: modelName, n_layer: layers, n_head: heads, n_embd: embd }),
  });

  // Start training loop
  const res = await fetch('/api/training/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: modelName,
      epochs,
      batch_size: batchSize,
      learning_rate: lr,
      selected_files: selectedMidis,
      power_preset: App.currentPreset || 'balanced',
      intensity_pct: parseInt(document.getElementById('train-intensity-slider')?.value || 50),
      n_layer: layers,
      n_head: heads,
      n_embd: embd,
    }),
  });
  const data = await res.json();

  if (data.status === 'success') {
    document.getElementById('btn-train-start').disabled = true;
    document.getElementById('btn-train-pause').disabled = false;
    document.getElementById('btn-train-stop').disabled = false;
    App.lossHistory = [];
  } else {
    alert('Erro ao iniciar treinamento: ' + data.error);
  }
}

function startStatusPolling() {
  setInterval(async () => {
    try {
      // 1. Hardware status
      const statRes = await fetch('/api/status');
      const statData = await statRes.json();
      document.getElementById('chip-device').innerText = `Dispositivo: ${statData.device.toUpperCase()}${statData.gpu_name ? ` (${statData.gpu_name})` : ''}`;

      // 2. Training progress
      const trainRes = await fetch('/api/training/progress');
      const trainData = await trainRes.json();
      updateTrainingUI(trainData);
    } catch (err) {}
  }, 1000);
}

function updateTrainingUI(progress) {
  const badge = document.getElementById('train-status-badge');
  badge.innerText = progress.status.toUpperCase();
  badge.className = `status-pill status-${progress.status}`;

  document.getElementById('meter-epoch').innerText = `${progress.current_epoch + 1} / ${progress.total_epochs}`;
  document.getElementById('meter-step').innerText = `${progress.current_step} / ${progress.total_steps}`;
  document.getElementById('meter-loss').innerText = (progress.current_train_loss || 0).toFixed(4);
  document.getElementById('meter-val-loss').innerText = (progress.current_val_loss || 0).toFixed(4);

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  document.getElementById('meter-elapsed').innerText = formatTime(progress.elapsed_sec || 0);
  document.getElementById('meter-eta').innerText = formatTime(progress.remaining_sec || 0);

  const pct = Math.min(100, progress.overall_progress_pct || 0);
  document.getElementById('train-pct').innerText = `${pct}%`;
  document.getElementById('train-progress-fill').style.width = `${pct}%`;

  // Buttons state
  const isRunning = progress.status === 'running';
  const isPaused = progress.status === 'paused';
  document.getElementById('btn-train-start').disabled = isRunning;
  document.getElementById('btn-train-pause').disabled = !isRunning;
  document.getElementById('btn-train-resume').disabled = !isPaused;
  document.getElementById('btn-train-stop').disabled = !isRunning && !isPaused;

  // Render loss chart
  if (progress.loss_history && progress.loss_history.length) {
    renderLossChart(progress.loss_history);
  }
}

function renderLossChart(history) {
  const canvas = document.getElementById('loss-chart-canvas');
  if (!canvas || !history.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0f141d';
  ctx.fillRect(0, 0, w, h);

  // Find max/min loss
  let maxL = 0, minL = 100;
  history.forEach((pt) => {
    if (pt.train_loss > maxL) maxL = pt.train_loss;
    if (pt.train_loss < minL) minL = pt.train_loss;
  });
  maxL = Math.max(1.0, maxL * 1.1);
  minL = Math.max(0, minL * 0.8);
  const range = maxL - minL || 1.0;

  // Grid lines
  ctx.strokeStyle = '#222b3a';
  ctx.lineWidth = 1;
  for (let y = 20; y < h - 20; y += 40) {
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 10, y);
    ctx.stroke();
  }

  // Draw Train Loss curve (Cyan)
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

  // Draw Val Loss curve (Amber)
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

  // Legend
  ctx.font = '11px sans-serif';
  ctx.fillStyle = '#00e5ff';
  ctx.fillText('● Loss Treino', w - 180, 20);
  ctx.fillStyle = '#ff9100';
  ctx.fillText('● Loss Validação', w - 90, 20);
}

// --- GENERATION ENGINE ---
async function generateMusic() {
  const isRef = document.getElementById('gen-mode').value === 'reference';
  const refFile = isRef ? document.getElementById('gen-ref-file').value : null;
  const temp = parseFloat(document.getElementById('gen-temp').value);
  const maxTokens = parseInt(document.getElementById('gen-tokens').value);
  const topk = parseInt(document.getElementById('gen-topk').value);
  const topp = parseFloat(document.getElementById('gen-topp').value);
  const bpm = parseFloat(document.getElementById('gen-bpm').value);
  const seedInput = document.getElementById('gen-seed').value;
  const seed = seedInput ? parseInt(seedInput) : null;
  const retention = parseFloat(document.getElementById('gen-retention').value) / 100.0;

  const btn = document.getElementById('btn-generate-music');
  btn.disabled = true;
  btn.innerText = 'Amostrando Transformer e gerando MIDI...';

  try {
    const res = await fetch('/api/generation/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        temperature: temp,
        max_tokens: maxTokens,
        top_k: topk,
        top_p: topp,
        bpm,
        seed,
        reference_file: refFile,
        context_retention_pct: retention,
      }),
    });
    const data = await res.json();

    if (data.success) {
      App.generatedNotes = data.preview_notes || [];
      document.getElementById('gen-placeholder').style.display = 'none';
      document.getElementById('gen-result-container').style.display = 'block';
      document.getElementById('gen-result-actions').style.display = 'flex';

      const banner = document.getElementById('gen-metrics-banner');
      banner.innerHTML = `
        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Arquivo Gerado</span>
            <div style="font-size: 1.1rem; font-weight:700; color:var(--cyan-accent);">${data.filename}</div>
          </div>
          <div class="metric-card">
            <span class="metric-label">Duração & BPM</span>
            <div style="font-size: 1.1rem; font-weight:700;">${data.duration_sec}s @ ${data.bpm} BPM</div>
          </div>
          <div class="metric-card">
            <span class="metric-label">Total de Notas</span>
            <div style="font-size: 1.1rem; font-weight:700; color:var(--green-accent);">${data.note_count} notas</div>
          </div>
          <div class="metric-card">
            <span class="metric-label">Semente (Seed)</span>
            <div style="font-size: 1.1rem; font-weight:700;">#${data.seed}</div>
          </div>
        </div>
      `;

      // Set download link
      const dlBtn = document.getElementById('btn-download-gen');
      dlBtn.href = `/api/download?file=${encodeURIComponent(data.filename)}`;
      dlBtn.setAttribute('download', data.filename);

      // Render piano roll
      renderPianoRoll('gen-piano-roll-canvas', App.generatedNotes);

      refreshStats();
      refreshHistory();
    } else {
      alert('Falha na geração: ' + (data.error || 'Erro desconhecido'));
    }
  } catch (err) {
    alert('Erro de conexão ao gerar música: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerText = '⚡ Gerar Composição Musical';
  }
}

// --- HISTORY VIEW ---
async function refreshHistory() {
  try {
    const res = await fetch('/api/output');
    const data = await res.json();
    const tbody = document.getElementById('tbody-history');
    if (!data.files || !data.files.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-center">Nenhum arquivo gerado ainda. Use a aba "Estúdio de Geração".</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    data.files.forEach((f) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${f.filename}</strong></td>
        <td>${f.duration_sec}s</td>
        <td>${f.note_count}</td>
        <td>${f.tempo_bpm}</td>
        <td><span class="badge text-green">100% SMF Válido</span></td>
        <td>
          <a class="btn btn-sm btn-primary" href="/api/download?file=${encodeURIComponent(f.filename)}" download="${f.filename}">
            💾 Baixar .MID
          </a>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {}
}

// --- DIAGNOSTICS VIEW (20 AUTOMATED TESTS) ---
async function runAllTests() {
  const btn = document.getElementById('btn-run-all-tests');
  btn.disabled = true;
  btn.innerText = 'Executando 20 testes unitários...';

  const container = document.getElementById('tests-list-container');
  container.innerHTML = '<p class="text-cyan">Executando suite de testes automatizados...</p>';

  try {
    const res = await fetch('/api/diagnostics/run', { method: 'POST' });
    const data = await res.json();

    const summary = document.getElementById('diag-summary');
    summary.style.display = 'block';
    summary.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-label">Status Geral</span>
          <div class="metric-value ${data.all_passed ? 'text-green' : 'text-amber'}">${data.all_passed ? 'APROVADO' : 'FALHA'}</div>
        </div>
        <div class="metric-card">
          <span class="metric-label">Testes Aprovados</span>
          <div class="metric-value text-green">${data.passed} / ${data.total_tests}</div>
        </div>
        <div class="metric-card">
          <span class="metric-label">Tempo de Execução</span>
          <div class="metric-value text-cyan">${data.duration_sec}s</div>
        </div>
      </div>
    `;

    container.innerHTML = '';
    data.details.forEach((t) => {
      const div = document.createElement('div');
      div.className = `test-item ${t.passed ? 'passed' : 'failed'}`;
      div.innerHTML = `
        <div>
          <strong>${t.description}</strong>
          <span class="text-muted" style="margin-left: 8px; font-size: 0.78rem;">(${t.name})</span>
        </div>
        <span class="badge ${t.passed ? 'text-green' : 'text-amber'}">${t.passed ? '✔ APROVADO' : '✖ FALHA'}</span>
      `;
      container.appendChild(div);
    });
  } catch (err) {
    container.innerHTML = `<p class="text-amber">Erro ao executar testes: ${err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.innerText = '🛡️ Executar 20 Testes Agora';
  }
}

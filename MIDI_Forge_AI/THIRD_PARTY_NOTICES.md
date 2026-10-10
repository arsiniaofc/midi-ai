# Terceiros, Atribuições e Licenças — MIDI Forge AI

Este aplicativo integra técnicas, arquiteturas e algoritmos originários dos seguintes projetos de código aberto:

---

## 1. aiMIDI
- **Autor / Repositório:** jimsweb (https://github.com/jimsweb/aiMIDI)
- **Licença:** MIT License
- **Componentes adaptados:**
  - Tokenização inspirada em REMI (Revamped MIDI)
  - Quantização temporal em compassos (Bars) e posições métricas (Positions)
  - Agrupamento de velocidades MIDI em bins discretos
  - Tratamento de durações de notas e tokens de controle BOS/EOS/PAD
  - Estratégia de agendamento de taxa de aprendizado e clipping de gradientes

---

## 2. Theremin MIDI
- **Autor / Repositório:** robrohan (https://github.com/robrohan/theremin-midi)
- **Licença:** MIT License
- **Componentes adaptados:**
  - Serialização de eventos musicais em cadeias de texto estruturadas
  - Codificação por pares de bytes (BPE - Byte Pair Encoding) sobre padrões de notas
  - Arquitetura autorregressiva GPT para geração musical
  - Condicionamento de geração a partir de contexto inicial de MIDI de referência
  - Decodificação de sequências de símbolos de volta para eventos Note-On / Note-Off

---

## 3. minGPT
- **Autor / Repositório:** Andrej Karpathy (https://github.com/karpathy/minGPT)
- **Licença:** MIT License
- **Componentes adaptados:**
  - Blocos de atenção causal multi-cabeça (CausalSelfAttention)
  - Estrutura de Transformer autorregressivo Pre-LN com ativação GELU
  - Amostragem com filtragem Top-K e Núcleo Top-P (Nucleus Sampling)

---

## 4. Mido
- **Autor / Repositório:** Ole Martin Bjørndalen (https://github.com/mido/mido)
- **Licença:** MIT License
- **Uso:** Leitura e gravação de arquivos Standard MIDI Files (SMF Tipo 0 e Tipo 1).

---

## 5. PyTorch
- **Organização:** The Linux Foundation / PyTorch Authors (https://github.com/pytorch/pytorch)
- **Licença:** BSD-3-Clause
- **Uso:** Framework de tensores, cálculo de gradientes automáticos e otimizadores.

================================================================================
                    MIDI FORGE AI — GERADOR E TREINADOR LOCAL
================================================================================

O MIDI Forge AI é um estúdio desktop completo e 100% local para Windows projetado
para treinar modelos Transformer/GPT em arquivos MIDI e gerar novas composições
musicais originais.

Baseado na integração dos projetos:
- aiMIDI (jimsweb): Tokenização REMI, quantização temporal em barras/posições,
  bins de velocidade e escalonamento de taxa de aprendizado.
- Theremin MIDI (robrohan): Serialização textual de eventos, Byte Pair Encoding
  (BPE), condicionamento autorregressivo por MIDI de referência.

================================================================================
GUIA RÁPIDO DE INSTALAÇÃO E USO NO WINDOWS
================================================================================

1. EXTRAÇÃO:
   Extraia a pasta completa "MIDI_Forge_AI" em qualquer local do seu computador
   (por exemplo: C:\MIDI_Forge_AI ou na sua Área de Trabalho).

2. PREPARAÇÃO INICIAL (PRIMEIRA VEZ):
   Dê um duplo-clique no arquivo:
   >> Setup_First_Run.bat <<
   O assistente irá configurar o ambiente virtual independente (.venv), instalar
   o PyTorch e o Mido e executar os testes de integridade locais.

3. INICIALIZAÇÃO:
   Dê um duplo-clique no arquivo:
   >> Start.bat <<
   O servidor local seguro (127.0.0.1:8080) será iniciado e a interface será
   aberta automaticamente no seu navegador padrão.

4. IMPORTAÇÃO DE COLEÇÃO MIDI:
   Na aba "Biblioteca MIDI", você pode:
   - Carregar seus arquivos .mid ou .midi (arraste e solte ou use o botão de envio).
   - Utilizar a coleção sintética de 8 MIDIs de exemplo já incluída (clássico,
     synthwave, lofi, blues, trance, ambient, funk e orquestral).
   - O aplicativo calcula automaticamente duração, andamento, número de notas e
     detecta duplicatas e arquivos corrompidos.

5. PREPARAÇÃO DOS DADOS:
   Na aba "Preparação de Dados":
   - Escolha o modo de representação:
     * REMI (aiMIDI): ideal para grades rítmicas e métricas consistentes.
     * Theremin BPE: ideal para transições melódicas fluidas e acordes.
   - Configure o tamanho da janela de contexto (ex: 128 ou 256 tokens).
   - Ative a transposição automática (-2 a +2 semitons) para ampliar seu dataset.
   - Clique em "Processar e Gerar Sequências".

6. TREINAMENTO DO MODELO:
   Na aba "Treinamento":
   - Escolha o número de camadas (layers), cabeças de atenção (heads) e dimensões.
   - Ajuste épocas (ex: 10 a 50) e tamanho do lote (batch size).
   - O perfil conservador padrão é otimizado para rodar em CPU convencional
     sem sobrecarregar a memória do computador. Se possuir GPU NVIDIA com CUDA,
     o PyTorch a utilizará automaticamente.
   - Clique em "Iniciar Treinamento". O gráfico de perda (Loss) e o progresso
     são atualizados em tempo real.

7. SALVAMENTO E CHECKPOINTS:
   Checkpoints são salvos automaticamente na pasta "checkpoints/" a cada época.
   Eles contêm os pesos da rede, estado do otimizador e métricas de perda.

8. PAUSA E RETOMADA:
   Você pode pausar o treinamento a qualquer momento clicando em "Pausar",
   ou parar com segurança em "Parar". Para continuar treinando de onde parou,
   basta selecionar o checkpoint desejado na lista e clicar em "Retomar".

9. ESTÚDIO DE GERAÇÃO:
   Na aba "Geração":
   - Geração Incondicional: gera música a partir de uma semente aleatória.
   - Geração Condicionada por MIDI de Referência: selecione um MIDI da biblioteca
     e o percentual de contexto a reter (ex: 30%). O modelo irá aprender o motivo
     inicial e compor a continuação original!
   - Ajuste Temperatura (criatividade), Top-K e Top-P.
   - Visualize a partitura / piano roll interativo e audicione no próprio navegador!

10. EXPORTAÇÃO PARA DAWS (FL STUDIO, ABLETON, REAPER):
    Clique em "Exportar MIDI". O arquivo gerado é gravado na pasta "output/" e
    baixado diretamente pelo navegador.
    O arquivo gerado segue rigorosamente o padrão Standard MIDI File (SMF Tipo 0/1)
    com eventos Note-On e Note-Off limpos, compatível de fábrica com FL Studio.

11. RECUPERAÇÃO EM CASO DE FALHA:
    - Se a janela for fechada acidentalmente, seus dados e checkpoints salvos
      permanecem intactos na pasta.
    - Se a porta 8080 estiver ocupada, encerre processos antigos pelo
      Gerenciador de Tarefas do Windows ou reinicie com Start.bat.
    - Para restaurar o ambiente do zero, basta excluir a pasta ".venv" e executar
      "Setup_First_Run.bat" novamente.

================================================================================
PRIVACIDADE TOTAL: NENHUM DADO SAI DO SEU COMPUTADOR.
TREINAMENTO E GERAÇÃO 100% LOCAIS E DESCONECTADOS DA INTERNET.
================================================================================

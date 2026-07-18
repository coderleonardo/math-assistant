# Initial Phases

### Fase 1: Aquisição de Dados e Preparação do Dataset

**Status:** Concluído ✅

* ✅ **Extração via API:** Uso da API do StackExchange para extrair perguntas e respostas resolvidas do fórum de matemática (MathOverflow/Math Stack Exchange).
* ✅ **Filtros Avançados:** Aplicação do filtro personalizado para obter o corpo de texto (body) completo das perguntas e a respectiva resposta aceita.
* ✅ **Limpeza de Dados:** Parseamento do HTML utilizando `BeautifulSoup`, removendo tags visuais, mas preservando rigorosamente as fórmulas em MathJax/LaTeX.
* ✅ **Formatação ShareGPT/Messages:** Estruturação do dataset final no formato JSONL com mensagens divididas em `system`, `user` e `assistant` para compatibilidade nativa com o chat template.

### Fase 2: Fine-Tuning do Modelo de Linguagem (QLoRA)

**Status:** Concluído ✅

* ✅ **Setup do Modelo Base:** Carregamento do `Qwen3-4B-Thinking-2507` (que possui nativamente 4 bilhões de parâmetros ) utilizando quantização em 4-bits.


* ✅ **Configuração dos Adaptadores:** Aplicação do LoRA focando nas matrizes de atenção do modelo para otimizar os pesos com baixo custo de VRAM.
* ✅ **Resolução de Conflitos de Memória:** Configuração do `device_map={"": 0}` para forçar o treinamento em uma única GPU, contornando erros de paralelismo.
* ✅ **Treinamento SFT:** Execução do `SFTTrainer` (biblioteca TRL) adaptando o modelo à nossa base de conhecimento matemático e salvando os adaptadores treinados.

### Fase 3: Preparação para Deploy e Inferência Core

**Status:** Em transição 🚧

* ✅ **Fusão dos Pesos (Merge & Unload):** Carregamento do modelo base em precisão 16-bits e fusão com os adaptadores LoRA para gerar um modelo autônomo unificado.
* 🚧 **Motor de Inferência (API Rest):** Hospedar a pasta do modelo fundido usando motores C++ e CUDA puro (como **vLLM** ou **SGLang** para produção ), ou ferramentas leves como Ollama e LMStudio.


* 🚧 **Parametrização Otimizada de Raciocínio:** * Configurar a API com Temperature = `0.6`, Top-P = `0.95`, Top-K = `20`.


* Definir o `max_new_tokens` para `32.768` em consultas gerais, ou até `81.920` para problemas matemáticos altamente complexos.





### Fase 4: O "Cérebro" do Agente (Orquestração e Memória)

**Status:** Próximos Passos 🔮

* 🔮 **Memória Longa (Knowledge Graph / RAG):** Implementar o motor de *Retriever* no seu banco de Grafos para buscar conceitos semânticos e injetá-los no prompt do usuário (aproveitando o contexto nativo de até 262.144 tokens do Qwen3 ).


* 🔮 **Memória Curta (Sessões de Chat):** Gerenciar conversas multi-turn. **Atenção:** O histórico repassado nas rodadas seguintes deve incluir *apenas a resposta final*, omitindo o raciocínio interno (`<think>`).


* 🔮 **Cache Semântico:** Uso de Redis para guardar respostas de cálculos idênticos já resolvidos, economizando VRAM.

### Fase 5: Ferramentas (Tool Calling)

**Status:** Avançado 🔮

* 🔮 **Agente de Ferramentas Matemáticas:** Utilizar o framework `Qwen-Agent` para facilitar a codificação de parsers e habilitar o "Tool Calling" nativo do modelo.


* 🔮 **Integração Externa:** Conectar o agente a uma calculadora Python isolada (Sandbox) ou API do Wolfram Alpha para cálculos determinísticos perfeitos.

### Fase 6: Otimização Extrema e Destilação (Model Distillation) 🆕

**Status:** Avançado / Escala 🔮

* 🔮 **Geração de Dataset Sintético (Teacher-Student):** Usar o seu agente `Qwen3-4B` já treinado (ou um modelo "Teacher" maior) para gerar milhares de soluções passo-a-passo (com raciocínio e resposta final) para novos problemas matemáticos.
* 🔮 **Treinamento do Modelo "Aluno":** Pegar um modelo base bem menor e mais rápido (ex: um modelo de 0.5B ou 1.5B parâmetros) e fazer o SFT (Supervised Fine-Tuning) exclusivamente com as saídas geradas pelo seu agente 4B.
* 🔮 **Avaliação de Custo/Benefício:** Substituir o modelo de 4B pelo modelo destilado menor em chamadas de API mais simples, reduzindo drasticamente o tempo de resposta (latência) e o custo de hospedagem na nuvem, preservando a lógica de raciocínio aprendida do modelo maior.
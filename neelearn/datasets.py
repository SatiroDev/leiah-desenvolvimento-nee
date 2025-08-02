from sklearn.utils import Bunch

def load_nee_assessments():
    data = [
        "Qual é o nome do protagonista da história?",                         # literal
        "Explique com suas palavras o que motivou o personagem a agir assim.", # discursiva
        "O que o autor quis dizer com a expressão 'coração de pedra'?",       # interpretativa
        "Aponte duas críticas sociais presentes no texto.",                   # crítica
        "Qual a alternativa correta sobre o tema abordado?",                  # objetiva
        "O que acontece primeiro na sequência dos fatos?",                    # simples
        "Relacione o final da história com o comportamento dos personagens.", # complexa
        "Qual é o nome do protagonista da história narrada no primeiro parágrafo?",                             # literal
        "Explique com suas próprias palavras por que o autor escolheu esse desfecho para a história.",          # discursiva
        "O que o autor quis dizer com a metáfora 'o tempo escorria como areia entre os dedos'?",                # interpretativa
        "Aponte e comente duas críticas sociais implícitas no texto lido.",                                     # crítica
        "Assinale a alternativa que melhor resume o tema central do texto.",                                    # objetiva
        "Qual foi a primeira ação tomada pelo personagem após receber a notícia?",                              # simples
        "Analise a relação entre o comportamento do personagem principal e o contexto histórico da narrativa.",  # complexa
        "Qual é o nome do personagem que aparece logo no início do texto?",                     # literal
        "Em que local se passa a maior parte da história?",                                     # literal
        
        "Explique, com suas palavras, o motivo da mudança de atitude do protagonista.",         # discursiva
        "Descreva o impacto que o conflito central causou no desfecho da narrativa.",           # discursiva

        "O que o autor quis dizer com a expressão 'as paredes falavam com ela'?",               # interpretativa
        "Interprete o significado simbólico do espelho na construção do personagem.",           # interpretativa

        "Quais críticas sociais estão presentes na abordagem do autor?",                        # crítica
        "Analise os elementos que revelam uma visão crítica sobre a desigualdade no texto.",    # crítica

        "Assinale a alternativa que melhor expressa o tema principal do texto.",                # objetiva
        "Qual das opções abaixo corresponde à opinião expressa pelo narrador?",                 # objetiva

        "Qual foi a primeira atitude da personagem após o telefonema?",                         # simples
        "Qual é o clima predominante no trecho analisado?",                                     # simples

        "Relacione o comportamento do personagem ao contexto político apresentado.",            # complexa
        "Compare a evolução emocional do personagem com os eventos históricos mencionados." ,    # complexa
        "Qual o nome da cidade onde a história começa?",                                         # literal
        "Qual é a profissão do narrador mencionada no segundo parágrafo?",                      # literal

        "Descreva, com suas palavras, a relação entre os dois personagens principais.",          # discursiva
        "Explique o que levou o personagem a tomar uma decisão tão drástica no final.",          # discursiva

        "O que representa a chuva constante ao longo da narrativa?",                             # interpretativa
        "Como podemos interpretar a presença do relógio quebrado no contexto do conto?",         # interpretativa

        "De que maneira o texto aborda o preconceito de forma implícita?",                       # crítica
        "Qual mensagem crítica o autor transmite ao retratar a desigualdade social?",            # crítica

        "Qual das alternativas apresenta a ideia central do segundo parágrafo?",                 # objetiva
        "Assinale a opção correta sobre a opinião do narrador acerca da infância.",              # objetiva

        "Qual foi a reação da personagem ao encontrar a carta?",                                 # simples
        "Quem era o visitante misterioso mencionado no terceiro parágrafo?",                     # simples

        "Compare os sentimentos do personagem no início e no fim da história.",                  # complexa
        "Analise como o ambiente influencia as decisões do protagonista ao longo do texto.",      # complexa
        "O que o autor quis dizer ao usar a palavra 'prisão' em um sentido simbólico?",
        "Como podemos compreender a metáfora usada no título do texto?",
        "Qual a interpretação possível para a atitude silenciosa do personagem?",
        "Qual é o nome do personagem principal?",
        "Onde se passa a história?",
        "Quem é o autor do texto?",
        "Qual é a cor do carro mencionado?",
        "Quantos personagens aparecem na narrativa?",
        "Quando aconteceu o evento principal?",
        "Qual é a profissão do protagonista?",
        "Qual foi a primeira ação do personagem?",
        "Quem falou a primeira frase no diálogo?",
        "Qual o objeto que o personagem encontrou?","Qual era a relação entre o personagem principal e sua mãe?",                             # complexa
        "Explique por que o personagem decidiu esconder a verdade.",                              # discursiva
        "O que o autor quis sugerir ao usar a imagem do mar revolto?",                             # interpretativa
        "Quantos dias se passaram entre o início e o fim da narrativa?",                          # simples
        "Assinale a alternativa que indica corretamente o clímax da história.",                   # objetiva
        "Aponte duas críticas ao sistema educacional presentes no texto.",                        # crítica
        "Qual é o nome do animal de estimação do personagem principal?",                           # literal
        "Descreva com suas palavras o ambiente onde a história se desenrola.",                    # discursiva
        "Interprete o uso da escuridão como símbolo no final do conto.",                          # interpretativa
        "Qual foi a consequência imediata da decisão do personagem?",                             # simples
        "Analise como os conflitos familiares influenciaram o desfecho da história.",             # complexa
        "Assinale a alternativa correta sobre a opinião do narrador em relação ao protagonista.", # objetiva
        "O que o autor critica ao retratar o abandono dos idosos?",                               # crítica
        "Qual a profissão do pai do protagonista mencionada no início da história?",              # literal
        "Explique por que o autor escolheu narrar a história em primeira pessoa.",                 # discursiva
        "Qual personagem representa a figura da autoridade na narrativa?",                        # literal
        "O que levou a personagem a mudar sua opinião ao longo da história?",                     # discursiva
        "Como interpretar a expressão 'o silêncio gritava' utilizada no texto?",                  # interpretativa
        "Quais aspectos sociais são criticados por meio da fala do personagem secundário?",       # crítica
        "Marque a opção que apresenta um resumo adequado da narrativa.",                          # objetiva
        "O que a personagem fez logo após o desaparecimento da irmã?",                            # simples
        "Analise o impacto da guerra na trajetória dos personagens principais.",                  # complexa
        "Qual era o objeto que a personagem carregava consigo o tempo todo?",                     # literal
        "Explique por que o narrador optou por omitir certos acontecimentos.",                    # discursiva
        "Qual o sentido simbólico das janelas fechadas ao longo do conto?",                       # interpretativa
        "Aponte duas críticas ao sistema de saúde pública contidas no texto.",                    # crítica
        "Assinale a alternativa correta sobre o gênero textual da narrativa lida.",               # objetiva
        "O que aconteceu com o personagem após o acidente de carro?",                             # simples
        "Relacione o comportamento do personagem com o sentimento de culpa evidenciado no final.",# complexa
        "Qual é o nome do bairro onde se passa a maior parte da história?",                       # literal
        "Descreva a importância do diário para a compreensão da personalidade da personagem.",    # discursiva
        "Interprete a metáfora 'seus olhos eram tempestades'.",                                   # interpretativa
        "Que críticas sociais podem ser percebidas nas falas da narradora?",                      # crítica
        "Qual das alternativas apresenta a melhor explicação para o conflito central?",           # objetiva
        "O que a personagem fez primeiro ao chegar na escola?",                                  # simples
        "Qual a motivação do personagem principal para tomar a decisão final?",            # discursiva
        "Interprete o simbolismo da luz no desfecho da narrativa.",                        # interpretativa
        "Cite duas críticas sociais presentes nas atitudes dos personagens coadjuvantes.", # crítica
        "Assinale a alternativa que melhor representa o tema principal do texto.",        # objetiva
        "O que acontece logo após o diálogo entre os protagonistas?",                     # simples
        "Compare o comportamento do personagem antes e depois do conflito central.",      # complexa
        "Quem é o narrador da história e qual sua importância?",                          # literal
        "Descreva o impacto emocional do evento traumático no personagem principal.",     # discursiva
        "Explique o significado da metáfora 'coração de pedra' usada no texto.",          # interpretativa
        "Quais críticas sociais são evidentes no ambiente descrito na narrativa?",        # crítica
        "Qual alternativa apresenta o melhor resumo do enredo?",                          # objetiva
        "Qual foi a primeira reação do personagem após receber a notícia?",               # simples
        "Analise a relação entre o contexto histórico e as ações do personagem principal.",# complexa
        "Qual é o nome da cidade onde se passa a maior parte da história?",               # literal
        "Explique, com suas palavras, a transformação do protagonista ao longo da narrativa.", # discursiva
        "Interprete o uso da metáfora 'as sombras do passado' no texto.",                 # interpretativa
        "Quais críticas sociais podem ser percebidas na fala do personagem secundário?",  # crítica
        "Assinale a alternativa correta sobre o ponto de vista narrativo.",               # objetiva
        "O que ocorre imediatamente após o evento que desencadeia o conflito?",           # simples
        "Compare os sentimentos do personagem no início e no fim da narrativa.",          # complexa
        "Quem é o personagem que simboliza a esperança na história?",                    # literal
        "Descreva a importância do diário encontrado pelo protagonista.",                 # discursiva
        "Explique o significado da metáfora 'tempestade interna' no desenvolvimento do personagem.", # interpretativa
        "Quais aspectos sociais são criticados na descrição da cidade?",                  # crítica
        "Assinale a alternativa que apresenta o tema central do texto.",                  # objetiva
        "Qual foi a primeira atitude do personagem após o conflito?",                      # simples
        "Analise o desenvolvimento emocional do protagonista ao longo da narrativa.",     # complexa
        "Quem é o personagem que representa a autoridade na história?",                   # literal
        "Descreva com suas palavras o ambiente onde a narrativa se desenrola.",            # discursiva
        "Interprete o significado da metáfora 'caminho sem saída' no contexto do texto.",  # interpretativa
        "Quais críticas sociais são abordadas por meio dos diálogos dos personagens?",     # crítica
        "Assinale a alternativa que melhor sintetiza o tema central da história.",        # objetiva
        "O que acontece imediatamente após a chegada do visitante misterioso?",           # simples
        "Compare o comportamento dos personagens principais em momentos-chave da trama.", # complexa
        "Qual é o nome do bairro onde se passa a maior parte da narrativa?",              # literal
        "Explique a importância do diário para a compreensão da personalidade do personagem.", # discursiva
        "Qual o significado simbólico da escuridão no final da história?",                # interpretativa
        "Quais críticas sociais podem ser percebidas na descrição da desigualdade econômica?", # crítica
        "Assinale a alternativa correta sobre a opinião do narrador em relação ao protagonista.", # objetiva
        "Qual foi a reação da personagem ao receber a notícia inesperada?",                # simples
        "Analise como o ambiente influencia as decisões do personagem principal.",        # complexa
        "Quem é o narrador da história e qual é seu papel na narrativa?",                 # literal
        "Descreva a transformação do personagem ao longo do enredo.",                      # discursiva
        "Explique o uso da metáfora 'tempestade interna' para expressar o conflito do protagonista.", # interpretativa
        "Quais aspectos sociais são criticados no texto através da fala do personagem secundário?", # crítica
        "Qual a motivação do personagem principal para tomar a decisão final?",            # discursiva
        "Interprete o simbolismo da luz no desfecho da narrativa.",                        # interpretativa
        "Cite duas críticas sociais presentes nas atitudes dos personagens coadjuvantes.", # crítica
        "Assinale a alternativa que melhor representa o tema principal do texto.",        # objetiva
        "O que acontece logo após o diálogo entre os protagonistas?",                     # simples
        "Compare o comportamento do personagem antes e depois do conflito central.",      # complexa
        "Quem é o narrador da história e qual sua importância?",                          # literal
        "Descreva o impacto emocional do evento traumático no personagem principal.",     # discursiva
        "Explique o significado da metáfora 'coração de pedra' usada no texto.",          # interpretativa
        "Quais críticas sociais são evidentes no ambiente descrito na narrativa?",        # crítica
        "Qual alternativa apresenta o melhor resumo do enredo?",                          # objetiva
        "Qual foi a primeira reação do personagem após receber a notícia?",               # simples
        "Analise a relação entre o contexto histórico e as ações do personagem principal.",# complexa
        "Qual é o nome da cidade onde se passa a maior parte da história?",               # literal
        "Explique, com suas palavras, a transformação do protagonista ao longo da narrativa.", # discursiva
        "Interprete o uso da metáfora 'as sombras do passado' no texto.",                 # interpretativa
        "Quais críticas sociais podem ser percebidas na fala do personagem secundário?",  # crítica
        "Assinale a alternativa correta sobre o ponto de vista narrativo.",               # objetiva
        "O que ocorre imediatamente após o evento que desencadeia o conflito?",           # simples
        "Compare os sentimentos do personagem no início e no fim da narrativa.",          # complexa
        "Quem é o personagem que simboliza a esperança na história?",                    # literal
        "Descreva a importância do diário encontrado pelo protagonista.",                 # discursiva
        "Explique o significado da metáfora 'tempestade interna' no desenvolvimento do personagem.", # interpretativa
        "Quais aspectos sociais são criticados na descrição da cidade?",                  # crítica
        "Assinale a alternativa que apresenta o tema central do texto.",                  # objetiva
        "Qual foi a primeira atitude do personagem após o conflito?",                      # simples
        "Analise o desenvolvimento emocional do protagonista ao longo da narrativa.",     # complexa
        "Quem é o personagem que representa a autoridade na história?",                   # literal
        "Descreva com suas palavras o ambiente onde a narrativa se desenrola.",            # discursiva
        "Interprete o significado da metáfora 'caminho sem saída' no contexto do texto.",  # interpretativa
        "Quais críticas sociais são abordadas por meio dos diálogos dos personagens?",     # crítica
        "Assinale a alternativa que melhor sintetiza o tema central da história.",        # objetiva
        "O que acontece imediatamente após a chegada do visitante misterioso?",           # simples
        "Compare o comportamento dos personagens principais em momentos-chave da trama.", # complexa
        "Qual é o nome do bairro onde se passa a maior parte da narrativa?",              # literal
        "Explique a importância do diário para a compreensão da personalidade do personagem.", # discursiva
        "Qual o significado simbólico da escuridão no final da história?",                # interpretativa
        "Quais críticas sociais podem ser percebidas na descrição da desigualdade econômica?", # crítica
        "Assinale a alternativa correta sobre a opinião do narrador em relação ao protagonista.", # objetiva
        "Qual foi a reação da personagem ao receber a notícia inesperada?",                # simples
        "Analise como o ambiente influencia as decisões do personagem principal.",        # complexa
        "Quem é o narrador da história e qual é seu papel na narrativa?",                 # literal
        "Descreva a transformação do personagem ao longo do enredo.",                      # discursiva
        "Explique o uso da metáfora 'tempestade interna' para expressar o conflito do protagonista.", # interpretativa
        "Quais aspectos sociais são criticados no texto através da fala do personagem secundário?", # crítica
        "Qual foi o motivo da mudança de comportamento do protagonista?",                   # discursiva
        "Interprete a metáfora 'janelas fechadas' presente no texto.",                      # interpretativa
        "Cite duas críticas sociais relacionadas ao sistema educacional abordadas na narrativa.", # crítica
        "Assinale a alternativa que melhor representa o tema principal da história.",       # objetiva
        "O que acontece logo após o personagem descobrir o segredo?",                       # simples
        "Compare o desenvolvimento emocional dos personagens principais ao longo do texto.", # complexa
        "Quem é o personagem que simboliza a esperança na narrativa?",                     # literal
        "Descreva o impacto do conflito central na trajetória do protagonista.",            # discursiva
        "Explique o significado da metáfora 'tempo escorrendo como areia' usada pelo autor.", # interpretativa
        "Quais críticas sociais são evidentes na descrição do ambiente urbano?",            # crítica
        "Qual alternativa apresenta um resumo adequado do enredo?",                        # objetiva
        "Qual foi a reação inicial do personagem após o evento traumático?",                # simples
        "Analise a relação entre o contexto histórico e o comportamento do protagonista.",  # complexa
        "Qual é o nome da cidade onde se passa a maior parte da narrativa?",                # literal
        "Explique com suas palavras a transformação do personagem principal.",              # discursiva
        "Interprete o uso da metáfora 'tempestade interna' para representar conflitos emocionais.", # interpretativa
        "Quais aspectos sociais são criticados por meio dos diálogos dos personagens?",     # crítica
        "Assinale a alternativa correta sobre a opinião do narrador acerca do protagonista.", # objetiva
        "O que acontece imediatamente após a chegada do visitante misterioso?",            # simples
        "Compare os sentimentos do personagem principal no início e no fim da história.",   # complexa
        "Quem é o narrador da história e qual é seu papel na narrativa?",                  # literal
        "Descreva a importância do diário encontrado pelo protagonista para a narrativa.",  # discursiva
        "Explique o significado da metáfora 'coração em ruínas' usada pelo autor.",         # interpretativa
        "Quais críticas sociais podem ser percebidas nas falas do personagem coadjuvante?", # crítica
        "Assinale a alternativa correta sobre o tema central do texto.",                   # objetiva
        "Qual foi a reação do protagonista ao descobrir a verdade?",                       # simples
        "Analise a relação entre os personagens principais e seu impacto na narrativa.",  # complexa
        "Quem é o personagem que simboliza a justiça na história?",                       # literal
        "Descreva o ambiente descrito no capítulo inicial da narrativa.",                  # discursiva
        "Interprete o simbolismo da chuva constante ao longo do texto.",                   # interpretativa
        "Quais críticas sociais são feitas por meio da descrição da pobreza no enredo?",   # crítica
        "Assinale a alternativa que melhor resume o conflito central da história.",        # objetiva
        "O que acontece imediatamente após o confronto entre os personagens principais?",  # simples
        "Compare a evolução emocional do protagonista durante o desenrolar da trama.",     # complexa
        "Qual é o nome da cidade onde a maior parte da história ocorre?",                  # literal
        "Explique a importância do diário para o desenvolvimento do personagem principal.",# discursiva
        "Qual o significado da metáfora 'sombras do passado' no texto?",                   # interpretativa
        "Quais aspectos sociais são criticados na fala do personagem antagonista?",        # crítica
        "Assinale a alternativa correta sobre a perspectiva narrativa adotada.",           # objetiva
        "Qual foi a primeira ação do personagem após receber a notícia inesperada?",       # simples
        "Analise o impacto do contexto histórico nas decisões do protagonista.",           # complexa
        "Quem é o narrador da história e qual seu papel na trama?",                        # literal
        "Descreva as mudanças comportamentais do protagonista ao longo da narrativa.",    # discursiva
        "Explique o significado da metáfora 'coração de pedra' no desenvolvimento do personagem.", # interpretativa
        "Quais críticas sociais podem ser percebidas nas interações entre os personagens?",# crítica
        "Assinale a alternativa correta sobre o tema principal do texto.",                 # objetiva
        "O que ocorre imediatamente após o diálogo entre os personagens?",                # simples
        "Compare os sentimentos do personagem principal do início ao fim da narrativa.",   # complexa
        "Quem é o personagem que representa a esperança na história?",                    # literal
        "Descreva a importância do diário encontrado pelo protagonista.",                  # discursiva
        "Qual foi a reação inicial do personagem principal ao receber a notícia?",           # simples
        "Analise como os conflitos internos influenciam as decisões do protagonista.",       # complexa
        "Quem é o personagem que representa a figura da autoridade na narrativa?",          # literal
        "Descreva o ambiente onde se desenrola a maior parte da história.",                  # discursiva
        "Interprete o significado da metáfora 'mar revolto' usada no texto.",                # interpretativa
        "Quais críticas sociais são evidentes na descrição do sistema educacional?",         # crítica
        "Assinale a alternativa que melhor sintetiza o tema principal do texto.",            # objetiva
        "O que acontece imediatamente após a descoberta do segredo?",                       # simples
        "Compare o comportamento dos personagens antes e depois do evento principal.",      # complexa
        "Qual é o nome do local onde a história se passa?",                                 # literal
        "Explique a importância do diário para compreender a personalidade do personagem.",  # discursiva
        "Qual o significado simbólico da escuridão no final da narrativa?",                  # interpretativa
        "Quais críticas sociais podem ser percebidas através das falas dos personagens?",    # crítica
        "Assinale a alternativa correta sobre a opinião do narrador em relação ao protagonista.", # objetiva
        "Qual foi a reação da personagem ao encontrar a carta misteriosa?",                  # simples
        "Analise a influência do ambiente nas decisões do protagonista.",                    # complexa
        "Quem é o narrador da história e qual seu papel na narrativa?",                      # literal
        "Descreva a transformação do personagem principal ao longo do texto.",               # discursiva
        "Explique o uso da metáfora 'tempestade interna' para expressar o conflito emocional.", # interpretativa
        "Quais aspectos sociais são criticados por meio da fala do personagem secundário?",  # crítica
        "Assinale a alternativa que apresenta o tema central da narrativa.",                 # objetiva
        "O que ocorre logo após o diálogo entre os personagens principais?",                 # simples
        "Compare os sentimentos do protagonista no início e no fim da história.",            # complexa
        "Quem é o personagem que simboliza a esperança na narrativa?",                      # literal
        "Descreva a importância do diário encontrado pelo protagonista para a trama.",       # discursiva
        "Qual foi a primeira reação do personagem principal ao receber a notícia inesperada?", # simples
        "Analise o impacto das escolhas do protagonista na evolução da trama.",                # complexa
        "Quem é o personagem que representa a figura de autoridade na história?",            # literal
        "Descreva o ambiente em que se desenvolve a narrativa.",                              # discursiva
        "Interprete o significado da metáfora 'o tempo escorreu entre os dedos' usada no texto.", # interpretativa
        "Quais críticas sociais são evidentes na descrição do contexto familiar?",             # crítica
        "Assinale a alternativa que melhor sintetiza o tema central da narrativa.",           # objetiva
        "O que acontece logo após o personagem descobrir o segredo?",                         # simples
        "Compare o comportamento dos personagens principais antes e depois do conflito.",    # complexa
        "Qual é o nome da cidade onde se passa a maior parte da história?",                   # literal
        "Explique a importância do diário para a compreensão do protagonista.",               # discursiva
        "Qual o significado simbólico da escuridão no desfecho da narrativa?",                # interpretativa
        "Quais críticas sociais podem ser percebidas nas falas do personagem antagonista?",   # crítica
        "Assinale a alternativa correta sobre o ponto de vista narrativo adotado.",           # objetiva
        "Qual foi a primeira ação do personagem após receber a notícia inesperada?",           # simples
        "Analise como o contexto histórico influencia as decisões do protagonista.",          # complexa
        "Quem é o narrador da história e qual seu papel na narrativa?",                       # literal
        "Descreva as transformações do personagem principal ao longo da narrativa.",          # discursiva
        "Explique o uso da metáfora 'tempestade interna' para expressar os conflitos emocionais.", # interpretativa
        "Quais aspectos sociais são criticados por meio das interações dos personagens?",     # crítica
        "Assinale a alternativa correta sobre o tema principal do texto.",                    # objetiva
        "O que ocorre imediatamente após o diálogo entre os personagens?",                   # simples
        "Compare os sentimentos do personagem principal no início e no fim da narrativa.",    # complexa
        "Quem é o personagem que simboliza a esperança na história?",                        # literal
        "Descreva a importância do diário encontrado pelo protagonista para a narrativa.",    # discursiva
        "Qual é o papel do narrador na construção da história?",                           # literal
        "Descreva como o cenário influencia a atmosfera do texto.",                        # discursiva
        "Analise o conflito interno do protagonista e suas consequências.",                # complexa
        "Interprete a metáfora 'flores murchas' usada para descrever a esperança no texto.", # interpretativa
        "Quais críticas sociais são feitas ao sistema político na narrativa?",             # crítica
        "Assinale a alternativa que sintetiza o ponto de vista do autor sobre o tema.",    # objetiva
        "O que ocorre logo após o personagem receber uma notícia inesperada?",             # simples
        "Compare a relação entre os personagens principais e como ela evolui na história.", # complexa
        "Qual é o nome do local onde a ação principal acontece?",                          # literal
        "Explique a importância do objeto encontrado pelo protagonista.",                  # discursiva
        "Como a metáfora 'mares agitados' reflete o estado emocional do personagem?",      # interpretativa
        "Que críticas sociais são evidenciadas através da descrição das condições de vida?", # crítica
        "Assinale a alternativa correta que resume o tema central do texto.",             # objetiva
        "Qual foi a reação inicial do personagem diante do conflito?",                     # simples
        "Analise a influência do contexto histórico no comportamento dos personagens.",    # complexa
        "Quem é o personagem que simboliza a resistência na narrativa?",                   # literal
        "Descreva as mudanças psicológicas do protagonista durante o enredo.",            # discursiva
        "Interprete o simbolismo da escuridão presente no final da narrativa.",            # interpretativa
        "Quais críticas sociais estão implícitas nas falas do personagem secundário?",     # crítica
        "Assinale a alternativa correta sobre o gênero textual da obra.",                  # objetiva
        "O que acontece logo após o desfecho da trama?",                                  # simples
        "Compare os sentimentos do protagonista em dois momentos diferentes da narrativa.",# complexa
        "Quem é o narrador e qual é sua importância para o desenvolvimento da história?", # literal
        "Explique a relevância do diário para o entendimento do personagem principal.",    # discursiva
        "Qual o significado da metáfora 'corações partidos' no contexto do texto?",        # interpretativa
        "Qual é o tema principal abordado na narrativa?",                                # objetiva
        "Descreva a personalidade do protagonista ao longo da história.",                 # discursiva
        "Quem é o personagem que simboliza a transformação no texto?",                    # literal
        "Analise a influência do ambiente na tomada de decisão do personagem.",           # complexa
        "Interprete a metáfora 'céu cinzento' e seu impacto na atmosfera do texto.",      # interpretativa
        "Quais críticas sociais são feitas à desigualdade econômica na narrativa?",       # crítica
        "Assinale a alternativa que melhor resume o conflito central da história.",       # objetiva
        "O que ocorre logo após o personagem principal descobrir um segredo importante?", # simples
        "Compare as atitudes dos personagens antes e depois do evento decisivo.",          # complexa
        "Qual é o nome do local onde a maior parte da narrativa acontece?",               # literal
        "Explique a importância dos objetos simbólicos presentes na história.",           # discursiva
        "Interprete o significado da metáfora 'correntes invisíveis' usada no texto.",    # interpretativa
        "Quais críticas sociais são evidenciadas através das ações dos personagens?",     # crítica
        "Assinale a alternativa correta sobre o ponto de vista narrativo adotado.",       # objetiva
        "Qual foi a reação inicial do protagonista após o conflito?",                     # simples
        "Analise como o contexto histórico influencia a evolução dos personagens.",       # complexa
        "Quem é o narrador e qual sua função na construção da trama?",                    # literal
        "Descreva a evolução emocional do personagem principal durante a narrativa.",     # discursiva
        "Explique o simbolismo da metáfora 'tempestade interna' para os conflitos do protagonista.", # interpretativa
        "Quais aspectos sociais são criticados por meio da fala dos personagens secundários?", # crítica
        "Assinale a alternativa correta sobre o tema central do texto.",                  # objetiva
        "O que acontece imediatamente após o diálogo entre os personagens principais?",  # simples
        "Compare os sentimentos do protagonista no início e no final da história.",       # complexa
        "Quem é o personagem que representa a esperança na narrativa?",                  # literal
        "Descreva a importância do diário para o desenvolvimento da trama.",              # discursiva
        "Qual é o significado da metáfora 'coração partido' usada pelo autor?",           # interpretativa
        "Quais críticas sociais podem ser percebidas nas interações entre os personagens?", # crítica
        "Assinale a alternativa correta que resume o gênero textual da obra.",            # objetiva
        "Qual foi a primeira atitude do personagem após receber a notícia inesperada?",   # simples
        "Analise o impacto do conflito interno nas decisões do protagonista.",            # complexa
        "Qual é a mensagem principal transmitida pelo autor na narrativa?",                # objetiva
        "Descreva as emoções que o protagonista experimenta ao longo da história.",         # discursiva
        "Quem é o personagem que representa a sabedoria na trama?",                         # literal
        "Analise como o ambiente social influencia as decisões do protagonista.",            # complexa
        "Interprete a metáfora 'caminho tortuoso' e seu impacto no desenvolvimento da história.", # interpretativa
        "Quais críticas sociais estão presentes na descrição do ambiente familiar?",         # crítica
        "Assinale a alternativa que melhor representa o tema central do texto.",             # objetiva
        "O que acontece logo após o personagem principal enfrentar um obstáculo?",           # simples
        "Compare o comportamento do protagonista antes e depois do conflito principal.",     # complexa
        "Qual é o nome da cidade onde se passa a maior parte da narrativa?",                 # literal
        "Explique a importância do objeto simbólico encontrado pelo personagem principal.",  # discursiva
        "Interprete o significado da metáfora 'noite sem estrelas' usada no texto.",         # interpretativa
        "Quais críticas sociais podem ser percebidas na fala do personagem secundário?",     # crítica
        "Assinale a alternativa correta sobre o ponto de vista narrativo adotado na história.", # objetiva
        "Qual foi a primeira reação do personagem após receber uma notícia chocante?",       # simples
        "Analise a influência do contexto histórico na trajetória do protagonista.",         # complexa
        "Quem é o narrador da história e qual o seu papel na trama?",                        # literal
        "Descreva a transformação emocional do personagem principal ao longo da narrativa.", # discursiva
        "Explique o simbolismo da metáfora 'tempestade interna' no desenvolvimento do personagem.", # interpretativa
        "Quais aspectos sociais são criticados por meio das atitudes dos personagens coadjuvantes?", # crítica
        "Assinale a alternativa correta sobre o tema central da obra.",                      # objetiva
        "O que acontece imediatamente após o diálogo entre os personagens principais?",      # simples
        "Compare os sentimentos do protagonista no começo e no fim da narrativa.",           # complexa
        "Quem é o personagem que simboliza a esperança na história?",                       # literal
        "Descreva a importância do diário para a compreensão da trama.",                     # discursiva
        "Qual o significado da metáfora 'coração partido' usada pelo autor?",                # interpretativa
        "Quais críticas sociais podem ser percebidas nas interações entre os personagens?",  # crítica
        "Assinale a alternativa correta sobre o gênero textual da narrativa.",               # objetiva
        "Qual foi a primeira atitude do personagem após receber a notícia inesperada?",      # simples
        "Analise o impacto do conflito emocional nas decisões do protagonista.",             # complexa
        "Qual foi a consequência imediata da escolha feita pelo personagem no clímax da história?",           # simples
        "Analise a maneira como o autor constrói a tensão entre os personagens ao longo da narrativa.",       # complexa
        "Quem é mencionado como responsável pela mudança de cenário no início da história?",                 # literal
        "Descreva a relação entre o tempo e as emoções do protagonista ao longo do texto.",                   # discursiva
        "Interprete o uso da expressão 'sorrisos quebrados' no desenvolvimento emocional dos personagens.",   # interpretativa
        "Quais críticas sociais podem ser identificadas nas descrições do sistema de transporte público?",    # crítica
        "Assinale a alternativa que demonstra uma conclusão coerente com os eventos narrados.",               # objetiva
        "O que o personagem fez imediatamente após escutar o barulho vindo da porta?",                        # simples
        "Compare os efeitos da ausência e da presença da figura paterna no comportamento do protagonista.",   # complexa
        "Quem é o personagem secundário que mais influencia nas decisões do protagonista?",   
        "Qual foi a primeira reação da personagem ao perceber que estava sozinha?",                     # simples
        "Analise como a memória influencia as ações do protagonista durante o enredo.",                 # complexa
        "Quem é o personagem que aparece apenas no final e muda o rumo da história?",                  # literal
        "Descreva os sentimentos do narrador ao reviver os acontecimentos passados.",                  # discursiva
        "Interprete o sentido simbólico da palavra 'labirinto' no desenvolvimento do texto.",          # interpretativa
        "Quais críticas sociais são evidenciadas na relação entre patrão e empregado descrita?",       # crítica
        "Assinale a alternativa que apresenta corretamente a função do narrador na narrativa.",         # objetiva
        "O que o protagonista fez logo após sair da sala escura?",                                     # simples
        "Compare as reações dos dois personagens diante da mesma perda.",                              # complexa
        "Quem é o personagem que serve como contraponto ao herói da história?", 
        "Qual foi a primeira atitude do personagem ao notar a ausência de sua irmã?",                         # simples
        "Analise como os flashbacks interferem na estrutura narrativa do conto.",                             # complexa
        "Quem é o personagem mencionado nas cartas encontradas pelo protagonista?",                           # literal
        "Descreva a forma como o autor representa o sentimento de culpa no personagem principal.",            # discursiva
        "Interprete a metáfora 'seu olhar era um abismo' usada para descrever o antagonista.",                # interpretativa
        "Quais críticas sociais podem ser percebidas na forma como os idosos são tratados no enredo?",        # crítica
        "Assinale a alternativa correta sobre a moral implícita na história lida.",                           # objetiva
        "O que acontece logo após o personagem ouvir uma voz familiar em meio à multidão?",                   # simples
        "Compare a trajetória emocional dos personagens em diferentes fases da narrativa.",                   # complexa
        "Quem é a figura misteriosa que aparece no terceiro ato da história?",                                # literal
        "Descreva os impactos psicológicos que o cenário de guerra provocou no personagem principal.",        # discursiva
        "Explique o uso simbólico da metáfora 'paredes que sussurravam lembranças' no texto.",                # interpretativa
        "Quais críticas sociais surgem a partir da ambientação da história em uma cidade fictícia?",          # crítica
        "Assinale a opção que representa a interpretação mais coerente do título da narrativa.",              # objetiva
        "Qual foi a primeira reação do personagem ao se ver sozinho na estação de trem?",                     # simples
        "Analise o papel do silêncio como elemento narrativo ao longo da história.",                          # complexa
        "Quem é o personagem que age como conselheiro do protagonista?",                                      # literal
        "Descreva a importância do conflito familiar para o desenvolvimento do enredo.",                      # discursiva
        "Interprete o significado da expressão 'tempestade emocional' no contexto vivido pelo personagem.",   # interpretativa
        "Quais críticas sociais são feitas à política por meio dos acontecimentos descritos?",                # crítica
        "Assinale a alternativa correta sobre a mudança de ponto de vista na narrativa.",                     # objetiva
        "O que o personagem fez após encontrar o bilhete escondido no livro antigo?",                         # simples
        "Compare o estilo de vida do protagonista com o de sua irmã, e como isso afeta suas decisões.",       # complexa
        "Quem é o personagem citado apenas nos pensamentos do narrador?",                                     # literal
        "Descreva como a ausência do pai moldou o comportamento do personagem principal.",                    # discursiva
        "Explique o uso da metáfora 'labirinto sem saída' no desenvolvimento do enredo.",                     # interpretativa
        "Quais críticas sociais estão implícitas no retrato da escola pública no início do texto?",           # crítica
        "Assinale a alternativa correta que apresenta o foco narrativo da segunda parte da história.",        # objetiva
        "O que acontece com o personagem logo após tomar uma decisão impulsiva?",                             # simples
        "Analise como os elementos simbólicos contribuem para o desfecho da narrativa.",  
        "Qual é o nome do personagem principal?",
        "O que significa a palavra 'liberdade' no texto?",
        "Como a relação entre os personagens revela conflitos sociais?",
        "Quais são as consequências históricas da Revolução Industrial para a educação?",
        "Qual é o nome do protagonista da história?",
        "Explique com suas palavras o motivo da mudança de atitude do personagem.",
        "Quais críticas sociais são feitas à desigualdade econômica na narrativa?"
    ]

    target = [
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,   # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,   # complexa
        2,  # literal
        2,  # literal
        6,  # discursiva
        6,  # discursiva
        3,  # interpretativa
        3,  # interpretativa
        4,  # crítica
        4,  # crítica
        5,  # objetiva
        5,  # objetiva
        0,  # simples
        0,  # simples
        1,  # complexa
        1,   # complexa
        2,  # literal
        2,  # literal
        6,  # discursiva
        6,  # discursiva
        3,  # interpretativa
        3,  # interpretativa
        4,  # crítica
        4,  # crítica
        5,  # objetiva
        5,  # objetiva
        0,  # simples
        0,  # simples
        1,  # complexa
        1,   # complexa
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,  # complexa
        6,  # discursiva
        3,  # interpretativa
        0,  # simples
        5,  # objetiva
        4,  # crítica
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        0,  # simples
        1,  # complexa
        5,  # objetiva
        4,  # crítica
        2,  # literal
        6,   # discursiva
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,   # simples
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        2,  # literal
        6,  # discursiva
        1,  # complexa
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        5,  # objetiva
        6,  # discursiva
        2,  # literal
        1,  # complexa
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        5,  # objetiva
        6,  # discursiva
        2,  # literal
        1,  # complexa
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        2,  # literal
        6,  # discursiva
        3,  # interpretativa
        4,  # crítica
        5,  # objetiva
        0,  # simples
        1,  # complexa
        0,
        3,
        4,
        1,
        2,
        6,
        4
    ]

    target_names = [
        'simples',
        'complexa',
        'literal',
        'interpretativa',
        'crítica',
        'objetiva',
        'discursiva'
    ]

    return Bunch(
        data=data,
        target=target,
        target_names=target_names
    )
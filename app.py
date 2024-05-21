import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
import scikit_posthocs as sp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from io import StringIO

################################Funções do app#########################################################
def calculate_lipinski_properties(smiles):
    # cria o mol a partir do smiles
    mol = Chem.MolFromSmiles(smiles)

    # Calcula as propriedades de Lipinski 
    molecular_weight = Descriptors.MolWt(mol)
    num_hydrogen_bond_donors = Descriptors.NumHDonors(mol)
    num_hydrogen_bond_acceptors = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)
    qed_value = QED.qed(mol)
    
    # Verifica se passa pelos critérios da Ro5 de Lipinski
    is_pass = (
        molecular_weight <= 500 and
        num_hydrogen_bond_donors <= 5 and
        num_hydrogen_bond_acceptors <= 10 and
        logp <= 5
    )

    # Retorna as propriedades de Lipinski como um dicionario
    lipinski_properties = {        
        'HBD': num_hydrogen_bond_donors,
        'MW': molecular_weight,
        'QED': qed_value,        
        'LogP': logp,
        'HBA': num_hydrogen_bond_acceptors        
    }

    return lipinski_properties

def create_radar_plot_with_threshold(lipinski_properties, min_values, max_values, title, color):
    properties = list(lipinski_properties.keys())
    values = list(lipinski_properties.values())

    # Formatar os rótulos dos eixos para incluir os valores máximos
    properties_labels = [f"{prop} ({max_val})" for prop, max_val in zip(properties, max_values)]

    # Normaliza os valores entre 0 to 1
    normalized_values = [(value - min_val) / (max_val - min_val) for value, min_val, max_val in zip(values, min_values, max_values)]
    normalized_values += normalized_values[:1]  # Close the loop with the first value

    # Cria a lista de angulos para cada propriedade, encerrando o loop
    angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Limira o valor máximo normalizado para 1, encerra o loop
    threshold_values = [1] * len(properties)
    threshold_values += threshold_values[:1]  # Encerra o loop

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
    ax.fill(angles, normalized_values, color=color, alpha=0.25)
    ax.plot(angles, normalized_values, color=color, linewidth=2)
    
    # Plot a linha de limite
    ax.plot(angles, threshold_values, 'g--', linewidth=2, label='Limite')

    # Configura as labels para propriedades com os valores máximos
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(properties_labels, fontsize=8, fontfamily='Arial')

    # Adiciona os valores das variáveis dentro da área de plotagem
    value_padding = 0.85  # Ajuste este valor conforme necessário
    for angle, value, normalized_value in zip(angles[:-1], values, normalized_values[:-1]):
        ax.text(angle, normalized_value * value_padding, f"{value:.1f}", ha='center', va='center', fontsize=8, color='black')


    # Remove radial labels
    ax.set_yticks([])  # Remove a grade radial

    # Legenda e Título
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), prop={'size': 8})
    plt.title(title.capitalize(), fontsize=15)
    plt.tight_layout()
    return fig

def chemical_struture_2d(smiles):
    m = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(m)    
    return img

def smiles_to_molecular_formula(smiles):
    # Converte o SMILES em um objeto Mol
    mol = Chem.MolFromSmiles(smiles)
    # Calcula a fórmula molecular
    formula_molecular = CalcMolFormula(mol)
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    formula_molecular = formula_molecular.translate(SUB)   
    return formula_molecular


############################Estrutura do Aplicativo#####################################################

st.set_page_config(page_title='Análise de Estruturas',
                    page_icon="💊",
                    layout="wide",
                    initial_sidebar_state="auto",
                    menu_items=None)

   
    
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Apresentação do Projeto','Importação dos Dados',"Gráficos de Dispersão", "Gráficos Radar","Análises Estatísticas", 'Séries Temporais', 'Glossário'])

with tab1:
    st.markdown("<h4 style='text-align: center; color: black;'>Plataforma Interativa para Exploração do Espaço Químico em Química Medicinal</h4>", unsafe_allow_html=True)

    st.markdown('''Este projeto apresenta uma plataforma educacional interativa desenvolvida para estudantes de Química Medicinal, objetivando facilitar a exploração detalhada do espaço químico de medicamentos. Utilizando dados extraídos do **ChEMBL versão 33**, a plataforma permite aos usuários visualizar interativamente propriedades físico-químicas de medicamentos, realizar análises estatísticas descritivas, e estudar as tendências no desenvolvimento de medicamentos através de análises de séries temporais ao longo de várias décadas.
                
A ferramenta incorpora funcionalidades para aplicação de filtros baseados nas regras de Lipinski, o que ajuda na identificação de moléculas com potencial farmacológico, propriedades físico-quimicas (tais como massa molecular, clogP, DLH, ALH),  vias de administração dos medicamentos, data de sua aprovação. Esses recursos possibilitam os estudantes investigarem as principais influências nas características químicas dos compostos. 

Esta iniciativa representa um passo significativo na educação moderna em Química Medicinal, preparando os estudantes não apenas para entenderem melhor a ciência por trás do desenvolvimento de medicamentos, mas também para utilizarem ferramentas de análise de dados que são cruciais no avanço da pesquisa farmacêutica contemporânea.

''')
   

with tab2:
    uploaded_file = st.file_uploader("Escolha um arquivo do Excel") 
    # if uploaded_file:   
    #     dataframe = pd.read_excel(uploaded_file)
    #     # dataframe = pd.read_csv(uploaded_file)
    #     st.dataframe(dataframe)
    
    dataframe = pd.read_excel('data/medicamentos.xlsx')
    st.dataframe(dataframe)
    
    

with tab3:
    col1, col2= st.columns(2)
    # if uploaded_file:
    with col1:        
        colunas_numericas = list(dataframe.select_dtypes(include='number').columns)
        eixos_selecionados = st.multiselect(
        'Selecione os eixos **X** e **Y** para serem plotados no gráfico de dispersão',
        colunas_numericas,
        max_selections = 2,            
        placeholder='Selecione os eixos X e Y...')
    with col2:
        cor_selecionada = st.multiselect(
        'Selecione 1 propriedade para adicionar cor às propriedades a serem plotadas no gráfico de dispersão',
        list(dataframe.columns),
        max_selections = 1,
        placeholder='Selecione uma propriedade...')


    if eixos_selecionados:            
        if len(eixos_selecionados)==2:
            fig, ax = plt.subplots( figsize=(10,7))

            x = eixos_selecionados[0]
            y = eixos_selecionados[1]
            if len(cor_selecionada) > 0:
                fig = px.scatter(dataframe, x=x, y=y, height=600, color=cor_selecionada[0], hover_data=['farmaco'] )
            else:
                fig = px.scatter(dataframe, x=x, y=y, height=600, hover_data=['farmaco'])
            fig.update_layout(plot_bgcolor='white')
            fig.update_yaxes(title_font_color="black")
            fig.update_xaxes(title_font_color="black")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()    
            
with tab4:
    # if uploaded_file:
    farmaco_selecionado = st.selectbox(
                                        'Selecione um medicamento:',                                            
                                        options=dataframe['farmaco'].sort_values(ascending=True)
                                    )
    smiles_selecionado = dataframe[dataframe['farmaco'] == farmaco_selecionado]['canonical_smiles'].iloc[0]
    classe_medicamento = dataframe[dataframe['farmaco'] == farmaco_selecionado]['classe_farmacologica'].iloc[0]
    
    estrutura_2d = chemical_struture_2d(smiles_selecionado)
    formula_molecular = smiles_to_molecular_formula(smiles_selecionado)

    min_values = [0, 100, 0, 0, 0]
    max_values = [5, 500, 1, 5, 10]
    ro5_props = calculate_lipinski_properties(fr"{smiles_selecionado}")
    polar_plot = create_radar_plot_with_threshold(ro5_props, min_values, max_values, farmaco_selecionado, "#0B1B82")
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(polar_plot)
    with col2:                       
        st.image(estrutura_2d)                        
        df_prop = pd.DataFrame(ro5_props, index=[0])
        st.write(f'**Classe**: {classe_medicamento.capitalize()}')
        st.write(f'**Fórmula Molecular**: {formula_molecular}')
        st.write(df_prop)
            

with tab5:    
    # if uploaded_file:
    st.subheader('**Resumo Estatístico das Propriedades**')
    via_admin = st.radio( 'Selecione uma via de administração:',['Todas','Oral', 'Parenteral', 'Tópica'], horizontal = True)
    if via_admin == 'Todas':            
        df_estatistico = dataframe.describe([.25,.5,.75,.9])
    elif via_admin == 'Oral':
        df_estatistico = dataframe[dataframe['via_administracao'] == 'oral'].describe([.25,.5,.75,.9]) 
    elif via_admin == 'Parenteral':
        df_estatistico = dataframe[dataframe['via_administracao'] == 'parenteral'].describe([.25,.5,.75,.9])
    elif via_admin == 'Tópica':
        df_estatistico = dataframe[dataframe['via_administracao'] == 'topical'].describe([.25,.5,.75,.9])

            
    df_estatistico = df_estatistico.rename(index={'count': 'contagem',
                                                        'mean': 'média',
                                                        'std': 'dp',
                                                        })
    st.write(df_estatistico)

with tab6:
    # if uploaded_file:

    prop_quimica = st.selectbox('Selecione uma propriedade para comparação:',
                            ('massa_molecular', 'log_p', 'atomos_pesados', 'alh', 'dlh', 'lig.rot.', 'num_ar', 'tpsa', 'qed'),
                            index=None,
                            placeholder='Selecione uma opção...'
                            )       
    sns.set_theme(style='white', context='poster', palette='twilight')
    if prop_quimica:
        st.markdown("<h4 style='text-align: center; color: black;'>Histogramas de Propriedades Físico-Químicas por Vintênio</h4>", unsafe_allow_html=True)
        # Plotagem dos histogramas
        fig, ax = plt.subplots(figsize=(20,10))
        boxplot = sns.boxplot(x="periodo", y=prop_quimica, data=dataframe,
                                color='wheat',
                                medianprops={"color": "r", "linewidth": 2})
        boxplot.set_xlabel('Período')
        st.pyplot(fig)

        st.markdown("<h5 style='text-align: center; color: black;'>Análise Post-hoc de Propriedades Físico-Químicas com Teste de Wilcoxon</h5>", unsafe_allow_html=True)

        #Plotagem Heatmap
        fig2, ax = plt.subplots()
        # ajuste fontes dos eixos X e Y
        yticks, ylabels = plt.yticks()
        xticks, xlabels = plt.xticks()
        ax.set_xticklabels(xlabels, size=10)
        ax.set_yticklabels(ylabels, size=10)
        
                                
        sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.0)
        pc = sp.posthoc_mannwhitney(dataframe, val_col=prop_quimica, group_col="periodo", p_adjust='holm')
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                        'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        _ = sp.sign_plot(pc, **heatmap_args)
        st.pyplot(fig2)

with tab7:
    st.markdown('''
1. **cLogP** - O coeficiente de partição logarítmico (cLogP) é um valor calculado que estima a lipofilicidade de uma molécula, representando o logaritmo da razão de sua concentração entre as fases octanol e água.

2. **Átomos pesados diferentes de hidrogênio** - Refere-se a todos os átomos em uma molécula que não são átomos de hidrogênio. Estes átomos geralmente contribuem mais para a massa molecular total e para as propriedades químicas da molécula.

3. **ALH** - (Aceptores de Ligação de Hidrogênio) - Refere-se ao número de grupos funcionais em uma molécula que podem aceitar ligações de hidrogênio. Estes grupos geralmente possuem átomos eletronegativos como oxigênio ou nitrogênio, que podem formar ligações de hidrogênio com átomos de hidrogênio doadores em outras moléculas.
4. **DLH** - (Doadores de Ligação de Hidrogênio) - Refere-se ao número de grupos funcionais em uma molécula que podem doar um átomo de hidrogênio para formar ligações de hidrogênio. Esses grupos tipicamente incluem -OH ou -NH2, que contêm hidrogênios que podem ser compartilhados com aceitadores de hidrogênio.

5. **Ligações rotacionáveis** - Ligações simples entre dois átomos não terminais que permitem a rotação livre, proporcionando flexibilidade à molécula. São importantes para determinar a conformação molecular e a capacidade de ajuste a sítios de ligação específicos.

6. **Fsp3** - É a fração de átomos de carbono sp3 em uma molécula, onde um átomo de carbono sp3 é aquele que é hibridizado tetraedricamente. Um maior valor de Fsp3 geralmente indica uma maior tridimensionalidade da molécula.

7. **Número de anéis aromáticos** - Refere-se à contagem total de anéis aromáticos em uma molécula. Anéis aromáticos são estruturas cíclicas que contêm ligações alternadas entre duplas e simples, conferindo estabilidade e propriedades eletrônicas específicas.

8. **TPSA** - Área de Superfície Polar Total (TPSA) é a soma de todas as áreas dos átomos de uma molécula que são capazes de formar ligações de hidrogênio, seja como doadores ou como aceitadores. Esta medida é utilizada para prever a capacidade de uma molécula de atravessar membranas celulares.

9. **QED** - (Quantitative Structure Drug-likeness) - É uma métrica quantitativa que avalia a "semelhança com fármacos" de uma molécula com base em suas propriedades estruturais. O QED é calculado a partir de uma série de parâmetros físico-químicos idealizados para fármacos, incluindo solubilidade, permeabilidade, peso molecular, e outros, fornecendo um valor que reflete a probabilidade de uma molécula ser um fármaco eficaz.


'''

    )
            

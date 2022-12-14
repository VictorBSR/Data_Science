# House Rocket - Análise de preços e recomendações de Portifolio

![icon](https://user-images.githubusercontent.com/42360197/207433028-735a3b18-1aa9-432f-9d16-6d2f027a0d64.png)

House Rocket é uma empresa fictícia que visa analisar dados de propriedades (imóveis) e realizar operações de compra e venda, maximizado o lucro e levando em conta principalmente:
- As condições do imóvel e
- A época do ano

O projeto em questão visa criar uma solução completa, na forma de um painel interativo hospedado na nuvem, para análise do ponto de vista do CEO da empresa sobre quais são os melhores negócios disponíveis no mercado. O painel inclui visualizações, análise de estatística descritiva, insights de negócio e recomendações de compra e venda.


# 1. Descrição - Questões de Negócio
Encontrar as melhores oportunidades de negócio: compra de casas em boas condições e com preços baixos, e venda desses imóveis adquiridos por um preço superior e justo. Os atributos dos imóveis, tais como localização, número de cômodos, áreas e datas de construção e reforma influenciam diretamente na sua atratividade e preço.
- **1. Quais imóveis a empresa deve adquirir e por qual preço? **
- **2. Para os imóveis adquiridos, quando é o melhor momento para vendê-los e por qual preço? **

# 2. Dataset e atributos
Originalmente baixado em <url>https://www.kaggle.com/harlfoxem/housesalesprediction/discussion/207885</url>. O dataset contém dados de propriedades na cidade de Seattle/USA. Definições dos atributos relevantes:

|    Atributo     |                         Descrição                            |
| :-------------: | :----------------------------------------------------------: |
|       id        |              Identificação única para cada imóvel            |
|      date       |               Data do anúncio de venda do imóvel             |
|      price      |                 Preço de venda atual do imóvel               |
|    bedrooms     |                      Número de quartos                       |
|    bathrooms    |                     Número de banheiros                      |
|   sqft_living   |    Medida em pés quadrados da área total interior do imóvel  |
|    sqft_lot     |        Medida em pés quadrados da área total do terreno      |
|     floors      |                 Número de andares do imóvel                  |
|   waterfront    |     Possui ou não frente para a água (0 = não e 1 = sim)     |
|      view       |   Indica a qualidade da vista da propriedade (0 = baixa e 4 = alta) |
|    condition    | Indica a qualidade das condições gerais da propriedade (0 = baixa e 5 = alta) |
|      grade      | Indica a qualidade da construção e o design do imóvel (1-3 = baixa, 4-10 = médio e 11-13 = alta) |
|  sqft_basement  |      Medida em pés quadrados da área total do porão/subsolo  |
|    yr_built     |                  Ano de construção do imóvel                 |
|  yr_renovated   |                Ano de reforma do imóvel                      |
|     zipcode     |                  CEP da localização do imóvel                |
|       lat       |                           Latitude                           |
|      long       |                          Longitude                           |
| sqft_livining15 | Medida em pés quadrados da área total interior de habitação para os 15 vizinhos mais próximo |
|   sqft_lot15    | Medida em pés quadrados da área total do terreno dos 15 vizinhos mais próximo |


# 3. Premissas do Negócio
As seguintes premissas foram adotadas para o dataset e o projeto como um todo:
- Imóveis com mais de 11 quartos foram considerados outliers e dessa forma. ignorados
- Na maioria das análises (exceto de evolução de preço ao longo do tempo), foi considerado somente o registro mais recente para cada propriedade, no caso dela ter múltiplas ocorrências em períodos distintos.
- Valores iguais a 0 (zero) significam que aquele atributo não está presente ou não é aplicável (ex.: não faz fronteira com a água, não sofreu reforma, não possui porão, etc.) e aqueles iguais a 1 (um) siginificam que tal atributo está presente.

# 4. Planejamento da solução
Foram adotadas as seguintes etapas, que guiaram a evolução deste projeto:
1) Entendimento do Negócio
2) Importação dos dados (via Kaggle)
3) Limpeza e tratamento dos dados
4) EDA (Análise Exploratória de Dados) - criação de visualizações e tabelas
5) Elaboração de Insights de Negócio - hipóteses com possível valor para o negócio
6) Recomendações de compra e venda

# 5. Os principais Insights de Negócio
**Insights de negócio** são hipóteses que levam em conta uma afirmação, análise e comparação entre 2 ou mais variáveis. Após analisada, gera-se uma conclusão geralmente na forma de gráficos e tabelas. Caso seja acionável (possível tomar decisões para beneficiar o negócio a partir dos encontrados), a hipótese se torna um *insight de negócio*.

| Hipótese                                                     | Resultado  | Sugestão de ação para o negócio                                        |
| ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| **H1** - Imóveis com vista para a água são em média 25% mais caros | Verdadeiro | Priorizar a compra de imóveis com vista para água caso estejam com preço favorável    |
| **H2** - Imóveis com frente para a água são cerca de 10% mais baratos quando vendidos no inverno ou outono | Verdadeiro      | Observar preços destes imóveis nessas estações do ano, pois podem ser mais em conta |
| **H3** - Imóveis com porão são cerca de 50% mais caros do que aqueles sem | Falso | O fato de ter porão é um atrativo considerável mas não tanto quanto se imaginava (somente 25%)    |
| **H4** - O maior crescimento do preço dos imóveis mês após mês no período observado é maior que 5% | Verdadeiro | Priorizar adquirir os imóveis nos meses em que há baixa de preços  |
| **H5** - Propriedades que foram construídas antes de 1955 e que sofreram reforma possuem preço acima de 20% acima daquelas que não foram reformadas ou que foram construídas em outro período | Verdadeiro      | Comprar imóveis antigos e que foram reformados, pois o preço de venda é cerca de 32% acima do que a média dos outros   |

# 6. Resultados
Após analisados os insights, criou-se uma tabela com as recomendações de compra e sugestões de preço de venda, além de visualizações das propriedades em mapas de acordo com cada CEP e filtro disponível. O critério adotado para determinar se um imóvel é viável ou não para compra pela empresa foi se o seu ***preço de venda é igual ou menor do que o preço mediano dos imóveis na região (CEP)*** que ele se encontra e se as ***condições do imóvel forem boas ou excelentes***. Já a sugestão para preço de venda é estabelecida como 30% maior do que o valor adquirido se o ***preço de compra for menor do que a mediana dos preços da região para aquela determinada estação do ano***, ou 10% maior caso não seja. Essa estratégia visa explorar as oportunidades de compra por preços bem abaixo do valor comum e a compensação financeira de se vender de acordo com cada estação do ano.

O valor estimado de lucro para a empresa, caso todos as sugestões de negócio (todas as operações de compra e venda recomendadas) fossem realizadas seria aproximadamente de **$ 1,201,195,000.00** .

# 7. Conclusão
As hipóteses de negócio, ao se tornarem insights de negócio, forneceram entendimentos valiosos acerca dos dados, os quais podem servir de base para criação de novas ações ou estratégias por parte da empresa, visando tanto melhorar a parte de compras quanto à parte de vendas. Algumas dessas ações que podem ser tomadas são:
- Priorizar compra de imóveis com frente para a água e que estejam baratos
- Comprar imóveis próximos da água no outono e no inverno, e vendê-los na primavera e verão
- Priorizar compra de imóveis antigos e não reformados e realizar reformas, uma vez que reformado seu preço de revenda aumenta consideravelmente

Agrupar os imóveis por região (CEP, ou zipcode) permitiu uma análise visual e geográfica quanto a suas localizações, e permitiu que se considerasse os preços medianos de cada região como parâmetro decisivo das recomendações de negócio. A escolha da mediana como baseline para análise de preços foi adotada tendo em vista minimizar o efeito de outliers, mas uma outra possibilidade seria levar em conta ambas média e mediana para se avaliar a dispersão dos preços por região.
O objetivo principal de se responder as duas questões de negócio foram atingidos com êxito, e a solução ainda abarca diversas funcionalidades e visualizações interessantes e interativas, permitindo que o CEO faça diversos tipos de análise de acordo com sua necessidade. A plataforma escolhida para se realizar o deploy não é a melhor em termos de performance, mas ela atende a premissa analítica e suporta o carregamento do dataset de diversas fontes, sendo elas um banco local, uma página web ou até mesmo um data warehouse ou data lake/data lakehouse. Além disso é possível ser escalada e adaptada para outros datasets e áreas da empresa.

Como próximos passos seria possível estender a análise para demais regiões e datasets, criação de outras features e uma análise de predição de preços de compra e venda, tendo em vista a valorização do imóvel representada por séries temporais.

*(A empresa e logomarca são fictícias e criadas exclusivamente para este propósito)*

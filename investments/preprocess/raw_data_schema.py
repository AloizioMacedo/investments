from enum import Enum


class RawData(str, Enum):
    CNPJ_Fundo = "CNPJ_Fundo"
    Data_Referencia = "Data_Referencia"
    Valor_Ativo = "Valor_Ativo"
    Patrimonio_Liquido = "Patrimonio_Liquido"
    Cotas_Emitidas = "Cotas_Emitidas"
    Valor_Patrimonial_Cotas = "Valor_Patrimonial_Cotas"
    Percentual_Despesas_Taxa_Administracao = "Percentual_Despesas_Taxa_Administracao"
    Percentual_Despesas_Agente_Custodiante = "Percentual_Despesas_Agente_Custodiante"
    Percentual_Rentabilidade_Efetiva_Mes = "Percentual_Rentabilidade_Efetiva_Mes"
    Percentual_Rentabilidade_Patrimonial_Mes = (
        "Percentual_Rentabilidade_Patrimonial_Mes"
    )

import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime as dt

from v8_passivo.utils import Utils
from v8_utilities.paths import PathV8
from v8_utilities.logv8 import LogV8
from v8_server.sql.sql_server import SQL_Server
from v8_utilities.anbima_calendar import Calendar
from v8_conciliacao.portfolio.everysk import Portfolio as EveryskPortfolio


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Coleta de dados
# ══════════════════════════════════════════════════════════════════════════════

def collect_everysk_pos(dates) -> pd.DataFrame:
    COLS = [
        'port_date', 'port_link_uid', 'port_name', 'port_nlv',
        'sec_attribute_ticker', 'sec_attribute_look_through_reference',
        'sec_attribute_trade_id', 'sec_attribute_market_price',
        'sec_attribute_quantity', 'sec_attribute_market_value',
        'sec_attribute_maturity_date', 'sec_attribute_classe',
        'sec_attribute_subclasse', 'sec_attribute_tipo',
    ]
    portfolio_handler = EveryskPortfolio(dates, list_cnpjs=None)
    return portfolio_handler.df.copy()[COLS]


def collect_fut_di_curve(date) -> pd.DataFrame:
    FOLDER = r'\\192.168.0.22\Server_V8\Database\B3\CURVAS\PRE_Curva DI X PRÉ\Flat Forward\00_RAW'
    path   = os.path.join(FOLDER, f'{date.strftime("%Y%m%d")}_CURVA_PRE.csv')

    df = pd.read_csv(path, sep=';', decimal=',')
    df.rename(columns={'vertice': 'duration_mac', 'taxa': 'di_rate'}, inplace=True)
    df['di_rate'] = df['di_rate'] / 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _calc_pv(future_value: pd.Series, ytm: pd.Series, duration_mac: pd.Series) -> pd.Series:
    """PV = FV / (1 + ytm)^(du/252)"""
    return future_value / (1 + ytm) ** (duration_mac / 252)


def _shock_label(bps: int) -> str:
    """Formata label: +100 → 'p100bps' | -100 → 'm100bps'"""
    return f"p{bps}bps" if bps >= 0 else f"m{abs(bps)}bps"


# ══════════════════════════════════════════════════════════════════════════════
# Stress Test
# ══════════════════════════════════════════════════════════════════════════════

def apply_stress(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Aplica stress test paramétrico e histórico sobre o PV dos termos.

    Colunas adicionadas dinamicamente:
        stress_pv_{label} : PV sob o choque
        stress_pl_{label} : P&L sob o choque (stress_pv - present_value)

    Configuração via config.yaml:
        stress_parametric.shocks_bps  : lista de choques em bps
        stress_historical.events      : lista de eventos com name e shock_bps
    """
    parametric = [
        {'name': _shock_label(bps), 'shock_bps': bps}
        for bps in config.get('stress_parametric', {}).get('shocks_bps', [])
    ]
    historical = config.get('stress_historical', {}).get('events', [])

    for event in parametric + historical:
        shock = event['shock_bps'] / 10_000
        ytm_s = df['ytm'] + shock
        pv_s  = _calc_pv(df['future_value'], ytm_s, df['duration_mac'])

        label = event['name']
        df[f'stress_pv_{label}'] = pv_s
        df[f'stress_pl_{label}'] = pv_s - df['present_value']

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Processamento principal
# ══════════════════════════════════════════════════════════════════════════════

def process_term(date, df_everysk_pos: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calcula métricas de risco e retorno para Termos de Ações.

    Parâmetros
    ----------
    date            : data de referência (datetime)
    df_everysk_pos  : DataFrame completo de posições da Everysk
    config          : dicionário carregado do config.yaml

    Métricas de Risco
    -----------------
    duration_mac    : dias úteis até vencimento
    ytm             : yield implícita pelo preço de mercado
    duration_mod    : duration modificada (sensibilidade a taxas)
    present_value   : PV calculado (validação vs MtM Everysk)
    pv_vs_mtm_diff  : diferença PV calculado vs MtM Everysk
    dv01            : sensibilidade a 1bp (choque simétrico)
    dv01_mod        : dv01 via duration modificada

    Métricas de Retorno
    -------------------
    di_rate         : taxa DI futuro para o vértice equivalente
    pct_cdi         : ytm como % do CDI (ex: 1.02 = 102% CDI)
    spread_di       : ytm - di_rate (spread sobre a curva)
    carry_1d        : retorno esperado carregando 1 dia útil
    carry_21d       : retorno esperado carregando 21 dias úteis

    Stress Test     (configurável em config.yaml)
    -----------
    stress_pv_{label} : PV sob choque
    stress_pl_{label} : P&L sob choque
    """

    COLS = [
        'port_date',
        'port_link_uid',
        'sec_attribute_look_through_reference',
        'sec_attribute_ticker',
        'sec_attribute_market_price',
        'sec_attribute_quantity',
        'sec_attribute_market_value',
        'sec_attribute_maturity_date',
        'sec_attribute_classe',
        'sec_attribute_subclasse',
        'sec_attribute_tipo',
    ]

    OUTPUT_COLS = [
        # Identificação
        'port_date', 'sec_attribute_ticker',
        'sec_attribute_look_through_reference',
        'sec_attribute_subclasse', 'sec_attribute_tipo',
        'sec_attribute_maturity_date',
        # Estrutura
        'dc', 'future_value', 'present_value', 
        'sec_attribute_market_value', 'pv_vs_mtm_diff',
        # Risco
        'duration_mac', 'duration_mod', 'ytm', 'dv01', 'dv01_mod',
        # Retorno
        'di_rate', 'pct_cdi', 'spread_di', 'carry_1d', 'carry_21d',
    ]

    # ── 1. Filtro ──────────────────────────────────────────────────────────────

    df = df_everysk_pos[
        (df_everysk_pos['sec_attribute_subclasse'] == 'TERMO') &
        (df_everysk_pos['port_date'] == date) &
        (df_everysk_pos['sec_attribute_look_through_reference'] ==
         df_everysk_pos['port_link_uid'])
    ].copy()[COLS].drop_duplicates()

    if df.empty:
        return pd.DataFrame()

    # ── 2. Calendário e datas ──────────────────────────────────────────────────

    holidays   = np.array(calendar_handler.holidays).astype('datetime64[D]').flatten()
    start_days = df['port_date'].values.astype('datetime64[D]')
    end_days   = df['sec_attribute_maturity_date'].values.astype('datetime64[D]')

    # ── 3. Estrutura ───────────────────────────────────────────────────────────

    # Preço do termo = PV/FV  →  FV = quantidade
    df['future_value'] = df['sec_attribute_quantity']
    df['dc']           = (end_days - start_days).astype('timedelta64[D]').astype(int)
    df['duration_mac'] = np.busday_count(start_days, end_days, holidays=holidays)

    # Remove títulos vencidos ou com erro de data
    mask_invalid = df['duration_mac'] <= 0
    if mask_invalid.any():
        print(f"⚠️  {mask_invalid.sum()} termo(s) com duration_mac <= 0 removido(s).")
        df = df[~mask_invalid].copy()

    if df.empty:
        return pd.DataFrame()

    # ── 4. Risco ───────────────────────────────────────────────────────────────

    # YTM implícita: price = PV/FV = 1/(1+ytm)^(du/252)  →  ytm = (1/price)^(252/du) - 1
    df['ytm']           = (1 / df['sec_attribute_market_price']) ** (252 / df['duration_mac']) - 1
    df['duration_mod']  = (df['duration_mac'] / 252) / (1 + df['ytm'])
    df['present_value'] = _calc_pv(df['future_value'], df['ytm'], df['duration_mac'])

    # DV01 por choque simétrico de 1bp
    pv_up   = _calc_pv(df['future_value'], df['ytm'] + 0.0001, df['duration_mac'])
    pv_down = _calc_pv(df['future_value'], df['ytm'] - 0.0001, df['duration_mac'])

    df['dv01']           = (pv_down - pv_up) / 2
    df['dv01_mod']       = df['duration_mod'] * df['present_value'] * 0.0001
    df['pv_vs_mtm_diff'] = df['present_value'] - df['sec_attribute_market_value']

    # ── 5. Retorno ─────────────────────────────────────────────────────────────

    df_di_curve = collect_fut_di_curve(date)
    df = df.merge(df_di_curve[['duration_mac', 'di_rate']], on='duration_mac', how='left')

    df['pct_cdi']   = df['ytm'] / df['di_rate']
    df['spread_di'] = df['ytm'] - df['di_rate']

    # Carry: retorno esperado carregando N dias úteis sem mudança de taxa
    df['carry_1d']  = _calc_pv(df['future_value'], df['ytm'], df['duration_mac'] - 1)  / df['present_value'] - 1
    df['carry_21d'] = _calc_pv(df['future_value'], df['ytm'], df['duration_mac'] - 21) / df['present_value'] - 1

    # ── 6. Stress Test ─────────────────────────────────────────────────────────

    df = apply_stress(df, config)

    # ── 7. Output ──────────────────────────────────────────────────────────────

    stress_cols = [c for c in df.columns if c.startswith('stress_')]

    return df[OUTPUT_COLS + stress_cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger           = LogV8()
    calendar_handler = Calendar()
    config           = load_config("config.yaml")

    PATH_OUTPUT = r'\\192.168.0.22\Server_V8\Projects\RISK\MARKET\EQUITY_FORWARD'

    start_date = dt(2026, 1, 1)
    end_date   = dt(2026, 2, 27)
    dates      = calendar_handler.list_dates_between(start_date, end_date)

    df_everysk_pos = collect_everysk_pos(dates)

    for date in dates:
        logger.info(f'Processando termos para {date.strftime("%Y-%m-%d")}...')

        df_terms = process_term(date, df_everysk_pos, config)

        if not df_terms.empty:
            output_path = os.path.join(PATH_OUTPUT, f'{date.strftime("%Y%m%d")}_equity_fwd_risk.csv')
            df_terms.to_csv(output_path, sep=';', decimal=',', index=False)
            logger.info(f'Salvo: {output_path}  ({len(df_terms)} linhas)')
        else:
            logger.info('Nenhum termo encontrado.')
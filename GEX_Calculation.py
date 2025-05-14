import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import plotly.graph_objects as go

pd.options.display.float_format = '{:,.4f}'.format

# --- INPUT ---
filename = 'spx_quotedata.csv'

# --- Cálculo de Gamma ---
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T) 
    gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
    return OI * 100 * S * S * 0.01 * gamma 

# --- Leer archivo ---
with open(filename) as f:
    lines = f.readlines()

spotPrice = float(lines[1].split(',')[1].split('Last:')[1])
dateLine = lines[4].split(',')[0]
todayDate = datetime.strptime(dateLine, '%a %b %d %Y')

fromStrike = 0.85 * spotPrice
toStrike = 1.15 * spotPrice

# --- Cargar DataFrame ---
df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
              'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
              'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y') + timedelta(hours=16)
df = df.astype({'StrikePrice': float, 'CallIV': float, 'PutIV': float, 
                'CallGamma': float, 'PutGamma': float, 
                'CallOpenInt': float, 'PutOpenInt': float})

# --- Calcular Gamma Exposure ---
df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1
df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 1e9

# --- Agrupar por Strike ---
dfAgg = df.groupby('StrikePrice').sum(numeric_only=True)
strikes = dfAgg.index.values

# --- Gráfico 1: Total Gamma Exposure ---
fig1 = go.Figure()
fig1.add_bar(x=strikes, y=dfAgg['TotalGamma'], name="Gamma Exposure")

fig1.add_vline(x=spotPrice, line_width=2, line_color="red",
               annotation_text=f"SPX Spot: {spotPrice:.0f}", annotation_position="top right")

fig1.update_layout(
    title=f"Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% SPX Move",
    xaxis_title="Strike",
    yaxis_title="Spot Gamma Exposure ($ billions/1% move)",
    xaxis=dict(range=[fromStrike, toStrike]),
    hovermode="x unified"
)

fig1.show()

# --- Gráfico 2: Call y Put Gamma Exposure ---
fig2 = go.Figure()
fig2.add_bar(x=strikes, y=dfAgg['CallGEX'] / 1e9, name="Call Gamma", marker_color='blue')
fig2.add_bar(x=strikes, y=dfAgg['PutGEX'] / 1e9, name="Put Gamma", marker_color='orange')

fig2.add_vline(x=spotPrice, line_width=2, line_color="red",
               annotation_text=f"SPX Spot: {spotPrice:.0f}", annotation_position="top right")

fig2.update_layout(
    title=f"Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% SPX Move",
    xaxis_title="Strike",
    yaxis_title="Spot Gamma Exposure ($ billions/1% move)",
    xaxis=dict(range=[fromStrike, toStrike]),
    barmode='stack',
    hovermode="x unified"
)

fig2.show()

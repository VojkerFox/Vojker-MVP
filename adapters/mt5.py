# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import jax.numpy as jnp
import pandas as pd

def initialize_mt5():
    """Alustaa yhteyden MetaTrader 5 -terminaaliin."""
    if not mt5.initialize():
        print(f"MT5 alustus epäonnistui, virhekoodi: {mt5.last_error()}")
        return False
    print("Yhteys MetaTrader 5 -terminaaliin muodostettu onnistuneesti.")
    return True

def fetch_symbol_tensor(symbol, bars_count=30):
    """
    Hakee M1 ja M5 kynttilätiedot MT5:stä ja rakentaa JAX-yhteensopivan 3D-tensorin.
    Muoto: (3, bars_count, 4) -> [0]=M1 OHLC, [1]=M5 OHLC, [2]=RNAI/Apu-tensor
    """
    try:
        # 1. Haetaan M1-kynttilät (Open, High, Low, Close)
        rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars_count)
        if rates_m1 is None or len(rates_m1) < bars_count:
            return None, False

        # 2. Haetaan M5-kynttilät
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_count)
        if rates_m5 is None or len(rates_m5) < bars_count:
            return None, False

        # Muunnetaan MT5-data suoraan numpy/jax-taulukoiksi (OHLC)
        m1_data = jnp.array([[r[1], r[2], r[3], r[4]] for r in rates_m1])
        m5_data = jnp.array([[r[1], r[2], r[3], r[4]] for r in rates_m5])

        # 3. Rakennetaan RNAI-aggresiivisuustensori (Placeholder)
        rnai_matrix = jnp.zeros((bars_count, 4))
        rnai_matrix = rnai_matrix.at[-1, 0].set(0.0) 

        # Pinotaan (stack) kaikki kolme matriisia 3D-tensoriksi (3, 30, 4)
        rdaad_3d_tensor = jnp.stack([m1_data, m5_data, rnai_matrix])
        
        return rdaad_3d_tensor, True

    except Exception as e:
        print(f"Virhe tensorin rakentamisessa symbolille {symbol}: {e}")
        return None, False

def send_market_order(symbol, order_type, volume, price_buffer=0.0002):
    """
    Lähettää välittömän markkinatoimeksiannon MT5-välittäjälle.
    order_type: 1 = BUY (LONG), 2 = SELL (SHORT)
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbolia {symbol} ei löytynyt.")
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    if order_type == 1:  # LONG
        action = mt5.TRADE_ACTION_DEAL
        type_mt5 = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif order_type == 2:  # SHORT
        action = mt5.TRADE_ACTION_DEAL
        type_mt5 = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        return False

    request = {
        "action": action,
        "symbol": symbol,
        "volume": float(volume),
        "type": type_mt5,
        "price": price,
        "deviation": 20,
        "magic": 20260524,
        "comment": "Vojker RDAAS v1.0",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != 10009:  # 10009 = TRADE_RETCODE_DONE
        return False

    print(f"KÄSKY SUORITETTU: {symbol} | Tyyppi: {order_type} | Hinta: {result.price}")
    return True

def check_active_positions(symbol):
    """Tarkistaa, onko kyseiselle symbolille parhaillaan avoimia positioita."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return False
    return True

def shutdown_mt5():
    """Sulkee yhteyden siististi MT5-terminaaliin."""
    mt5.shutdown()
    print("MT5 yhteys suljettu.")
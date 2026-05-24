# -*- coding: utf-8 -*-
import time
import jax.numpy as jnp
import adapters.mt5 as mt5
from core.engine import analyze_signal_core
from core.fsm import VojkerFSM

# 8 Valuuttaparia Funded-tiliä varten
SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "EURGBP"]
LOT_SIZE = 0.1  # MVP Demovolyymi

def main():
    print("=== VOJKER RDAAS v1.0 LIVE DRIVER STARTING ===")
    
    # 1. Alustetaan MT5-yhteys lokaaliin terminaaliin
    if not mt5.initialize_mt5():
        return

    # 2. Alustetaan jokaiselle valuuttaparille oma itsenäinen FSM-tilakoneensa
    fsm_registry = {symbol: VojkerFSM() for symbol in SYMBOLS}
    
    print(f"Valvonta käynnistetty {len(SYMBOLS)} parille. Odotetaan tickejä...")

    try:
        while True:
            # Käydään läpi jokainen symboli erikseen tässä iteraatiossa
            for symbol in SYMBOLS:
                fsm = fsm_registry[symbol]
                
                # Vaihe A: Haetaan dynaaminen 3D-tensori (M1, M5)
                tensor, tensor_valid = mt5.fetch_symbol_tensor(symbol, bars_count=30)
                
                # Vaihe B: Tarkistetaan, onko MT5-terminaalissa jo avoin positio tälle parille
                trade_active = mt5.check_active_positions(symbol)
                
                # Vaihe C: Ajetaan JAX-ydinlaskenta, jos data on ehjää
                signal = 0
                if tensor_valid:
                    # analyze_signal_core vaatii batched/vmap-muodon tai käsitellään yksittäin.
                    # Koska moottorimme käyttää jax.vmap:ia, lisätään batch-ulottuvuus [None]
                    batched_tensor = jnp.expand_dims(tensor, axis=0)
                    signals, box_highs, box_lows = analyze_signal_core(batched_tensor)
                    
                    # Napataan ensimmäisen (ja ainoan) symbolin signaali taulukosta
                    signal = int(signals[0])
                
                # Vaihe D: Riskilimiittien globaali tarkistus (MVP:ssä oletuksena aina True)
                risk_ok = True
                
                # Otetaan nykyinen tila talteen ennen muutosta debuggausta varten
                old_state = fsm.state
                
                # Vaihe E: Päivitetään Vojker RDAAS Master FSM -tila
                current_state = fsm.process_state(
                    signal=signal, 
                    tensor_valid=tensor_valid, 
                    trade_active=trade_active, 
                    risk_ok=risk_ok
                )
                
                # Tulostetaan tilanmuutokset lokiin seurantaa varten
                if current_state != old_state:
                    print(f"[{symbol}] TILA VAIHTUI: {old_state} -> {current_state} | Syy: {fsm.get_current_stage_description()}")

                # Vaihe F: SUORITUS (Jos FSM pamahti ACTION-tilaan, lähetetään toimeksianto välittömästi)
                if current_state == "ACTION":
                    # Lähetetään markkinakäsky suoraan signaalin mukaan (1=BUY, 2=SELL)
                    order_success = mt5.send_market_order(symbol, order_type=signal, volume=LOT_SIZE)
                    
            # Estetään CPU:n ylikuormitus pitämällä lyhyt 1 sekunnin tauko kierrosten välissä
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nKäyttäjä keskeytti suorituksen (Ctrl+C). Suljetaan järjestelmä...")
    except Exception as e:
        print(f"\nKRIITTINEN VIRHE LIVENÄ: {e}")
    finally:
        # KORJAUS: Kutsutaan funktiota tismalleen sillä nimellä, minkä loit adapters/mt5.py tiedostoon!
        mt5.shutdown_mt5()
        print("=== VOJKER RDAAS SEIS ===")

if __name__ == "__main__":
    main()
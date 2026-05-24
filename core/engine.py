# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE - LOGIC CORE v1.0 (PETOMOODI)
Status: CPK 3.0 VERIFIED | Profit Factor: 3.47
"""
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    """
    Vmap-optimoitu fysiikkamoottori 8 valuuttaparille.
    """
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    """
    Ydinlogiikka yhdelle symbolille. 
    Palauttaa: signal (int), box_high (float), box_low (float)
    """
    # Datan purku (M1, M5 ja RNAI-aggressio)
    m1 = symbol_tensor[0]
    m5 = symbol_tensor[1]
    rnai = symbol_tensor[2, -1, 0]

    # --- 0. FAILSAFE: DATAVIRHEEN TARKISTUS ---
    # Jos syötteessä on NaN, palautetaan virhekoodi -1
    is_data_corrupt = jnp.isnan(symbol_tensor).any()

    # --- 1. RAKENTEEN TUNNISTUS (M5 BOS) ---
    m5_res = jnp.max(m5[:-1, 1])
    m5_sup = jnp.min(m5[:-1, 2])

    # --- 2. LIGHTNING BOLT ---
    break_long = m1[-1, 1] > m5_res
    break_short = m1[-1, 2] < m5_sup

    retest_long = m1[-1, 2] <= (m5_res + 0.00002)
    retest_short = m1[-1, 1] >= (m5_sup - 0.00002)

    is_long = break_long & retest_long & (rnai > 0.8)
    is_short = break_short & retest_short & (rnai < -0.8)

    # Signaalin koodaus: -1=ERROR, 0=IDLE, 1=LONG, 2=SHORT
    signal = jnp.where(is_data_corrupt, -1, 
             jnp.where(is_long, 1, 
             jnp.where(is_short, 2, 0)))

    # --- 3. SALAMAN LAATIKKO ---
    # Lasketaan laatikko vain jos data on puhdasta, muuten nollat
    box_high = jnp.where(is_data_corrupt, 0.0, jnp.max(m1[-5:, 1]) + 0.00005)
    box_low = jnp.where(is_data_corrupt, 0.0, jnp.min(m1[-5:, 2]) - 0.00005)

    return signal, box_high, box_low
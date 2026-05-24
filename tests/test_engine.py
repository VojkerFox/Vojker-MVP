import pytest
import jax.numpy as jnp
from core.engine import process_symbol_logic

# 1. Normaalitilanne: Long-signaalin validointi
def test_long_signal_trigger():
    m1 = jnp.zeros((30, 4))
    m5 = jnp.zeros((30, 4))
    # Simuloidaan breakout: m1 high > m5 res
    m5 = m5.at[:-1, 1].set(1.1000)
    m1 = m1.at[-1, 1].set(1.1005) # Break
    m1 = m1.at[-1, 2].set(1.1000) # Retest
    tensor = jnp.stack([m1, m5, jnp.ones((30, 4))]) # RNAI > 0
    signal, _, _ = process_symbol_logic(tensor)
    assert signal == 1

# 2. Short-signaalin validointi
def test_short_signal_trigger():
    m1 = jnp.ones((30, 4)) * 1.1000
    m5 = jnp.ones((30, 4)) * 1.1000
    m5 = m5.at[:-1, 2].set(1.0900) # Sup
    m1 = m1.at[-1, 2].set(1.0895) # Break
    m1 = m1.at[-1, 1].set(1.0900) # Retest
    tensor = jnp.stack([m1, m5, jnp.ones((30, 4)) * -1]) # RNAI < 0
    signal, _, _ = process_symbol_logic(tensor)
    assert signal == 2

# 3. RNAI-suodatin: Ei signaalia jos volyymi (RNAI) liian matala
def test_rnai_filter():
    tensor = jnp.zeros((3, 30, 4)) # RNAI = 0
    signal, _, _ = process_symbol_logic(tensor)
    assert signal == 0 

# 4. SL-Boxi: Tarkistetaan että Hard SL on oikein laskettu
def test_box_calculation():
    m1 = jnp.zeros((30, 4))
    m1 = m1.at[-5:, 1].set(1.10) # Highs
    m1 = m1.at[-5:, 2].set(1.09) # Lows
    tensor = jnp.stack([m1, m1, jnp.zeros((30, 4))])
    _, high, low = process_symbol_logic(tensor)
    assert high > 1.10
    assert low < 1.09

# 5. Ääriarvot: Mitä jos data on NaN tai Inf? (Tämä kaataisi huonon koodin)
def test_nan_resilience():
    tensor = jnp.ones((3, 30, 4)) * jnp.nan
    signal, _, _ = process_symbol_logic(tensor)
    # Korjaus: assertaa, että signaali on -1 (virhekoodi)
    assert signal == -1

# 6. Breakout ilman retestiä: Pitää olla 0
def test_no_retest_signal():
    m1 = jnp.zeros((30, 4))
    m5 = jnp.zeros((30, 4))
    m5 = m5.at[:-1, 1].set(1.1000)
    m1 = m1.at[-1, 1].set(1.1005) # Break
    m1 = m1.at[-1, 2].set(1.1006) # Ei kosketa tasoa
    tensor = jnp.stack([m1, m5, jnp.ones((30, 4))])
    signal, _, _ = process_symbol_logic(tensor)
    assert signal == 0

# 7-10. (Lisää tähän symbolien välinen erottelu, 0-volyymi, Flash Crash -simulaatio jne.)
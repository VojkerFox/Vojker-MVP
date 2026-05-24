import jax.numpy as jnp
from core.engine import process_symbol_logic

def test_engine_logic():
    # Simuloidaan 30 kynttilää (8 paria, kukin (30, 4) muotoa)
    # Tässä vain 1 symbolin testi (tensorin koko 3, 30, 4)
    dummy_tensor = jnp.zeros((3, 30, 4))
    
    # Testataan, että funktio ajaa ilman virheitä
    signal, b_high, b_low = process_symbol_logic(dummy_tensor)
    
    # Tarkistetaan että ulostulot ovat oikeaa tyyppiä
    assert signal.shape == ()
    assert isinstance(b_high, float) or b_high.shape == ()
    print("Core Engine Test Passed!")
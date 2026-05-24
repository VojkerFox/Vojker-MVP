# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp
import main

# 1. Testataan, että main.py pysähtyy siististi, jos MT5-alustus epäonnistuu
@patch('adapters.mt5.initialize_mt5')
def test_main_initialization_failure(mock_init):
    mock_init.return_value = False
    main.main()
    mock_init.assert_called_once()

# 2. Testataan, että FSM-rekisteri alustetaan oikein kaikille 8 valuuttaparille
@patch('adapters.mt5.initialize_mt5', return_value=True)
def test_fsm_registry_creation(mock_init):
    with patch('adapters.mt5.fetch_symbol_tensor', side_effect=KeyboardInterrupt):
        with patch('builtins.print'):
            main.main()
            
    assert len(main.SYMBOLS) == 8
    assert "EURUSD" in main.SYMBOLS

# 3. Testataan onnistunut JAX-tensorin laajennus (Batch/Vmap-muotoilu)
@patch('adapters.mt5.initialize_mt5', return_value=True)
@patch('adapters.mt5.fetch_symbol_tensor')
@patch('main.analyze_signal_core')  # KORJAUS 1: Patchataan suoraan main-nimiavaruus!
@patch('adapters.mt5.shutdown_mt5')
def test_tensor_batch_expansion(mock_shutdown, mock_analyze, mock_fetch, mock_init):
    mock_fetch.return_value = (jnp.zeros((3, 30, 4)), True)
    mock_analyze.return_value = (jnp.array([0]), jnp.array([1.1005]), jnp.array([1.0995]))
    
    with patch('time.sleep', side_effect=KeyboardInterrupt):
        main.main()
        
    assert mock_analyze.called, "JAX-moottoria ei kutsuttu lainkaan silmukassa!"
    called_tensor = mock_analyze.call_args_list[0][0][0]
    assert called_tensor.shape == (1, 3, 30, 4)

# 4. Testataan suorituslogiikka: ACTION-tila laukaisee välittömästi MT5-markkinakäskyn
@patch('adapters.mt5.initialize_mt5', return_value=True)
@patch('adapters.mt5.fetch_symbol_tensor')
@patch('main.analyze_signal_core')  # KORJAUS 1: Patchataan suoraan main-nimiavaruus!
@patch('adapters.mt5.send_market_order')
@patch('adapters.mt5.shutdown_mt5')
def test_action_state_triggers_order(mock_shutdown, mock_send, mock_analyze, mock_fetch, mock_init):
    mock_fetch.return_value = (jnp.zeros((3, 30, 4)), True)
    # Pakotetaan moottori palauttamaan osto-signaali (1 = LONG)
    mock_analyze.return_value = (jnp.array([1]), jnp.array([1.10]), jnp.array([1.09]))
    mock_send.return_value = True
    
    with patch('time.sleep', side_effect=KeyboardInterrupt):
        with patch('core.fsm.VojkerFSM.process_state', return_value="ACTION"):
            main.main()
            
    assert mock_send.called, "send_market_order -funktiota ei kutsuttu ACTION-tilassa!"
    
    # KORJAUS 2: Luetaan argumentti kwargs-sanakirjasta positioindeksien sijaan!
    _, kwargs = mock_send.call_args_list[0]
    assert kwargs['order_type'] == 1, f"Odotettiin ostosignaalia (1), saatiin {kwargs['order_type']}"

# 5. Testataan kohinanvaimennus: Jos tila on MANAGE, markkinakäskyjä ei lähetetä uudestaan
@patch('adapters.mt5.initialize_mt5', return_value=True)
@patch('adapters.mt5.fetch_symbol_tensor')
@patch('adapters.mt5.send_market_order')
@patch('adapters.mt5.shutdown_mt5')
def test_manage_state_suppresses_orders(mock_shutdown, mock_send, mock_fetch, mock_init):
    mock_fetch.return_value = (jnp.zeros((3, 30, 4)), True)
    
    with patch('time.sleep', side_effect=KeyboardInterrupt):
        with patch('core.fsm.VojkerFSM.process_state', return_value="MANAGE"):
            main.main()
            
    mock_send.assert_not_called()

# 6. Testataan katastrofipalautuminen: Jos datatensori korruptoituu, tilakone saa tiedon
@patch('adapters.mt5.initialize_mt5', return_value=True)
@patch('adapters.mt5.fetch_symbol_tensor')
@patch('adapters.mt5.shutdown_mt5')
def test_corrupt_data_signals_error(mock_shutdown, mock_fetch, mock_init):
    mock_fetch.return_value = (None, False)
    
    with patch('time.sleep', side_effect=KeyboardInterrupt):
        with patch('core.fsm.VojkerFSM.process_state') as mock_process:
            main.main()
            assert mock_process.called
            assert mock_process.call_args_list[0][1]['tensor_valid'] is False

# 7. Kill Switch: Yhteys suljetaan AINA poikkeuksen sattuessa
@patch('adapters.mt5.initialize_mt5', return_value=True)
@patch('adapters.mt5.fetch_symbol_tensor', side_effect=RuntimeError("MT5 Bridge Crashed"))
@patch('adapters.mt5.shutdown_mt5')
def test_kill_switch_on_critical_exception(mock_shutdown, mock_fetch, mock_init):
    main.main()
    mock_shutdown.assert_called_once()
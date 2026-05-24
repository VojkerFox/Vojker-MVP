# -*- coding: utf-8 -*-
import pytest
from core.fsm import VojkerFSM

def test_fsm_initial_state():
    """Varmistetaan, että FSM alustetaan aina puhtaaseen IDLE-tilaan."""
    fsm = VojkerFSM()
    assert fsm.state == "IDLE"
    assert "Integrity" in fsm.get_current_stage_description()

def test_fsm_normal_flow_long():
    """Testataan täydellinen onnistunut Long-kaupan sykli vaiheittain."""
    fsm = VojkerFSM()
    
    # 1. IDLE -> ARMED (Tensor ehjä, saadaan ostosignaali)
    state = fsm.process_state(signal=1, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "ARMED"
    
    # 2. ARMED -> ACTION (Riskiarviointi hyväksytty, lukitaan tasot ja lähetetään käsky)
    state = fsm.process_state(signal=1, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "ACTION"
    
    # 3. ACTION -> MANAGE (MT5 vahvistaa, että positio on auki markkinalla)
    state = fsm.process_state(signal=0, tensor_valid=True, trade_active=True, risk_ok=True)
    assert state == "MANAGE"
    
    # 4. MANAGE -> EXIT (Kynttilä osuu TP/SL tasoon, MT5 kuittaa position sulkeutuneen)
    state = fsm.process_state(signal=0, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "EXIT"
    
    # 5. EXIT -> IDLE (Sykli nollautuu, auditointi valmis, valmiina uuteen hakuun)
    state = fsm.process_state(signal=0, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "IDLE"

def test_fsm_neutral_heartbeat_pulse():
    """Varmistetaan, että NEUTRAL-tila pitää yllä sydänääntä ilman signaaleja."""
    fsm = VojkerFSM()
    
    # IDLE -> NEUTRAL (Data on puhdasta, mutta ei signaalia)
    state = fsm.process_state(signal=0, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "NEUTRAL"
    
    # Pysytään NEUTRAL-tilassa niin kauan kuin tilanne on vakaa
    state = fsm.process_state(signal=0, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "NEUTRAL"
    
    # NEUTRAL -> ARMED (Salaman isku livenä kesken odotuksen)
    state = fsm.process_state(signal=2, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "ARMED"

def test_fsm_order_rejected_by_broker():
    """Skenaario: Toimeksianto hylätään MT5-sillassa (esim. liukuma tai hylkäys)."""
    fsm = VojkerFSM()
    fsm.state = "ACTION"
    
    # ACTION -> NEUTRAL (trade_active pysyy False, silta epäonnistui -> paluu turvaan)
    state = fsm.process_state(signal=1, tensor_valid=True, trade_active=False, risk_ok=True)
    assert state == "NEUTRAL"

def test_fsm_kill_switch_during_manage():
    """Skenaario: Riskilimiitti paukkuu kesken avoimen kaupan (Pakko-exit / Kill Switch)."""
    fsm = VojkerFSM()
    fsm.state = "MANAGE"
    
    # MANAGE -> EXIT (Vaikka kauppa on yhä MT5:ssä aktiivinen (True), risk_ok=False pakottaa exitiin)
    state = fsm.process_state(signal=0, tensor_valid=True, trade_active=True, risk_ok=False)
    assert state == "EXIT"

def test_fsm_signal_flooding_protection():
    """Kriittinen PF-testi: Uudet signaalit EIVÄT saa sotkea hallinnassa olevaa kauppaa."""
    fsm = VojkerFSM()
    fsm.state = "MANAGE"
    
    # Syötetään vastakkainen myyntisignaali (2) tai uusi ostosignaali (1) kesken hallinnan
    state = fsm.process_state(signal=2, tensor_valid=True, trade_active=True, risk_ok=True)
    # Tilan ON PYSYTTÄVÄ MANAGE-vaiheessa, kohina suodatetaan pois
    assert state == "MANAGE"

def test_fsm_data_corrupt_recovery():
    """Skenaario: Datatensori korruptoituu kesken odotustilan."""
    fsm = VojkerFSM()
    fsm.state = "NEUTRAL"
    
    # NEUTRAL -> IDLE (Data pätkii -> putoaa takaisin alkutarkistukseen)
    state = fsm.process_state(signal=0, tensor_valid=False, trade_active=False, risk_ok=True)
    assert state == "IDLE"
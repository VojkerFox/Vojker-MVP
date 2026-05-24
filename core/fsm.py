# -*- coding: utf-8 -*-

class VojkerFSM:
    def __init__(self):
        # Alustetaan Master FSM tuon 6-vaiheisen syklin mukaisesti
        self.state = "IDLE"  # IDLE, ARMED, ACTION, MANAGE, EXIT, NEUTRAL
        self.symbol_data = {}

    def process_state(self, signal, tensor_valid, trade_active, risk_ok):
        """
        Vojker RDAAS FSM - 6-vaiheinen deterministinen tilakone.
        Aja tämä jokaisella tickillä / kynttilän sulkeutumisella.
        """
        
        # --- VAIHE 1: IDLE (Validoidaan datatensori) ---
        if self.state == "IDLE":
            if tensor_valid and signal == 0:
                self.state = "NEUTRAL"
            elif tensor_valid and signal in [1, 2]:
                self.state = "ARMED"

        # --- VAIHE 2: NEUTRAL (Odotustila, sydänääni / heartbeat) ---
        elif self.state == "NEUTRAL":
            if not tensor_valid:
                self.state = "IDLE"
            elif signal in [1, 2]:
                self.state = "ARMED"

        # --- VAIHE 3: ARMED (Lasketaan dynaamiset tasot & SL / Salaman laatikko) ---
        elif self.state == "ARMED":
            if not risk_ok or signal == 0:
                self.state = "NEUTRAL"
            elif risk_ok and signal in [1, 2]:
                # Tasot lukittu, siirrytään toimeksiantoon
                self.state = "ACTION"

        # --- VAIHE 4: ACTION (Toimeksiannon lähetys MT5:een) ---
        elif self.state == "ACTION":
            if trade_active:
                self.state = "MANAGE"
            else:
                # Jos toimeksianto epäonnistui tai hylättiin, palataan turvaan
                self.state = "NEUTRAL"

        # --- VAIHE 5: MANAGE (Kaupan hallinta, trailing, TP/SL valvonta) ---
        elif self.state == "MANAGE":
            if not trade_active:
                self.state = "EXIT"
            elif not risk_ok:
                # Riskilimiitti ylittyi livenä -> Pakko-exit (Kill Switch)
                self.state = "EXIT"

        # --- VAIHE 6: EXIT (Positio suljettu, varmistetaan SHA256 audit-grade tila) ---
        elif self.state == "EXIT":
            if not trade_active:
                # Sykli valmis, palataan lähtöpisteeseen puhtaalta pöydältä
                self.state = "IDLE"

        return self.state

    def get_current_stage_description(self):
        """Palauttaa alitilan tarkistuksen debuggausta varten"""
        descriptions = {
            "IDLE": "Validate Tensor Integrity",
            "NEUTRAL": "Monitor Heartbeat / Pulse",
            "ARMED": "Calculate Salaman Laatikko & Dynamic SL",
            "ACTION": "Submit Order to MT5 Bridge",
            "MANAGE": "Monitor Risk Limits & Position",
            "EXIT": "Verify Audit-Grade Close & Side-Effects"
        }
        return descriptions.get(self.state, "Unknown State")
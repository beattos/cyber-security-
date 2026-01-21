from dataclasses import dataclass

@dataclass
class Decision:
    verdict: str   # "ALERT" | "REVIEW" | "PASS"
    p_malware: float

def decide(p_malware: float, t_alert: float = 0.80, t_review: float = 0.55) -> Decision:
    if p_malware >= t_alert:
        return Decision("ALERT", p_malware)
    if p_malware >= t_review:
        return Decision("REVIEW", p_malware)
    return Decision("PASS", p_malware)

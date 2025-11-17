# Lab 1 — Cyber Threat Intelligence: Mapping to Tactics & Techniques

**Report source:** Microsoft — *AI vs AI: Detecting an AI-obfuscated phishing campaign*  
🔗 https://www.microsoft.com/en-us/security/blog/2025/09/24/ai-vs-ai-detecting-an-ai-obfuscated-phishing-campaign/

---

## Short Summary
The phishing campaign analyzed in Microsoft’s “AI vs AI: Detecting an AI-obfuscated phishing campaign”
employed a social-engineering strategy consistent with MITRE ATT&CK patterns for *Initial Access* and *Credential Access*.
Attackers distributed self-addressed emails using the BCC field to hide the mass distribution of identical messages,
creating the illusion of one-to-one communication. Each email contained an attachment named to resemble a PDF but
actually formatted as an SVG file embedding heavily obfuscated, AI-generated code.
When the file was opened, the SVG redirected the recipient to a counterfeit CAPTCHA-style webpage that harvested credentials.
This activity maps to Initial Access – Phishing: Spearphishing Link (**T1566.002**) and Spearphishing Attachment (**T1566.001**),
followed by Execution – User Execution: Malicious Link (**T1204.001**), Defense Evasion – Obfuscated Files or Information – SVG Smuggling (**T1027.017**),
and Credential Access – Phishing for Information (**T1566**). Microsoft Defender for Office 365 detected the attack
through AI-based behavioral analysis and sandboxing, identifying anomalously complex parameters and code structures that were statistically unlikely to be human-authored.
This case demonstrates how offensive AI can automate and obscure phishing payloads, while defensive AI can learn to detect such synthetic patterns
through anomaly and entropy analysis.

---

## Included Files
- `evidence_map.md` — quotes from the article mapped to MITRE techniques  
- `mitre_mapping_table.md` — tactics → techniques (ID) → short mitigation  
- `sources.bib` — citations  
- `assets/screenshot.png` — image used for mapping  

**Group members:** Daniel Buts, Daniel Ziv

---

# Evidence Map — Quotes → MITRE Mapping

| # | Article quote | Observed behavior | MITRE Technique (ID) | Justification |
|--:|:--------------|:------------------|:----------------------|:---------------|
| 1 | "Self-addressed email with BCC recipients" | Mass delivery hidden from recipients | Spearphishing Link/Attachment (T1566.002 / T1566.001) | BCC hides recipient list → spearphishing Initial Access. |
| 2 | "Attachment named like a PDF but formatted as SVG" | Attachment disguises payload as image | Obfuscated Files or Information – SVG Smuggling (T1027.017) | SVG contains encoded payload in image format. |
| 3 | "SVG redirected to a fake CAPTCHA page" | Credential harvesting via web page | Phishing for Information (T1566) | Redirect to login form collecting credentials. |
| 4 | "Microsoft detected anomalously complex, AI-generated parameters" | AI-generated obfuscation detected by defensive AI | Obfuscated Files or Information (T1027) + AI indicator | Detection flagged synthetic structure and entropy. |

---

# MITRE Mapping Table

| Tactic | Technique (ID) | Short Description | Mitigation |
|:--|:--:|:--|:--|
| Initial Access | Spearphishing: Link (T1566.002) | Targeted email with malicious link | Sandbox links, verify SPF/DKIM |
| Initial Access | Spearphishing: Attachment (T1566.001) | Weaponized attachment disguised as PDF | Scan attachments, block macros |
| Execution | User Execution: Malicious Link (T1204.001) | User clicks link → fraud site | Click-time analysis, user training |
| Defense Evasion | Obfuscated Files – SVG Smuggling (T1027.017) | Hidden payload inside SVG | Sanitize/rasterize SVGs |
| Credential Access | Phishing for Information (T1566) | Fake CAPTCHA page collects creds | MFA + phishing site blocking |

---

# Sources

```bibtex
@misc{microsoft2025ai_vs_ai,
  author = {Microsoft Threat Intelligence},
  title = {AI vs AI: Detecting an AI-obfuscated phishing campaign},
  year = {2025},
  howpublished = {\url{https://www.microsoft.com/en-us/security/blog/2025/09/24/ai-vs-ai-detecting-an-ai-obfuscated-phishing-campaign/}},
  note = {Accessed: 2025-11-10}
}
```

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

### Included Files
- `evidence_map.md` — quotes from the article mapped to MITRE techniques  
- `mitre_mapping_table.md` — tactics → techniques (ID) → short mitigation  
- `sources.bib` — citations  
- `assets/screenshot.png` — image used for mapping  

**Group members:** Daniel Buts

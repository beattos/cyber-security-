# MITRE Mapping Table

| Tactic | Technique (ID) | Short Description | Mitigation |
|:--|:--:|:--|:--|
| Initial Access | Spearphishing: Link (T1566.002) | Targeted email with malicious link | Sandbox links, verify SPF/DKIM |
| Initial Access | Spearphishing: Attachment (T1566.001) | Weaponized attachment disguised as PDF | Scan attachments, block macros |
| Execution | User Execution: Malicious Link (T1204.001) | User clicks link → fraud site | Click-time analysis, user training |
| Defense Evasion | Obfuscated Files – SVG Smuggling (T1027.017) | Hidden payload inside SVG | Sanitize/rasterize SVGs |
| Credential Access | Phishing for Information (T1566) | Fake CAPTCHA page collects creds | MFA + phishing site blocking |

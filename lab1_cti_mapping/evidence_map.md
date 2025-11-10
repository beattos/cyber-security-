# Evidence Map — Quotes → MITRE Mapping

| # | Article quote | Observed behavior | MITRE Technique (ID) | Justification |
|--:|:--------------|:------------------|:----------------------|:---------------|
| 1 | "Self-addressed email with BCC recipients" | Mass delivery hidden from recipients | Spearphishing Link/Attachment (T1566.002 / T1566.001) | BCC hides recipient list → spearphishing Initial Access. |
| 2 | "Attachment named like a PDF but formatted as SVG" | Attachment disguises payload as image | Obfuscated Files or Information – SVG Smuggling (T1027.017) | SVG contains encoded payload in image format. |
| 3 | "SVG redirected to a fake CAPTCHA page" | Credential harvesting via web page | Phishing for Information (T1566) | Redirect to login form collecting credentials. |
| 4 | "Microsoft detected anomalously complex, AI-generated parameters" | AI-generated obfuscation detected by defensive AI | Obfuscated Files or Information (T1027) + AI indicator | Detection flagged synthetic structure and entropy. |

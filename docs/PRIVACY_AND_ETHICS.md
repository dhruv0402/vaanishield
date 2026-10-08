# Privacy & Ethics Constraints — VaaniShield

These constraints are **non-negotiable** for anyone (human or AI coding agent) building out
this repository. They exist because this project necessarily involves cloning real people's
voices, which is legally and ethically sensitive even for research purposes.

## 1. Voice is personal (and biometric-adjacent) data under India's DPDPA

Every recording collected or generated must have a lawful basis — in practice, explicit
informed consent from the speaker. This applies to:
- Bona-fide (genuine) recordings used as training/eval data.
- Any source audio used to generate a cloned voice.

**Never scrape voice audio from YouTube, Instagram, or any other public source to use as a
cloning source.** Even if technically accessible, using it to build a voice clone without the
speaker's explicit consent is both an ethical and likely legal violation.

## 2. Consent must specifically cover cloning, not just recording

A generic "I consent to be recorded for research" is not sufficient. Use the template in
`data/consent_forms/` and ensure it explicitly covers:
- Being recorded.
- Having synthetic (cloned) versions of their voice generated from that recording.
- Those synthetic clips being used in model training/evaluation and potentially shown in a
  demo, paper, or presentation.

## 3. Respect the license terms of voice-cloning tools

Tools like Coqui XTTS, Bark, and OpenVoice generally include ethical-use clauses prohibiting
impersonation of real people without consent, and prohibiting fraud-enabling use. Read each
tool's license before use. `data/scripts/clone_voices.py` must only ever be run against
consented source speakers — never against arbitrary audio.

## 4. Never use real fraud-call recordings

Even if real scam-call audio becomes available (e.g. from a bank's fraud desk or a public
awareness campaign), do not use recordings of real victims or real fraudsters without their
consent. Construct synthetic scam-call scenarios instead (scripted, acted, or model-generated
scam dialogue) — this is what `data/scripts/generate_codeswitch_scripts.py` is for.

## 5. If this ever moves past a research prototype

A telecom-side or bank-side deployment that listens to live calls would fall under India's
lawful-interception framework (Telegraph Act / IT Act). That kind of deployment must be done
by the telecom/bank under their own lawful-interception authority — this repository is a
research prototype and is not built or licensed for that use. Do not add live-call
interception capability against arbitrary third-party numbers.

## 6. The explainability layer must not leak conversation content

`vaanishield/fusion/explain.py` must produce explanations in terms of **acoustic patterns**
(e.g. "unnatural pitch discontinuity at the language-switch point around 4.2s") and must
never transcribe, quote, or otherwise expose what the caller actually said. The point of the
explanation is to justify the fraud-risk score, not to surface conversation content.

## 7. Storage & retention

- Signed consent forms are personal data — keep them local (`data/consent_forms/signed/`,
  gitignored) and never commit them to version control.
- Raw and processed audio directories are gitignored by default (`data/raw/`,
  `data/processed/`) — do not remove those `.gitignore` entries.
- Any demo deployment should not persist uploaded call audio beyond the session unless the
  uploader has explicitly consented to storage.

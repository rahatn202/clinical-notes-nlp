# Labeling guide

These rules define the "correct answer" for each note. The LLM prompt (`prompts/v1.md`) uses the same rules, so the LLM and the human labeler work from one definition.

Label what the note **says**, not what is likely. If something isn't stated, choose "not stated" (or leave it empty).

## Anesthesia type (choose all that apply)

| Value | Use when the note describes |
|---|---|
| general | general anesthesia, including endotracheal or LMA anesthesia, TIVA (total intravenous anesthesia) |
| mac_sedation | monitored anesthesia care (MAC), IV sedation, conscious sedation |
| spinal | spinal or subarachnoid anesthesia |
| epidural | epidural or caudal anesthesia |
| regional_block | nerve blocks and other regional anesthesia, including Bier block |
| local | local anesthetic or local infiltration (lidocaine, Marcaine and similar, injected at the site) |
| topical | topical anesthesia |

- Choose more than one when the note says so: "local with IV sedation" = local + mac_sedation.
- General anesthesia plus "local infiltrated at the incision for postoperative pain" = general + local.
- **Do not** count anesthesia mentioned only as a risk ("anesthetic risks were discussed"), a plan, or the patient's history.
- If no anesthesia is described, leave it empty.

## Airway device

endotracheal_tube (also orotracheal, "intubated"), lma (laryngeal mask airway), nasotracheal, mask (face mask), or not_stated.

## Estimated blood loss

- Write the number in mL. 1 cc = 1 mL; 1 liter = 1000 mL.
- "Less than 50 mL" → 50. "None" → 0.
- "Minimal", "negligible" or "scant" with no number → leave the number empty and tick **minimal**.
- Not mentioned → leave the number empty and don't tick minimal.

## Complications

- **none**: the note says there were no complications.
- **any**: the note describes a complication of the procedure.
- **not_stated**: complications are not mentioned.

## When unsure

Tick **unsure** and write a short comment. Unsure notes stay in the evaluation, and the report lists how many there are.

## Scop general

Construirea unui **sistem multi-container Docker**, orientat pe **personalizarea conținutului (știri)** folosind **două LLM-uri în lanț**, pornind de la o descriere liberă a utilizatorului.

---

## Componentele sistemului

### 1. Container UI (Frontend)

* O interfață simplă (web UI).
* Utilizatorul introduce o **descriere personală** (interese, preferințe, stil, domenii de interes etc.).
* UI-ul:

  * Trimite inputul utilizatorului către **LLM1**.
  * Primește la final răspunsul procesat (știrea personalizată) de la **LLM2** și îl afișează.

---

### 2. Container LLM1 (Profilare utilizator)

* Primește **descrierea utilizatorului** de la UI.
* Rolul său este de **interpretare / abstractizare**:

  * Generează o descriere structurată sau semi-structurată despre:

    * ce i-ar putea plăcea utilizatorului,
    * ce tip de conținut îl interesează,
    * ce ton / nivel de detaliu este potrivit.
* Output-ul acestui container este un **text de tip profil**.
* Acest output este trimis mai departe către **LLM2** și va fi folosit drept **`SYS_PROMPT`**.

---

### 3. Container LLM2 (Procesare știri)

* Primește două tipuri de input:

  1. **SYS_PROMPT** → generat din output-ul LLM1 (profilul utilizatorului).
  2. **Prompt de utilizator / input extern** → un set de știri brute (text).
* Rolul său:

  * Să proceseze știrile **strict în baza profilului utilizatorului**.
  * Să selecteze, rezume, reformuleze sau prioritizeze informația astfel încât:

    * conținutul să fie relevant pentru utilizator,
    * stilul și accentul să respecte SYS_PROMPT-ul primit.
* Output-ul final este trimis înapoi către **UI**.

---

## Fluxul de date (end-to-end)

1. Utilizator → UI
2. UI → LLM1 (descriere utilizator)
3. LLM1 → UI / direct LLM2 (profil generat)
4. UI / orchestrator → LLM2 (SYS_PROMPT + știri)
5. LLM2 → UI (știre personalizată)
6. UI → Utilizator (afișare rezultat)

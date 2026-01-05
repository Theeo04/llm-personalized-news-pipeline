# Pipeline de știri personalizate (cu 2 LLM-uri)

## 1. Cum funcționează aplicația (explicație simplă)

Ideea este să pornești de la **o descriere liberă a ta** și să ajungi la **un rezumat de știri făcut pe gustul tău**.

1. **Tu scrii despre tine**  
   În UI (o pagină web simplă) scrii cine ești, ce te interesează, ce nu-ți place, ce ton preferi (mai serios, mai relaxat etc.).

2. **Primul LLM îți construiește un „profil”**  
   - LLM1 citește descrierea ta.
   - Din ea construiește un **profil structurat**, adică un text care spune clar:
     - ce îți place,
     - ce vrei să eviți,
     - ce ton preferi,
     - cât de detaliate să fie explicațiile.
   - Acest profil este folosit mai departe ca **instrucțiune de sistem (SYS_PROMPT)** pentru al doilea LLM.

3. **Serviciul de știri pregătește o listă de articole**  
   - Un mic serviciu numit `news` ține o listă de știri (de test).
   - Gateway-ul ia aceste știri și le transformă într-un text compact, ușor de citit de LLM.

4. **Al doilea LLM personalizează știrile pentru tine**  
   - LLM2 primește:
     - **profilul tău** ca instrucțiune de sistem (SYS_PROMPT),
     - **știrile** ca text de intrare (prompt).
   - Pe baza profilului, LLM2:
     - alege ce știri sunt relevante pentru tine,
     - le explică pe scurt,
     - păstrează tonul și nivelul de detaliu potrivite.

5. **UI îți afișează rezultatul**  
   - UI primește de la gateway textul final generat de LLM2.
   - Îți afișează sumarul personalizat de știri, plus ceva informații de debug (opțional).

Pe scurt:  
**Tu descrii ce îți place → LLM1 te „traduce” într-un profil clar → LLM2 folosește profilul ca regulă de lucru și filtrează știrile pentru tine.**

---

## 2. Cum este construită aplicația (partea practică)

Aplicația este formată din mai multe containere Docker care vorbesc între ele pe o rețea internă (`pnp-net`). Fiecare container are un rol clar.

### 2.1. Servicii folosite

1. **UI (container `ui`)**
   - Scris cu **Streamlit** (Python).
   - Rulează un mic server web (portul 8501 în Docker, mapat pe `localhost:8501`).
   - Trimite cereri HTTP către `gateway` la endpoint-ul:  
     `POST http://gateway:8000/process`
   - Primește răspunsul (știrile personalizate) și îl afișează în browser.

2. **Gateway (container `gateway`)**
   - Scris cu **FastAPI** (Python).
   - Este „creierul” de orchestrare:
     - primește descrierea ta de la UI,
     - cheamă **LLM1** pentru a construi profilul (SYS_PROMPT),
     - ia știrile de la serviciul `news`,
     - cheamă **LLM2** cu:
       - `system_prompt = profilul de la LLM1`
       - `user_prompt = știrile brute`
     - returnează către UI rezultatul LLM2.
   - Comunică prin HTTP (`requests`) cu:
     - `news` → `GET http://news:8080/news`
     - `llm1` → `POST http://llm1:11434/api/generate`
     - `llm2` → `POST http://llm2:11434/api/generate`

3. **Serviciul de știri (container `news`)**
   - Scris cu **FastAPI**.
   - Ține o listă de știri predefinite (mock de știri reale).
   - Expune endpoint-uri:
     - `GET /health` – verificare rapidă.
     - `GET /news` – întoarce știrile curente (într-un JSON simplu).
     - `POST /refresh` – reîncarcă lista de știri (tot predefinite în cod).
   - Folosește un volum (`news-data`) pentru a salva un fișier JSON cu cele mai recente știri.

4. **LLM1 și LLM2 (containere `llm1` și `llm2`)**
   - Ambele folosesc imaginea **`ollama/ollama:latest`**.
   - Se bazează pe modelul `llama3.2:3b` (descărcat în volumul lor `ollama-llm1`, `ollama-llm2`).
   - Gateway-ul le apelează prin API-ul Ollama:
     - `/api/generate` – pentru generarea de text.

---

## 3. Cum comunică serviciile între ele

Toate containerele sunt definite în `docker-compose.yaml` și se află în aceeași rețea Docker `pnp-net`.

### 3.1. Flux de la utilizator la rezultat

1. **Browser-ul tău** (pe calculatorul local):
   - Deschizi `http://localhost:8501` → interfața Streamlit (`ui`).

2. **UI → Gateway**
   - UI trimite o cerere:
     - `POST http://gateway:8000/process`
     - body JSON: `{ "user_description": "...", "news_blob": null, "options": {...} }`

3. **Gateway → LLM1**
   - Gateway ia `user_description`.
   - Construiește un prompt special pentru LLM1 (instrucțiuni pentru a construi profilul).
   - Apelează:
     - `POST http://llm1:11434/api/generate`
   - Rezultatul este profilul de utilizator (SYS_PROMPT).

4. **Gateway → News**
   - Dacă `news_blob` nu este trimis de UI, gateway ia știrile din serviciul:
     - `GET http://news:8080/news`
   - Transformă JSON-ul de știri într-un text relativ compact (`news_blob`).

5. **Gateway → LLM2**
   - Apelează:
     - `POST http://llm2:11434/api/generate`
   - Cu:
     - `system = profilul generat de LLM1`
     - `prompt = news_blob` (știrile)
   - Primește înapoi textul cu știrile personalizate.

6. **Gateway → UI → Utilizator**
   - Gateway trimite răspunsul către UI.
   - UI îl afișează în chat, plus metadate (opțional) într-un „Debug meta”.

---

## 4. Diagramă simplă (textuală)

```text
[ Tu, în browser ]
        |
        |  (descriere utilizator)
        v
[ UI (Streamlit, container `ui`) ]
        |
        |  POST /process
        v
[ Gateway (FastAPI, container `gateway`) ]
        |\
        | \__ cheamă LLM1 pentru profil
        |      (descriere utilizator)
        |        |
        |        v
        |   [ LLM1 (Ollama) ]
        |        |
        |        |  profil utilizator (SYS_PROMPT)
        |        v
        |  folosește profil + cere știri
        |        |
        |        v
        |   [ News service ]
        |        |
        |        |  JSON știri
        |        v
        |  transformă în `news_blob`
        |        |
        |        v
        |   cheamă LLM2
        |        |
        |        v
        |   [ LLM2 (Ollama) ]
        |        |
        |        |  știri personalizate
        |        v
        +------> răspuns final pentru UI
                   |
                   v
           [ UI afișează rezultatul ]
```

---

## 5. Note rapide utile

- Pornire (din directorul proiectului):

  ```bash
  docker compose up -d
  ```

- Verificare modele în LLM1:

  ```bash
  docker exec -it llm1 ollama list
  ```

- Acces UI:

  ```text
  http://localhost:8501
  ```

- Endpoint-uri utile:
  - Gateway health: `GET http://localhost:8000/health`
  - News health: `GET http://localhost:8080/health`
  - News data: `GET http://localhost:8080/news`

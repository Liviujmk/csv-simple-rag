# Simple RAG implementation to chat over a CSV file of vehicles informations

## How to use?

**1. Clone this repo**
```sh
git clone https://github.com/Liviujmk/csv-simple-rag.git
```

**2. Create python environment**
```sh
py -m venv venv
```

**3. Activate created environment**
```sh
WINDOWS(powershell): venv\Scripts\activate
LINUX: venv/Scripts/activate
```

**4. Install required packages**
```sh
pip install -r requirements.txt
```

**5. Create an env file in root directory with OPENAI_API_KEY**
```sh
OPENAI_API_KEY=...
```

**6. Run project in dev mode (and wait)**
```sh
fastapi dev api.py
```
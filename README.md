# 🌀 Zero Gap System Tool

Welcome to the **Zero Gap System** — a real-time options scanner that pairs high-efficiency **strangles** with precision-tuned **credit spreads** to extract premium in a modern, modular, and maybe-too-stylish Python app.
 
> — *Tyler Jacobi*

---

## 📈 What It Does

This Streamlit-powered tool scans **live options chains** (via Tradier API) and ranks combo trades based on:

- ✅ Risk-reward ratios for **credit spreads**
- ✅ Delta skew & premium efficiency for **strangles**
- ✅ Custom filter sliders for nerd-level control

And yes — you can even one-click **send trades to IBKR paper** (via `ib_insync`).  
Because why manually place 4-leg combo tickets when you can just ✨click✨?

---

## 🧠 Why It's Called "Zero Gap"

Because we don't do wide-leg inefficiency.  
The goal is **tight spreads + tight logic** — near or overlapping strikes with capital efficiency in mind.

Like an iron condor but cooler.  
Like a butterfly, but... less fluttery.

---

## ⚙️ How to Run It Locally

```bash
# Clone this repo
git clone https://github.com/tylerjacobi1/zero-gap-system.git
cd zero-gap-system

# (Optional) Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run main.py

# Benchmarking platform for Federated Learning
## Set up the project
### Install dependencies and project
⚠️ **WARNING:** : The point of this branch is to better simulate Federated Learning. As such, every client uses a separate node, choosing too many clients may lead to issues. We recommend using at most one less client than you have node, since the server also uses a node. So if you have 10 nodes, you can select 9 clients at most.

This branch is meant to be used on a cluster. 
Connect to your cluster using the ssh command, then open a tunnel linking your 8501 port to the cluster's 8501 port.

Install the dependencies defined in `requirements.txt` as well as the `pyproject.toml` file.
Use Python 3.13.5

```bash
pip install -r requirements.txt
```

## Run the project

```bash
streamlit run app.py
```
Open http:localhost:8501/ in your browser.

⚠️ **WARNING:** : Files must be contained in a folder named Green-FL-Project
